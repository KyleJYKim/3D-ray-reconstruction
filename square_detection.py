import numpy as np
import cv2 as cv
import json

NMS_WINSIZE = 13
H_SIZE = 512

WIDTH = 250
HEIGHT = 150


def order_points(pts):
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    Works by sorting on sum (TL has smallest, BR has largest) and
    difference (TR has smallest diff, BL has largest diff).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    return ordered


def detect_rectangle(frame):
    frame_gray = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
    frame_blur = cv.GaussianBlur(frame_gray, (5, 5), 0)
    _, frame_thresh = cv.threshold(frame_blur, 10, 255, cv.THRESH_BINARY)

    edges = cv.Canny(frame_thresh, 100, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    rects = []
    vis_frame = cv.cvtColor(frame_thresh, cv.COLOR_GRAY2BGR)

    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        rect_approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(rect_approx) == 4:
            # squeeze (4,1,2) to (4,2) and order the corners
            rect_ordered = order_points(rect_approx)
            rects.append(rect_ordered)

    # top and bottom rectangles
    if len(rects) != 2:
        return None

    # Sort by the maximum y-value so index0 = top rect, index1 = bottom rect
    rects_sorted = sorted(rects, key=lambda r: r[:, 1].max())

    # Visualisation
    # cv.drawContours(vis_frame, contours, -1, (0, 255, 0), 3)
    cv.polylines(vis_frame, [rects_sorted[0].reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 3)
    cv.polylines(vis_frame, [rects_sorted[1].reshape(-1, 1, 2).astype(np.int32)], True, (0, 0, 255), 3)
    cv.imshow("contour", vis_frame)
    # cv.waitKey(1)

    return rects_sorted   # list of two (4,2) float32 arrays


def load_calibration_data(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if "K" in data and "distortion" in data:
            dist = np.array(data["distortion"], dtype=np.float64).ravel()
            K = np.array(data["K"], dtype=np.float64)
        else:
            return None, None
    except FileNotFoundError:
        return None, None
    return K, dist


def decompose_homography(H, K):
    """
    Decompose H = K * [r1 | r2 | t]  ->  R (3x3), t (3,).
    Returns None if K is unavailable (falls back to raw H columns).
    """
    if K is not None:
        # Normalise: Hn = K^{-1} H
        Hn = np.linalg.inv(K) @ H
    else:
        Hn = H.copy()

    # Columns of the normalised homography
    r1 = Hn[:, 0]
    r2 = Hn[:, 1]
    t  = Hn[:, 2]

    # Recover scale from the two rotation columns (should both have unit norm)
    scale = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
    r1 /= scale
    r2 /= scale
    t  /= scale

    # r3 must be orthogonal to r1 and r2
    r3 = np.cross(r1, r2)

    # Build rotation matrix and project onto SO(3) via SVD
    R_raw = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:   # guard against reflection
        R = U @ np.diag([1, 1, -1]) @ Vt

    return R, t

def detect_laser_points(frame, rect_top_img, rect_btm_img):
    """
    Detect red laser points that fall inside either rectangle and return
    them as homogeneous image coordinates (x, y, 1).
 
    Strategy
    --------
    1. Isolate red channel via HSV masking (handles both low- and
       high-hue red which wraps around 180° in OpenCV).
    2. Build a binary ROI mask from the two rectangle polygons so we
       only look inside the calibration targets.
    3. For each contour in the masked result keep only bright, compact
       blobs (laser spots are small and very saturated).
    4. Return the sub-pixel centroid of every accepted blob as (x, y, 1).
    """
 
    # --- 1. Red HSV mask (red wraps: [0,10] ∪ [170,180]) ---
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 
    mask_red1 = cv.inRange(hsv, (0,   50, 50), (10,  255, 255))
    mask_red2 = cv.inRange(hsv, (170, 50, 50), (180, 255, 255))
    mask_red  = cv.bitwise_or(mask_red1, mask_red2)
 
    # Small morphological cleanup to remove single-pixel noise
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # mask_red = cv.morphologyEx(mask_red, cv.MORPH_OPEN,  kernel)
    # mask_red = cv.morphologyEx(mask_red, cv.MORPH_CLOSE, kernel)

    # --- 2. ROI mask: only pixels inside the two rectangles ---
    h, w = frame.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
 
    for rect in (rect_top_img, rect_btm_img):
        pts = rect.reshape(-1, 1, 2).astype(np.int32)
        cv.fillPoly(roi_mask, [pts], 255)
 
    mask_combined = cv.bitwise_and(mask_red, roi_mask)
 
 
    contours, _ = cv.findContours(mask_combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # --- 3. Find blobs, compute sub-pixel centroids, assign to top/btm rect ---
    # Pre-build the contour arrays once for pointPolygonTest
    poly_top = rect_top_img.reshape(-1, 1, 2).astype(np.int32)
    poly_btm = rect_btm_img.reshape(-1, 1, 2).astype(np.int32)

    pts_top = []   # homogeneous (x, y, 1) inside top rectangle
    pts_btm = []   # homogeneous (x, y, 1) inside bottom rectangle

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 2:    # skip single-pixel noise
            continue
        if area > 500:  # skip large blobs that are not laser spots
            continue

        M = cv.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        p  = np.array([cx, cy, 1.0], dtype=np.float64)

        # pointPolygonTest returns >= 0 when the point is on or inside the polygon
        if cv.pointPolygonTest(poly_top, (cx, cy), measureDist=False) >= 0:
            pts_top.append(p)
        elif cv.pointPolygonTest(poly_btm, (cx, cy), measureDist=False) >= 0:
            pts_btm.append(p)
        # points outside both rectangles are discarded

    # --- 4. Visualisation ---
    vis = frame.copy()
    cv.polylines(vis, [poly_top], True, (0, 255, 0), 2)   # green = top (wall)
    cv.polylines(vis, [poly_btm], True, (0, 0, 255), 2)   # blue  = bottom (table)
    for p in pts_top:
        cv.circle(vis, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)   # yellow
    for p in pts_btm:
        cv.circle(vis, (int(p[0]), int(p[1])), 5, (255, 0, 255), -1)   # magenta
    cv.imshow("laser points", vis)

    return pts_top, pts_btm   # two lists of np.array([x, y, 1])



def ray_plane_intersect(pt_h, K, plane_pt, plane_n):
    """
    Back-project homogeneous image point pt_h = (x, y, 1) and intersect
    with a plane defined by point plane_pt and unit normal plane_n.

    Ray from camera origin: d = K^{-1} * pt_h  (normalised)
    Intersection:           t = (plane_pt . plane_n) / (d . plane_n)
    3D point:               X = t * d
    """
    d = np.linalg.inv(K) @ pt_h
    d = d / np.linalg.norm(d)

    plane_n   = plane_n.ravel()
    plane_pt  = plane_pt.ravel()
    denom     = d @ plane_n

    if abs(denom) < 1e-9:   # ray parallel to plane
        return None
    t = (plane_pt @ plane_n) / denom
    if t < 0:               # intersection behind camera
        return None
    return t * d            # (3,) in camera coordinates


def fit_plane_svd(points_3d: list) -> tuple:
    """
    Fit a plane to >= 3 3D points using SVD (least-squares normal).

    Returns (centroid, unit_normal) where the plane is:
        (X - centroid) . unit_normal = 0

    The normal is the right-singular vector corresponding to the
    smallest singular value, i.e. the direction of least variance.
    """
    pts = np.array(points_3d, dtype=np.float64)   # (N, 3)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]                               # last row = smallest singular value
    normal = normal / np.linalg.norm(normal)
    return centroid, normal


def intersect_planes(p1, n1, p2, n2) -> tuple:
    """
    Intersect two planes to get a 3D line.

    Plane 1: (X - p1) . n1 = 0
    Plane 2: (X - p2) . n2 = 0

    The line direction is d = n1 x n2.
    A point on the line is found by solving the 2-plane system
    with a third constraint (set the component along d to zero).

    Returns (point_on_line, direction) or (None, None) if planes are parallel.
    """
    d = np.cross(n1, n2)
    if np.linalg.norm(d) < 1e-9:
        return None, None           # planes are parallel
    d = d / np.linalg.norm(d)

    # Solve for a point on the line:
    # Build 3x3 system [n1; n2; d^T] * x = [n1.p1; n2.p2; 0]
    A = np.array([n1, n2, d], dtype=np.float64)
    b = np.array([n1 @ p1, n2 @ p2, 0.0], dtype=np.float64)
    pt = np.linalg.solve(A, b)
    return pt, d


def main():
    K, dist = load_calibration_data("./intrinsics.json")

    analysis_vid = cv.VideoCapture("./data/cup1.mp4")

    # FIX 3: rect_canonical uses only x,y and must be float32
    rect_canonical = np.array([[0, 0, 0], [WIDTH, 0, 0], [WIDTH, HEIGHT, 0], [0, HEIGHT, 0]], dtype=np.float32)

    rect_H_top = None
    rect_H_btm = None

    # Simple namespace to accumulate 3D points across frames
    class run_state: pass

    running = True

    while running:
        keyp = cv.waitKey(1)
        running = keyp != ord('q')

        success, frame = analysis_vid.read()
        if not success:
            break

        if K is not None and dist is not None:
            frame = cv.undistort(frame, K, dist)

        # Detect only until we have both homographies
        if rect_H_top is None or rect_H_btm is None:
            
            print("Started the parametrization of Plane 1 and 2")
            
            rects = detect_rectangle(frame)
            if rects is None:
                print("Waiting for two rectangles…")
                continue

            rect_top_img, rect_btm_img = rects

            # findHomography needs (N,1,2) or (N,2); the arrays are (4,2)
            rect_H_top, mask_top = cv.findHomography(rect_top_img, rect_canonical)
            rect_H_btm, mask_btm = cv.findHomography(rect_btm_img, rect_canonical)

            print("H_top:\n", rect_H_top)
            R_top, t_top = decompose_homography(rect_H_top, K)
            print("R_top:\n", R_top)
            print("t_top:\n", t_top)

            print("H_btm:\n", rect_H_btm)
            R_btm, t_btm = decompose_homography(rect_H_btm, K)
            print("R_btm:\n", R_btm)
            print("t_btm:\n", t_btm)
            
            # the point and normal of the plane
            p_top = t_top
            n_top = R_top @ np.array([[0],[0],[1]])
            p_btm = t_btm
            n_btm = R_btm @ np.array([[0],[0],[1]])
            
            print("p_top:\n", p_top)
            print("n_top:\n", n_top)
            print("p_btm:\n", p_btm)
            print("n_btm:\n", n_btm)
            
            print("Completed the parametrization of Plane 1 (top) and 2 (bottom)")
        
        print("Started the parametrization of Laser Plane")

        laser_pts_top, laser_pts_btm = detect_laser_points(frame, rect_top_img, rect_btm_img)

        print(f"Top rect: {len(laser_pts_top)} point(s)  {[p[:2].tolist() for p in laser_pts_top]}")
        print(f"Bottom rect: {len(laser_pts_btm)} point(s)  {[p[:2].tolist() for p in laser_pts_btm]}")

        if K is None:
            continue

        # --- Back-project 2D laser points to 3D via ray-plane intersection ---
        pts3d_top = []
        pts3d_btm = []
        
        for p2d in laser_pts_top:
            p3d = ray_plane_intersect(p2d, K, p_top, n_top)
            if p3d is not None:
                pts3d_top.append(p3d)

        for p2d in laser_pts_btm:
            p3d = ray_plane_intersect(p2d, K, p_btm, n_btm)
            if p3d is not None:
                pts3d_btm.append(p3d)

        print(f"3D top: {len(pts3d_top)} pt(s)")
        print(f"3D btm: {len(pts3d_btm)} pt(s)")

        # --- Accumulate across frames until we have enough points ---
        if not hasattr(run_state, 'acc_top'):
            run_state.acc_top = []
            run_state.acc_btm = []

        run_state.acc_top.extend(pts3d_top)
        run_state.acc_btm.extend(pts3d_btm)

        # Need >= 3 non-collinear points on each plane to fit the laser plane
        if len(run_state.acc_top) < 3 or len(run_state.acc_btm) < 3:
            print(f"Accumulating... top={len(run_state.acc_top)}/3  btm={len(run_state.acc_btm)}/3")
            continue

        # --- Fit laser plane through all accumulated 3D points ---
        all_pts3d = run_state.acc_top + run_state.acc_btm
        laser_centroid, laser_normal = fit_plane_svd(all_pts3d)
        print(f"Laser plane normal: {laser_normal}")
        print(f"Laser plane point: {laser_centroid}")

        # --- Intersect laser plane with each calibration plane -> laser line ---
        line_pt_top, line_dir_top = intersect_planes(
            laser_centroid, laser_normal, p_top.ravel(), n_top.ravel()
        )
        line_pt_btm, line_dir_btm = intersect_planes(
            laser_centroid, laser_normal, p_btm.ravel(), n_btm.ravel()
        )

        if line_pt_top is not None:
            print(f"  Laser line on wall  : point={line_pt_top}  dir={line_dir_top}")
        if line_pt_btm is not None:
            print(f"  Laser line on table : point={line_pt_btm}  dir={line_dir_btm}")


    analysis_vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()