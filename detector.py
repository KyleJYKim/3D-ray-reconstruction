import numpy as np
import cv2 as cv
import json

LASER_THRESH = 30.0   # minimum column red-excess sum to count as laser
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
    Detect red laser points across the whole frame and classify each pixel
    into one of three buckets:

      pts_top  — inside the top (wall) calibration rectangle
      pts_btm  — inside the bottom (table) calibration rectangle
      pts_obj  — inside the object region between the two rectangles

    All points are returned as homogeneous image coordinates (x, y, 1).
    """

    # --- 1. Red HSV mask ---
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask_red = cv.inRange(hsv, (165, 50, 50), (175, 180, 180))
    mask_red = cv.medianBlur(mask_red, 7)

    # --- 2. Region polygons ---
    poly_top = rect_top_img.reshape(-1, 1, 2).astype(np.int32)
    poly_btm = rect_btm_img.reshape(-1, 1, 2).astype(np.int32)
    poly_obj_region = np.array([
        rect_top_img[2], rect_top_img[3],
        rect_btm_img[0], rect_btm_img[1],
    ], dtype=np.int32).reshape(-1, 1, 2)

    # --- 3. Mask per region and gather all non-zero pixels ---
    h, w = mask_red.shape

    def region_pts(poly):
        m = np.zeros((h, w), dtype=np.uint8)
        cv.fillPoly(m, [poly], 255)
        raw = cv.findNonZero(cv.bitwise_and(mask_red, m))
        if raw is None:
            return []
        return [np.array([float(p[0][0]), float(p[0][1]), 1.0]) for p in raw]

    pts_top = region_pts(poly_top)
    pts_btm = region_pts(poly_btm)
    pts_obj = region_pts(poly_obj_region)

    # --- 4. Visualisation ---
    vis = frame.copy()
    cv.polylines(vis, [poly_top], True, (0, 255, 0), 2)
    cv.polylines(vis, [poly_btm], True, (0, 0, 255), 2)
    cv.polylines(vis, [poly_obj_region], True, (255, 255, 0), 1)
    for p in pts_top[::4]:
        cv.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)
    for p in pts_btm[::4]:
        cv.circle(vis, (int(p[0]), int(p[1])), 2, (255, 0, 255), -1)
    for p in pts_obj[::4]:
        cv.circle(vis, (int(p[0]), int(p[1])), 2, (0, 165, 255), -1)
    cv.imshow("laser points", vis)

    return pts_top, pts_btm, pts_obj


def backproject_to_ray(pt, K):
    """
    Back-project homogeneous image point pt = (x, y, 1)
    Ray from camera origin: d_vec = K^{-1} * pt  (normalised)
    """
    d_vec = np.linalg.inv(K) @ pt
    d_vec = d_vec / np.linalg.norm(d_vec)

    return d_vec # (0_vec, d_vec)
    
def intersect_ray_plane(d_vec, plane_pt, plane_norm):
    """
    Intersect a ray with a plane defined by point plane_pt and unit normal plane_n.
    plane = (plane_pt, plane_norm)
    ray = (c, d_vec), where c = 0_vec
    Intersection: c + z @ d_vec,
    where z = ((plane_pt - c) @ plane_norm) / d_vec @ plane_norm.
    Therefore: z = (plane_pt @ plane_norm) / d_vec @ plane_norm
    """

    plane_pt = plane_pt.ravel()
    plane_norm = plane_norm.ravel()
    denom = d_vec @ plane_norm

    if abs(denom) < 1e-9:
        print("ray parallel to plane")
        return None
    
    z = (plane_pt @ plane_norm) / denom
    
    if z < 0:
        print("intersection behind camera")
        z = z * -1
        # return None
    
    return z * d_vec


def fit_plane_svd(points_3d):
    """
    Fit a plane to >= 3 3D points using SVD (least-squares normal).

    Returns (mean, unit_normal) where the plane is:
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


# def intersect_planes(p1, n1, p2, n2) -> tuple:
#     """
#     Intersect two planes to get a 3D line.

#     Plane 1: (X - p1) . n1 = 0
#     Plane 2: (X - p2) . n2 = 0

#     The line direction is d = n1 x n2.
#     A point on the line is found by solving the 2-plane system
#     with a third constraint (set the component along d to zero).

#     Returns (point_on_line, direction) or (None, None) if planes are parallel.
#     """
#     d = np.cross(n1, n2)
#     if np.linalg.norm(d) < 1e-9:
#         return None, None           # planes are parallel
#     d = d / np.linalg.norm(d)

#     # Solve for a point on the line:
#     # Build 3x3 system [n1; n2; d^T] * x = [n1.p1; n2.p2; 0]
#     A = np.array([n1, n2, d], dtype=np.float64)
#     b = np.array([n1 @ p1, n2 @ p2, 0.0], dtype=np.float64)
#     pt = np.linalg.solve(A, b)
#     return pt, d



def main():
    K, dist = load_calibration_data("./intrinsics.json")

    analysis_vid = cv.VideoCapture("./data/cup1.mp4")

    # FIX 3: rect_canonical uses only x,y and must be float32
    rect_canonical = np.array([[0, 0, 0], [WIDTH, 0, 0], [WIDTH, HEIGHT, 0], [0, HEIGHT, 0]], dtype=np.float32)

    rect_H_top = None
    rect_H_btm = None

    # Simple namespace to accumulate calibration points across frames
    class run_state: pass

    pts_file = open("points_raw.txt", "w")
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
            
            print("Started the parametrization of Plane 1 (top) and 2 (btm)")
            
            rects = detect_rectangle(frame)
            if rects is None:
                print("Waiting for two rectangles…")
                continue

            rect_top_img, rect_btm_img = rects

            # H must map world to image so decompose_homography gets H = K[r1|r2|t]
            rect_canonical_2d = rect_canonical[:, :2]   # drop z=0 column
            rect_H_top, mask_top = cv.findHomography(rect_canonical_2d, rect_top_img)
            rect_H_btm, mask_btm = cv.findHomography(rect_canonical_2d, rect_btm_img)

            print("H_top:\n", rect_H_top)
            top_Rotation, top_translation = decompose_homography(rect_H_top, K)
            print("top_Rotation:\n", top_Rotation)
            print("top_translation:\n", top_Rotation)

            print("H_btm:\n", rect_H_btm)
            btm_Rotation, btm_translation = decompose_homography(rect_H_btm, K)
            print("btm_Rotation:\n", btm_Rotation)
            print("top_translation:\n", btm_translation)
            
            # the point and normal of the plane
            top_pt = top_translation
            top_normal = top_Rotation @ np.array([[0],[0],[1]])
            btm_pt = btm_translation
            btm_normal = btm_Rotation @ np.array([[0],[0],[1]])
            
            print("top_pt:\n", top_pt)
            print("top_normal:\n", top_normal)
            print("btm_pt:\n", btm_pt)
            print("btm_normal:\n", btm_normal)
            
            print("Completed the parametrization of Plane 1 (top) and 2 (btm)")
        
        print("Started the parametrization of Laser Plane")

        laser_pts_top, laser_pts_btm, laser_pts_obj = detect_laser_points(frame, rect_top_img, rect_btm_img)

        print(f"Top rect:       {len(laser_pts_top)} point(s)  {[p[:2].tolist() for p in laser_pts_top]}")
        print(f"Btm rect:       {len(laser_pts_btm)} point(s)  {[p[:2].tolist() for p in laser_pts_btm]}")
        print(f"Obj (others):   {len(laser_pts_obj)} point(s)  {[p[:2].tolist() for p in laser_pts_obj]}")

        if K is None:
            continue

        ## Back-project 2D laser points to rays and intersect
        top_pts3d = []
        btm_pts3d = []
        
        for p2d in laser_pts_top:
            direction = backproject_to_ray(p2d, K)
            p3d = intersect_ray_plane(direction, top_pt, top_normal)
            if p3d is not None:
                top_pts3d.append(p3d)

        for p2d in laser_pts_btm:
            direction = backproject_to_ray(p2d, K)
            p3d = intersect_ray_plane(direction, btm_pt, btm_normal)
            if p3d is not None:
                btm_pts3d.append(p3d)

        print(f"3D top: {len(top_pts3d)} pt(s)")
        print(f"3D btm: {len(btm_pts3d)} pt(s)")

        # --- Fit laser plane once, then freeze it ---
        if not hasattr(run_state, 'lp_pt'):
            run_state.lp_pt   = None
            run_state.lp_norm = None
            run_state.acc_top = []
            run_state.acc_btm = []

        if run_state.lp_pt is None:
            run_state.acc_top.extend(top_pts3d)
            run_state.acc_btm.extend(btm_pts3d)

            if len(run_state.acc_top) < 3 or len(run_state.acc_btm) < 3:
                print(f"Accumulating... top={len(run_state.acc_top)}/3  btm={len(run_state.acc_btm)}/3")
                continue

            run_state.lp_pt, run_state.lp_norm = fit_plane_svd(
                run_state.acc_top + run_state.acc_btm)
            run_state.acc_top = None   # free — no longer needed
            run_state.acc_btm = None
            print(f"Laser plane fitted and frozen.")
            print(f"  point:  {run_state.lp_pt}")
            print(f"  normal: {run_state.lp_norm}")

        lp_pt, lp_norm = run_state.lp_pt, run_state.lp_norm

        print("Completed the parametrization of Laser Plane")

        print("Started the accumulation of laser points")

        # --- Intersect object laser rays with the fitted laser plane ---
        laser_pts3d = []
        
        for p2d in laser_pts_top:
            direction = backproject_to_ray(p2d, K)
            p3d = intersect_ray_plane(direction, lp_pt, lp_norm)
            if p3d is not None:
                laser_pts3d.append(p3d)

        for p2d in laser_pts_btm:
            direction = backproject_to_ray(p2d, K)
            p3d = intersect_ray_plane(direction, lp_pt, lp_norm)
            if p3d is not None:
                laser_pts3d.append(p3d)

        for p2d in laser_pts_obj:
            direction = backproject_to_ray(p2d, K)
            p3d = intersect_ray_plane(direction, lp_pt, lp_norm)
            if p3d is not None:
                laser_pts3d.append(p3d)

        for p3d in laser_pts3d:
            pts_file.write(f"{p3d[0]:.6f} {p3d[1]:.6f} {p3d[2]:.6f}\n")
        print(f"3D points on laser plane: {len(laser_pts3d)} pt(s)")

        print("Completed the accumulation of laser points")

    pts_file.close()

    # Build PLY: count lines first (O(1) memory), then stream-copy
    n_pts = sum(1 for _ in open("points_raw.txt"))
    if n_pts > 0:
        with open("points_raw.txt") as src, open("laser_scanned_obj.ply", "w") as dst:
            dst.write("ply\nformat ascii 1.0\n")
            dst.write(f"element vertex {n_pts}\n")
            dst.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for line in src:
                dst.write(line)
        print(f"Saved {n_pts} points to laser_scanned_obj.ply")

    analysis_vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()