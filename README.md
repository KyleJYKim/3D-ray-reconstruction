# Laser-Plane 3D Scanner

A single-camera laser triangulation pipeline. Two rectangular calibration targets (one on the wall, one on the table, mutually perpendicular) let us recover the laser plane from the red laser line they catch. An object placed between them is then scanned by intersecting camera rays with that laser plane, producing a 3D point cloud saved as `.ply`.

```
video frame
   │  undistort (K, dist)
   ▼
detect two rectangles ──► homographies H_top, H_btm ──► plane (p, n) for wall and table
   │
   ▼
detect red laser blobs ──► classify: top rect / bottom rect / object
   │
   ├─ top + bottom blobs ──► back-project onto wall/table planes ──► 3D points
   │                                   │
   │                                   ▼
   │                         SVD plane fit ──► laser plane (c, m)
   │                                   │
   └─ object blobs ───────► back-project onto laser plane ──► object 3D points ──► object.ply
```

## Requirements

- Python 3.9+
- `numpy`, `opencv-python`
- `intrinsics.json` with keys `K` (3×3 camera matrix) and `distortion` (OpenCV distortion coefficients)
- Input video at `./data/cup1.mp4`

```bash
pip install numpy opencv-python
python detector.py        # press q to quit
```

Output: `object.ply` (ASCII PLY, camera coordinates), rewritten whenever new object points are added.

## Pipeline

### 1. Undistortion

Each frame is undistorted with `cv.undistort(frame, K, dist)` so the pinhole model below holds.

### 2. Rectangle detection (`detect_rectangle`)

1. Grayscale → Gaussian blur (5×5) → binary threshold (10).
2. Canny edges → `findContours` (external only).
3. Each contour is simplified with `approxPolyDP`, with tolerance $\varepsilon = 0.02 \cdot \text{perimeter}$. Contours with exactly 4 vertices are kept.
4. Corners are ordered TL → TR → BR → BL (`order_points`):
   - TL = smallest $x+y$, BR = largest $x+y$
   - TR = smallest $y-x$, BL = largest $y-x$
5. Exactly two rectangles must be found. They are sorted by their maximum image $y$: the smaller one is the **top (wall)** rectangle, the other is the **bottom (table)** rectangle.

### 3. Homography and plane parameters

The canonical rectangle lies in the plane $Z=0$ with corners

$$(0,0),\ (W,0),\ (W,H),\ (0,H)$$

with $W$ = `WIDTH` = 250 and $H$ = `HEIGHT` = 150 (the 3D scene is expressed in whatever unit these are in).

For a planar point $(X, Y, 0)$ the pinhole projection reduces to a homography:

$$\tilde{x} \sim K\,[\,r_1\ r_2\ t\,]\,(X, Y, 1)^T = H\,(X,Y,1)^T$$

`cv.findHomography` is called with image corners as source and canonical corners as destination, so it returns the **inverse** direction of the equation above if used literally. See *Known issues* below.

Decomposition (`decompose_homography`):

1. Remove intrinsics: $H_n = K^{-1} H$, with columns $h_1, h_2, h_3$.
2. Recover scale from the two rotation columns, which must be unit length:
   $$\lambda = \tfrac{1}{2}\left(\lVert h_1 \rVert + \lVert h_2 \rVert\right),\quad r_1 = h_1/\lambda,\ r_2 = h_2/\lambda,\ t = h_3/\lambda$$
3. Complete the basis: $r_3 = r_1 \times r_2$.
4. Noise makes $[r_1\ r_2\ r_3]$ only approximately orthonormal, so project it onto $SO(3)$ with an SVD:
   $$R_{raw} = U\Sigma V^T \;\Rightarrow\; R = UV^T$$
   (if $\det R < 0$ the last singular direction is flipped).

Each calibration plane is then represented by a **point and a normal**:

$$p = t,\qquad n = R\,(0,0,1)^T = r_3$$

### 4. Laser detection (`detect_laser_points`)

1. Gaussian blur (15×15) to merge the stripe into compact blobs.
2. Convert to HSV. Red wraps around the hue axis, so two ranges are OR-ed: hue in $[0, 20]$ and $[170, 180]$, with $S, V \ge 50$.
3. `findContours` on the **full-frame** red mask (no ROI masking).
4. Blobs with area outside $[2, 500]$ px are discarded.
5. Sub-pixel centroid from image moments:
   $$c_x = \frac{m_{10}}{m_{00}},\qquad c_y = \frac{m_{01}}{m_{00}}$$
   stored in homogeneous form $(c_x, c_y, 1)$.
6. Classification with `pointPolygonTest` against each rectangle polygon:
   - inside the top rectangle → `pts_top`
   - else inside the bottom rectangle → `pts_btm`
   - else → `pts_obj` (stripe on the object)

### 5. Back-projection to 3D (`ray_plane_intersect`)

The camera sits at the origin. For a homogeneous pixel $\tilde{x} = (u, v, 1)^T$ the viewing ray direction is

$$d = \frac{K^{-1}\tilde{x}}{\lVert K^{-1}\tilde{x} \rVert}$$

A plane through point $p$ with normal $n$ satisfies $(X - p)\cdot n = 0$. Substituting $X = s\,d$:

$$s = \frac{p \cdot n}{d \cdot n},\qquad X = s\,d$$

The intersection is rejected if $|d \cdot n| < 10^{-9}$ (ray parallel to plane).

- Laser blobs inside the top/bottom rectangles are intersected with the wall/table planes $(p_{top}, n_{top})$ and $(p_{btm}, n_{btm})$.
- Object blobs are intersected with the **laser plane** (next step).

### 6. Laser plane fit (`fit_plane_svd`)

The 3D points recovered on the wall and table all lie on the laser plane. Once at least 3 points have been accumulated on each surface (points are accumulated across frames), a least-squares plane is fitted:

$$c = \frac{1}{N}\sum_i X_i,\qquad (X - c) = U\Sigma V^T$$

The laser-plane normal $m$ is the last row of $V^T$ (direction of least variance), already unit length. The plane is $(X - c)\cdot m = 0$.

### 7. Laser lines on the targets (`intersect_planes`)

The intersection line of the laser plane $(c, m)$ with a calibration plane $(p, n)$:

- direction: $\ell = \dfrac{m \times n}{\lVert m \times n \rVert}$
- a point on the line, from the 3×3 system
  $$\begin{bmatrix} m^T \\ n^T \\ \ell^T \end{bmatrix} x = \begin{bmatrix} m\cdot c \\ n\cdot p \\ 0 \end{bmatrix}$$
  (the third row fixes the free coordinate along the line).

Parallel planes ($\lVert m \times n \rVert < 10^{-9}$) return `None`. This line is mainly a sanity check: it should coincide with the stripe visible on the wall and table.

### 8. Object point cloud and PLY export (`write_ply`)

Each object blob $(u, v, 1)$ is back-projected with the formulas of step 5 against the laser plane $(c, m)$, and the resulting 3D points are accumulated over all frames. `write_ply` writes an ASCII PLY:

```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
end_header
x y z
...
```

Open `object.ply` in MeshLab, CloudCompare or Blender. Coordinates are in the camera frame, in the unit of `WIDTH`/`HEIGHT`.

## Assumptions

- The camera, the two targets and the laser are all **fixed** while the video runs. The object (or the laser sweep) is what changes. This is what allows accumulating points across frames and fitting a single laser plane.
- The two rectangles are detected once, in the first frame where exactly two are found; their homographies are then reused.
- Target dimensions (`WIDTH`, `HEIGHT`) match the physical rectangles.

## Known issues / things to verify

- **Normal sign.** A homography is defined only up to scale *including sign*, so the recovered $t$ (and hence $n$) can have the wrong orientation. The current `ray_plane_intersect` works around this by flipping $s$ when it comes out negative (`z = z * -1`). A cleaner fix is to enforce $t_z > 0$ right after decomposition (negate $H_n$ if not), and then drop the flip.
- **Homography direction.** `findHomography(image_pts, canonical_pts)` maps image → plane. The decomposition in step 3 assumes plane → image, i.e. $H_{plane \to image}$. If the recovered planes look wrong, swap the arguments to `findHomography(canonical_pts, image_pts)`.
- **Frame skipping.** While rectangles are not yet detected the loop `continue`s; make sure the first frames show both targets clearly.
- **Laser thresholds.** The HSV ranges and blob area limits are tuned for one setup; lower $S$/$V$ for a dim laser, raise them if warm lighting causes false positives.
