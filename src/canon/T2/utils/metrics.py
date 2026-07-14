import cv2
import numpy as np


def compute_reprojection_errors(points_3d, points_2d, Rt, K, homogenity=1):
    """
    Compute reprojection errors for one camera view.

    Args:
        points_3d: (N,3) or (4,N) points.
        points_2d: (N,2) image correspondences.
        Rt: 3x4 camera extrinsics.
        K: 3x3 intrinsics.
        homogenity: 1 if 3D points are homogeneous, 0 if already Euclidean.

    Returns:
        dict with mean and median errors.
    """
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    rvec, _ = cv2.Rodrigues(R)
    X = points_3d
    if homogenity == 1 and X.shape[0] == 4:
        X = cv2.convertPointsFromHomogeneous(X.T)[:,0,:]

    proj, _ = cv2.projectPoints(X, rvec, t, K, None)
    proj = proj.reshape(-1, 2)

    # Ensure equal length
    n = min(len(proj), len(points_2d))
    proj = proj[:n]
    pts2d = points_2d[:n]

    errors = np.linalg.norm(proj - pts2d, axis=1)

    return {
        "mean_error": float(np.mean(errors)) if len(errors) else None,
        "median_error": float(np.median(errors)) if len(errors) else None
    }



def compute_bounding_box(points_3d):
    """
    Axis-aligned bounding box and density.
    """
    if points_3d is None or len(points_3d) == 0:
        return {"dx": 0, "dy": 0, "dz": 0, "density": 0.0}

    pts = np.asarray(points_3d)
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    x_min, y_min, z_min = np.min(pts, axis=0)
    x_max, y_max, z_max = np.max(pts, axis=0)

    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    area = max(dx * dy, 1e-9)  # XY area for density

    return {
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "density": len(pts) / area,
    }
