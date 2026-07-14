"""Matching and epipolar geometry for T2 - 3D reconstruction project
This module implements feature matching, fundamental/essential matrix estimation,
and 3D point triangulation for structure-from-motion.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import itertools
from scipy.optimize import least_squares
from tqdm import tqdm

# Import from T1 for reuse
from canon.T1.plotting.plotting import david_lowe_ratio_test


class ImagePairMatcher:
    """Class for matching features between image pairs and estimating epipolar geometry."""
    
    def __init__(self, 
                 matcher_type: str = "BF",
                 ratio_threshold: float = 0.75,
                 cross_check: bool = True):
        """
        Initialize the matcher.
        
        Args:
            matcher_type: "BF" for BruteForce or "FLANN" for FLANN-based matcher
            ratio_threshold: Ratio for Lowe's ratio test
            cross_check: Whether to use cross-checking for BF matcher
        """
        self.ratio_threshold = ratio_threshold
        self.matcher_type = matcher_type
        self.cross_check = cross_check
        
        # Matchers will be created dynamically based on descriptor type
        self.bf_float_matcher = None
        self.bf_binary_matcher = None
        self.flann_float_matcher = None
        self.flann_binary_matcher = None
        
    def _get_matcher(self, descriptors: np.ndarray):
        is_binary = descriptors.dtype == np.uint8
        
        if self.matcher_type == "BF":
            if is_binary:
                if self.bf_binary_matcher is None:
                    self.bf_binary_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=self.cross_check)
                return self.bf_binary_matcher
            else:
                if self.bf_float_matcher is None:
                    self.bf_float_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=self.cross_check)
                return self.bf_float_matcher
                
        elif self.matcher_type == "FLANN":
            if is_binary:
                # Use LSH (Locality Sensitive Hashing) for binary descriptors like ORB and AKAZE
                if self.flann_binary_matcher is None:
                    FLANN_INDEX_LSH = 6
                    index_params = dict(algorithm=FLANN_INDEX_LSH,
                                      table_number=12,   # 12 is recommended
                                      key_size=20,       # 20 is recommended
                                      multi_probe_level=2) # 2 is recommended
                    search_params = dict(checks=50)
                    self.flann_binary_matcher = cv2.FlannBasedMatcher(index_params, search_params)
                return self.flann_binary_matcher
            else:
                # Use KDTREE for float descriptors like SIFT
                if self.flann_float_matcher is None:
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    self.flann_float_matcher = cv2.FlannBasedMatcher(index_params, search_params)
                return self.flann_float_matcher
        else:
            raise ValueError("matcher_type must be 'BF' or 'FLANN'")
    
    def match_features(self, 
                      desc1: np.ndarray, 
                      desc2: np.ndarray,
                      apply_ratio_test: bool = True) -> List[cv2.DMatch]:
        """
        Match features between two sets of descriptors.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            apply_ratio_test: Whether to apply Lowe's ratio test
        
        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None:
            return []
        
        # Get appropriate matcher based on descriptor type
        matcher = self._get_matcher(desc1)
        
        if self.matcher_type == "BF" and not apply_ratio_test:
            # Direct matching with cross-check
            matches = matcher.match(desc1, desc2)
            return sorted(matches, key=lambda x: x.distance)
        else:
            # kNN matching for ratio test
            knn_matches = matcher.knnMatch(desc1, desc2, k=2)
            
            if apply_ratio_test:
                good_matches = david_lowe_ratio_test(knn_matches, self.ratio_threshold)
            else:
                good_matches = [m[0] for m in knn_matches if len(m) >= 1]
            
            return good_matches
    
    def match_image_pair(self,
                        features1: Tuple[List[cv2.KeyPoint], np.ndarray],
                        features2: Tuple[List[cv2.KeyPoint], np.ndarray],
                        min_matches: int = 20) -> Optional[Dict]:
        """
        Match features between two images and return comprehensive results.
        
        Args:
            features1: (keypoints, descriptors) from first image
            features2: (keypoints, descriptors) from second image
            min_matches: Minimum number of matches required
        
        Returns:
            Dictionary with matching results or None if insufficient matches
        """
        kp1, desc1 = features1
        kp2, desc2 = features2
        
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < min_matches:
            return None
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        return {
            'matches': matches,
            'pts1': pts1,
            'pts2': pts2,
            'kp1': kp1,
            'kp2': kp2,
            'num_matches': len(matches)
        }


class EpipolarGeometryEstimator:
    """Class for estimating fundamental/essential matrices and camera poses."""
    
    def __init__(self, 
                 camera_matrix: Optional[np.ndarray] = None,
                 ransac_threshold: float = 1.0,
                 confidence: float = 0.99):
        """
        Initialize the estimator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix (K). If None, assumes uncalibrated setup.
            ransac_threshold: RANSAC threshold for inlier detection
            confidence: Confidence level for RANSAC
        """
        self.camera_matrix = camera_matrix
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        self.is_calibrated = camera_matrix is not None
    
    def estimate_fundamental_matrix(self, 
                                  pts1: np.ndarray, 
                                  pts2: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Estimate fundamental matrix between two views.
        
        Args:
            pts1: Points in first image (Nx1x2)
            pts2: Points in second image (Nx1x2)
        
        Returns:
            Tuple of (fundamental_matrix, inlier_mask)
        """
        if len(pts1) < 8:
            return None, np.array([])
        
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            cv2.FM_RANSAC,
            self.ransac_threshold,
            self.confidence
        )
        
        return F, mask
    
    def estimate_essential_matrix(self,
                                pts1: np.ndarray,
                                pts2: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Estimate essential matrix between two calibrated views.
        
        Args:
            pts1: Points in first image (Nx1x2)
            pts2: Points in second image (Nx1x2)
        
        Returns:
            Tuple of (essential_matrix, inlier_mask)
        """
        if not self.is_calibrated:
            raise ValueError("Camera matrix required for essential matrix estimation")
        
        if len(pts1) < 5:
            return None, np.array([])
        
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=self.confidence,
            threshold=self.ransac_threshold
        )
        
        return E, mask
    
    def recover_pose(self,
                    E: np.ndarray,
                    pts1: np.ndarray,
                    pts2: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover camera pose from essential matrix.
        
        Args:
            E: Essential matrix
            pts1: Points in first image
            pts2: Points in second image
            mask: Inlier mask (optional)
        
        Returns:
            Tuple of (num_inliers, R, t, triangulated_mask)
        """
        if not self.is_calibrated:
            raise ValueError("Camera matrix required for pose recovery")
        
        if mask is not None:
            pts1_inliers = pts1[mask.ravel() == 1]
            pts2_inliers = pts2[mask.ravel() == 1]
        else:
            pts1_inliers = pts1
            pts2_inliers = pts2
        
        num_inliers, R, t, tri_mask = cv2.recoverPose(
            E, pts1_inliers, pts2_inliers, self.camera_matrix
        )
        
        return num_inliers, R, t, tri_mask


class Triangulator:
    """Class for 3D point triangulation from multiple views."""
    
    def __init__(self, camera_matrix: np.ndarray):
        """
        Initialize triangulator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.camera_matrix = camera_matrix
    
    def triangulate_points_new(self,
                          pts1: np.ndarray,
                          pts2: np.ndarray,
                          P1: np.ndarray,
                          P2: np.ndarray, repeat: bool) -> np.ndarray:
        """
        Triangulate 3D points from two views.
        
        Args:
            pts1: Points in first image (Nx1x2)
            pts2: Points in second image (Nx1x2)
            R1: Rotation matrix for first camera
            t1: Translation vector for first camera
            R2: Rotation matrix for second camera
            t2: Translation vector for second camera
        
        Returns:
            3D points in homogeneous coordinates (4xN)
        """
        # Create projection matrices
        if not repeat:
            points1 = np.transpose(pts1)
            points2 = np.transpose(pts2)
        else:
            points1 = pts1
            points2 = pts2

        cloud = cv2.triangulatePoints(P1, P2, points1, points2)
        cloud = cloud / cloud[3]

        return points1, points2, cloud
    
    def triangulate_points(self,
                          pts1: np.ndarray,
                          pts2: np.ndarray,
                          R1: np.ndarray,
                          t1: np.ndarray,
                          R2: np.ndarray,
                          t2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from two views.
        
        Args:
            pts1: Points in first image (Nx1x2)
            pts2: Points in second image (Nx1x2)
            R1: Rotation matrix for first camera
            t1: Translation vector for first camera
            R2: Rotation matrix for second camera
            t2: Translation vector for second camera
        
        Returns:
            3D points in homogeneous coordinates (4xN)
        """
        P1 = self.camera_matrix @ np.hstack([R1, t1.reshape(-1, 1)])
        P2 = self.camera_matrix @ np.hstack([R2, t2.reshape(-1, 1)])
        
        # Reshape points for triangulation
        pts1_norm = pts1.reshape(-1, 2).T
        pts2_norm = pts2.reshape(-1, 2).T
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
        
        return points_4d

    def convert_to_3d(self, points_4d: np.ndarray) -> np.ndarray:
        """
        Convert homogeneous 4D points to 3D.
        
        Args:
            points_4d: 4D homogeneous points (4xN)
        
        Returns:
            3D points (Nx3)
        """
        # Convert from homogeneous coordinates
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T


def create_image_pairs(image_names: List[str], 
                      max_pairs: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Create pairs of images for matching.
    
    Args:
        image_names: List of image names
        max_pairs: Maximum number of pairs to create (None for all pairs)
    
    Returns:
        List of image name pairs
    """
    all_pairs = list(itertools.combinations(image_names, 2))
    
    if max_pairs is not None and len(all_pairs) > max_pairs:
        # Take evenly spaced pairs
        step = len(all_pairs) // max_pairs
        pairs = all_pairs[::step][:max_pairs]
    else:
        pairs = all_pairs
    
    return pairs


def match_image_collection(features: Dict[str, Tuple[List[cv2.KeyPoint], np.ndarray]],
                          matcher: ImagePairMatcher,
                          max_pairs: Optional[int] = None,
                          min_matches: int = 20) -> Dict[Tuple[str, str], Dict]:
    """
    Match features across a collection of images.
    
    Args:
        features: Dictionary mapping image names to (keypoints, descriptors)
        matcher: ImagePairMatcher instance
        max_pairs: Maximum number of pairs to process
        min_matches: Minimum matches required per pair
    
    Returns:
        Dictionary mapping image pairs to match results
    """
    image_names = list(features.keys())
    pairs = create_image_pairs(image_names, max_pairs)
    
    match_results = {}
    successful_pairs = 0
    
    print(f"Matching {len(pairs)} image pairs...")
    
    # Using tqdm for progress bar
    for img1, img2 in tqdm(pairs, desc="Matching pairs", unit="pair"):
        result = matcher.match_image_pair(
            features[img1], features[img2], min_matches
        )
        
        if result is not None:
            match_results[(img1, img2)] = result
            successful_pairs += 1
    
    print(f"Successfully matched {successful_pairs}/{len(pairs)} pairs")
    return match_results





def estimate_epipolar_matrix(sample_img: np.ndarray) -> np.ndarray:
    h, w = sample_img.shape[:2]

    fx = fy = max(w, h)  # Distância focal aproximada
    cx, cy = w / 2, h / 2  # Centro da imagem

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


def ReprojectionError(
    X: np.ndarray,
    pts: np.ndarray,
    Rt: np.ndarray,
    K: np.ndarray,
    homogenity: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    total_error = 0.0
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    r, _ = cv2.Rodrigues(R)
    if homogenity == 1:
        X = cv2.convertPointsFromHomogeneous(X.T)

    p, _ = cv2.projectPoints(X, r, t, K, distCoeffs=None)
    p = p[:, 0, :]
    p = np.float32(p)
    pts = np.float32(pts)
    if homogenity == 1:
        total_error = cv2.norm(p, pts.T, cv2.NORM_L2)
    else:
        total_error = cv2.norm(p, pts, cv2.NORM_L2)
    pts = pts.T
    tot_error = total_error / len(p)

    return tot_error, X, p


def PnP(
    X: np.ndarray,
    p: np.ndarray,
    K: np.ndarray,
    d: Optional[np.ndarray],
    p_0: np.ndarray,
    initial: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if initial == 1:
        X = X[:, 0, :]
        p = p.T
        p_0 = p_0.T

    ret, rvecs, t, inliers = cv2.solvePnPRansac(X, p, K, d, cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvecs)

    if inliers is not None:
        p = p[inliers[:, 0]]
        X = X[inliers[:, 0]]
        p_0 = p_0[inliers[:, 0]]
    else:
        print("This is an issue")

    return R, t, p, X, p_0


def OptimReprojectionError(x: np.ndarray) -> np.ndarray:
    Rt = x[0:12].reshape((3, 4))
    K = x[12:21].reshape((3, 3))
    rest = len(x[21:])
    rest = int(rest * 0.4)
    p = x[21:21 + rest].reshape((2, int(rest / 2)))
    X = x[21 + rest:].reshape((int(len(x[21 + rest:]) / 3), 3))
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    p = p.T
    num_pts = len(p)
    error = []
    r, _ = cv2.Rodrigues(R)

    p2d, _ = cv2.projectPoints(X, r, t, K, distCoeffs=None)
    p2d = p2d[:, 0, :]
    for idx in range(num_pts):
        img_pt = p[idx]
        reprojected_pt = p2d[idx]
        er = (img_pt - reprojected_pt) ** 2
        error.append(er)

    err_arr = np.array(error).ravel() / num_pts
    print(np.sum(err_arr))

    return err_arr


def BundleAdjustment(
    points_3d: np.ndarray,
    temp2: np.ndarray,
    Rtnew: np.ndarray,
    K: np.ndarray,
    r_error: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    opt_variables = np.hstack((Rtnew.ravel(), K.ravel()))
    opt_variables = np.hstack((opt_variables, temp2.ravel()))
    opt_variables = np.hstack((opt_variables, points_3d.ravel()))

    _ = np.sum(OptimReprojectionError(opt_variables))
    corrected_values = least_squares(fun=OptimReprojectionError, x0=opt_variables, gtol=r_error)

    corrected_values = corrected_values.x
    Rt = corrected_values[0:12].reshape((3, 4))
    K = corrected_values[12:21].reshape((3, 3))
    rest = len(corrected_values[21:])
    rest = int(rest * 0.4)
    p = corrected_values[21:21 + rest].reshape((2, int(rest / 2)))
    X = corrected_values[21 + rest:].reshape((int(len(corrected_values[21 + rest:]) / 3), 3))
    p = p.T

    return X, p, Rt


def Triangulation(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    repeat: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not repeat:
        points1 = np.transpose(pts1)
        points2 = np.transpose(pts2)
    else:
        points1 = pts1
        points2 = pts2

    cloud = cv2.triangulatePoints(P1, P2, points1, points2)
    cloud = cloud / cloud[3]

    return points1, points2, cloud


def get_intrinsic_matrix() -> np.ndarray:
    # return np.array([
    #     [2393.952166119461, -3.410605131648481e-13, 932.3821770809047],
    #     [0, 2398.118540286656, 628.2649953288065],
    #     [0, 0, 1]
    # ])

    return np.array([
        [1.92e+03, 0.00e+00, 5.40e+02],
        [0.00e+00, 1.92e+03, 9.60e+02],
        [0.00e+00, 0.00e+00, 1.00e+00]
    ])