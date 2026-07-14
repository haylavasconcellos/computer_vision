import cv2
import numpy as np
from canon.T1.process import feature_extraction
from canon.utils import image_utils

def draw_pairing_lines(img1: np.ndarray,
                       img2: np.ndarray,
                       alg: str,
                       filename: str,
                       path: str = "T1/interim",
                       img_format: str = ".jpg"):

    algs = {
        "SIFT": feature_extraction.SIFT,
        "ORB": feature_extraction.ORB,
        "AKAZE": feature_extraction.AKAZE
    }

    func = algs.get(alg.upper())

    kp1, des1 = func(img1)
    kp2, des2 = func(img2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Aplica Lowe ratio test
    good_matches = david_lowe_ratio_test(matches)

    # Desenha as linhas
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

    # --- Marcar início e fim ---
    h1, w1 = img1.shape[:2]

    for m in good_matches:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))  # ponto na img1
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))  # ponto na img2
        pt2 = (int(pt2[0] + w1), int(pt2[1]))      # compensar deslocamento da concatenação

        # "X" no ponto inicial (img1)
        cv2.drawMarker(img_matches, pt1, (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=12, thickness=2, line_type=cv2.LINE_AA)

        # "O" no ponto final (img2)
        cv2.circle(img_matches, pt2, 6, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    image_utils.save_image(img_matches, filename, path, img_format)

    return img_matches


def david_lowe_ratio_test(matches: cv2.DMatch, ratio=0.75):
    """
    Receives the matches returned by knnMatch and applies the ratio test.
    
    Args:
        matches (list[list[cv2.DMatch]]): list of lists of matches returned by knnMatch.
        ratio (float): Lowe's ratio test factor (default=0.75).
    
    Returns:
        list[cv2.DMatch]: list of good (accepted) matches.
    """
    good_matches = []
    for m, n in matches:  
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches