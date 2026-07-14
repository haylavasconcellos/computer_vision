"""Utility functions to image loading and processing
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

try:
# Works in scripts
    current_file = Path(__file__).resolve()
except NameError:
    # Fallback for interactive sessions like Jupyter
    current_file = Path(sys.argv[0]).resolve() if sys.argv[0] else Path.cwd()

current_dir = current_file.parent

BASE_DATA_PATH = current_dir.parent.parent.parent / "data"


def load_images(datasetDirFromData: str) -> dict[str, np.ndarray]:
    """
    Load all images from a raw panorama folder into a dictionary.

    Args:
        datasetDirFromData (str): Considering the data folder as base, define the dataset path from it.
            Ex: Dataset at path (data/T1/Dataset) -> datasetDirFromData = "T1/Dataset"

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping filename (without extension)
                             to the loaded OpenCV image.
    """
    path = BASE_DATA_PATH / Path(datasetDirFromData)
    images: dict[str, np.ndarray] = {}
    
    for file in sorted(path.glob("*")):  # iterate over files
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(file))
            if img is not None:
                # Use filename without extension as key
                key = file.stem  
                images[key] = img
    
    return images


def save_image(img: np.ndarray, filename : str, path: str = "interim", img_format : str = ".jpg") -> bool:
    """
    Save an OpenCV image to disk.

    Args:
        img (np.ndarray): The image to save.
        filename (str): Name of the output file (without extension).
        path (str, optional): Subfolder under BASE_DATA_PATH to save the image. Default is "interim".
        format (str, optional): Image format/extension (e.g., 'jpg', 'png'). Default is 'jpg'.

    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """
    # Ensure format does not have a leading dot
    img_format = img_format.lstrip(".")
    output_path = BASE_DATA_PATH / Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    finalDir = output_path / f"{filename}.{img_format}"
    
    success = cv2.imwrite(finalDir, img)
    return success


def draw_points(image: np.ndarray, pts: Union[List[cv2.KeyPoint], np.ndarray], repro: bool) -> np.ndarray:
    """
    Draw keypoints or reprojection points on an image.

    Args:
        image: Input image.
        pts: List of cv2.KeyPoint if repro == False, or array of 2D points if repro == True.
        repro: If False, draw keypoints in green. If True, draw reprojection points in red.

    Returns:
        Image with drawn points.
    """
    if not repro:
        image = cv2.drawKeypoints(image, pts, image, color=(0, 255, 0), flags=0)
    else:
        for p in pts:
            image = cv2.circle(image, tuple(map(int, p)), 2, (0, 0, 255), -1)
    return image


def img_downscale(img: np.ndarray, downscale: int) -> np.ndarray:
    """
    Downscale an image using Gaussian pyramid.

    Args:
        img: Input image.
        downscale: Number of pyramid downsamples (scaled by factor of 2).

    Returns:
        Downscaled image.
    """
    downscale = int(downscale / 2)
    for _ in range(downscale):
        img = cv2.pyrDown(img)
    return img


def to_ply(path: str, point_cloud: np.ndarray, colors: np.ndarray, densify: bool, filename: str) -> None:
    """
    Save a 3D point cloud to a PLY file.

    Args:
        path: Directory path to save the file.
        point_cloud: (N,3) array of 3D points.
        colors: (N,3) array of colors (BGR format).
        densify: If True, save as 'dense.ply', else as 'sparse.ply'.
    """
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    verts = np.hstack([out_points, out_colors])

    # cleaning point cloud
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.linalg.norm(temp, axis=1)
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]

    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''

    # filename = "dense.ply" if densify else "sparse.ply"
    full_path = f"{path}/{filename}.ply"

    with open(full_path, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')
