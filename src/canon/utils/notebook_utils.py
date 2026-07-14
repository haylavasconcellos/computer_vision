"""Utility functions to be used on Jupyter notebook enviroments
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def _calculate_fig_size(img: np.ndarray, pixel_size: float) -> tuple[float, float]:
    return (float(img.shape[0]) * pixel_size, float(img.shape[1]) * pixel_size)


def show_image(img: np.ndarray, show_axis: bool = False, pixel_size: float = 0.005):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    figsize = _calculate_fig_size(img_rgb, pixel_size)
    plt.figure(figsize=figsize)

    # Plot the image
    plt.imshow(img_rgb)
    
    if not show_axis:
        plt.axis('off')  # Hide axes
    
    
def show_image_with_keypoints(img: np.ndarray, keypoints: list[cv2.KeyPoint], show_axis: bool = False, pixel_size: float = 0.005):
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
        color=(0, 0, 255)  # bright red for visibility
    )
    img_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
    
    figsize = _calculate_fig_size(img_rgb, pixel_size)
    plt.figure(figsize=figsize)
    
    if not show_axis:
        plt.axis('off')  # Hide axes

    plt.imshow(img_rgb)