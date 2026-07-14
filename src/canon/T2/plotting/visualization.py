"""Plotting and visualization utilities for T2 - 3D reconstruction project
This module provides visualization functions for feature matching, epipolar geometry,
and 3D point clouds.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm

from canon.utils import image_utils


def draw_matches_with_epipolar_lines(img1: np.ndarray,
                                   img2: np.ndarray,
                                   kp1: List[cv2.KeyPoint],
                                   kp2: List[cv2.KeyPoint],
                                   matches: List[cv2.DMatch],
                                   F: np.ndarray,
                                   inlier_mask: np.ndarray,
                                   num_lines: int = 20) -> np.ndarray:
    """
    Draw matches and epipolar lines between two images.
    
    Args:
        img1: First image
        img2: Second image
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: Matches between keypoints
        F: Fundamental matrix
        inlier_mask: Inlier mask from RANSAC
        num_lines: Number of epipolar lines to draw
    
    Returns:
        Combined image with matches and epipolar lines
    """
    # Draw matches
    inlier_matches = [matches[i] for i in range(len(matches)) if inlier_mask[i]]
    
    # Sample matches for epipolar lines
    step = max(1, len(inlier_matches) // num_lines)
    sampled_matches = inlier_matches[::step][:num_lines]
    
    # Draw all inlier matches
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        matchColor=(0, 255, 0),  # Green for inliers
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Draw epipolar lines
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    for match in sampled_matches:
        # Get point coordinates
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        
        # Compute epipolar line in second image
        line = F @ np.array([pt1[0], pt1[1], 1])
        
        # Draw line in second image (offset by width of first image)
        x_start = w1
        x_end = w1 + w2
        
        if abs(line[1]) > 1e-6:  # Avoid division by zero
            y_start = int((-line[2] - line[0] * x_start) / line[1])
            y_end = int((-line[2] - line[0] * x_end) / line[1])
            
            # Clip to image bounds
            if 0 <= y_start <= h2 and 0 <= y_end <= h2:
                cv2.line(img_matches, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
    
    return img_matches


def draw_feature_comparison(images: Dict[str, np.ndarray],
                          comparison_results: Dict[str, Dict[str, Tuple]],
                          sample_image: str,
                          save_path: str = "T2/interim") -> None:
    """
    Create a comparison visualization of different feature detectors.
    
    Args:
        images: Dictionary of images
        comparison_results: Results from compare_detectors function
        sample_image: Name of image to visualize
        save_path: Path to save the comparison image
    """
    import os
    
    if sample_image not in images:
        print(f"Image {sample_image} not found")
        return
    
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    
    img = images[sample_image]
    detectors = list(comparison_results.keys())
    
    # Create subplot for each detector + original image
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Show original image first
    img_rgb_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb_original)
    axes[0].set_title("Imagem Original")
    axes[0].axis('off')
    
    # Define colors for each detector
    detector_colors = {
        'SIFT': (255, 0, 0),    # Red
        'ORB': (0, 255, 0),     # Green  
        'AKAZE': (0, 0, 255)    # Blue
    }
    
    for i, detector in enumerate(detectors):
        ax_idx = i + 1  # Skip first subplot (original image)
        if ax_idx >= len(axes):
            break
            
        if sample_image in comparison_results[detector]:
            kp, _ = comparison_results[detector][sample_image]
            
            # Draw keypoints with detector-specific color
            color = detector_colors.get(detector, (0, 255, 0))
            img_with_kp = cv2.drawKeypoints(
                img, kp, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=color
            )
            
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)
            
            axes[ax_idx].imshow(img_rgb)
            axes[ax_idx].set_title(f"{detector}\n{len(kp)} keypoints")
            axes[ax_idx].axis('off')
        else:
            axes[ax_idx].text(0.5, 0.5, f"{detector}\nERROR", 
                            ha='center', va='center', transform=axes[ax_idx].transAxes)
            axes[ax_idx].axis('off')
    
    # Hide unused subplot if any
    if len(detectors) < 3:
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Comparação de Detectores - {sample_image}", fontsize=16, y=1.02)
    
    # Save the comparison
    save_file = os.path.join(save_path, f"feature_comparison_{sample_image}.png")
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Comparação de features salva: {save_file}")
    
    # Also save individual keypoint images for each detector
    for detector in detectors:
        if sample_image in comparison_results[detector]:
            kp, _ = comparison_results[detector][sample_image]
            color = detector_colors.get(detector, (0, 255, 0))
            
            img_with_kp = cv2.drawKeypoints(
                img, kp, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=color
            )
            
            # Save individual keypoint image
            individual_file = os.path.join(save_path, f"{detector}_{sample_image}_keypoints.jpg")
            cv2.imwrite(individual_file, img_with_kp)
            print(f" {detector} keypoints salvos: {individual_file}")


def plot_3d_points(points_3d: np.ndarray,
                  colors: Optional[np.ndarray] = None,
                  title: str = "3D Point Cloud",
                  save_path: Optional[str] = None) -> None:
    """
    Plot 3D points using matplotlib.
    
    Args:
        points_3d: 3D points (Nx3)
        colors: Colors for points (Nx3) in range [0,1]
        title: Plot title
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c=colors, s=1, alpha=0.6)
    else:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                         points_3d[:, 1].max() - points_3d[:, 1].min(),
                         points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D plot saved to {save_path}")
    
    plt.show()


def visualize_matching_results(match_results: Dict[Tuple[str, str], Dict],
                             images: Dict[str, np.ndarray],
                             save_pairs: int = 5,
                             save_path: str = "T2/interim") -> None:
    """
    Visualize matching results for top image pairs.
    
    Args:
        match_results: Results from match_image_collection
        images: Dictionary of images
        save_pairs: Number of top pairs to visualize
        save_path: Path to save visualizations
    """
    # Sort pairs by number of matches
    sorted_pairs = sorted(match_results.items(), 
                         key=lambda x: x[1]['num_matches'], 
                         reverse=True)
    
    # Use tqdm for progress when saving multiple match visualizations
    for (img1_name, img2_name), result in tqdm(sorted_pairs[:save_pairs], 
                                              desc="Saving match visualizations", 
                                              unit="pair"):
        img1 = images[img1_name]
        img2 = images[img2_name]
        
        matches = result['matches']
        kp1 = result['kp1']
        kp2 = result['kp2']
        
        # Draw matches
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Save visualization
        filename = f"matches_{img1_name}_{img2_name}_{len(matches)}matches"
        success = image_utils.save_image(img_matches, filename, save_path)
        
        if success:
            # Only print for small datasets to avoid spam
            if save_pairs <= 10:
                print(f"Saved match visualization: {filename}")


def create_matching_summary_plot(match_results: Dict[Tuple[str, str], Dict],
                               save_path: str = "T2/interim/matching_summary.png") -> None:
    """
    Create a summary plot of matching statistics.
    
    Args:
        match_results: Results from match_image_collection
        save_path: Path to save the summary plot
    """
    if not match_results:
        print("No match results to plot")
        return
    
    # Extract statistics
    num_matches = [result['num_matches'] for result in match_results.values()]
    pair_names = [f"{pair[0]}-{pair[1]}" for pair in match_results.keys()]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Histogram of match counts
    ax1.hist(num_matches, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Matches')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Match Counts')
    ax1.grid(True, alpha=0.3)
    
    # Top pairs by match count
    sorted_results = sorted(match_results.items(), 
                           key=lambda x: x[1]['num_matches'], 
                           reverse=True)
    
    top_pairs = sorted_results[:15]  # Top 15 pairs
    top_names = [f"{pair[0][0]}-{pair[0][1]}" for pair in top_pairs]
    top_counts = [pair[1]['num_matches'] for pair in top_pairs]
    
    ax2.barh(range(len(top_names)), top_counts)
    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels(top_names, fontsize=8)
    ax2.set_xlabel('Number of Matches')
    ax2.set_title('Top 15 Image Pairs by Match Count')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Matching summary saved to {save_path}")
    
    # Print statistics
    print(f"\nMatching Statistics:")
    print(f"Total pairs processed: {len(match_results)}")
    print(f"Average matches per pair: {np.mean(num_matches):.1f}")
    print(f"Median matches per pair: {np.median(num_matches):.1f}")
    print(f"Max matches: {max(num_matches)}")
    print(f"Min matches: {min(num_matches)}")


def save_epipolar_visualization(img1: np.ndarray,
                              img2: np.ndarray,
                              match_result: Dict,
                              F: np.ndarray,
                              inlier_mask: np.ndarray,
                              pair_name: str,
                              save_path: str = "T2/interim") -> None:
    """
    Save epipolar geometry visualization for an image pair.
    
    Args:
        img1: First image
        img2: Second image
        match_result: Match result dictionary
        F: Fundamental matrix
        inlier_mask: Inlier mask
        pair_name: Name for the pair (for filename)
        save_path: Path to save visualization
    """
    img_with_epilines = draw_matches_with_epipolar_lines(
        img1, img2,
        match_result['kp1'], match_result['kp2'],
        match_result['matches'], F, inlier_mask
    )
    
    filename = f"epipolar_{pair_name}_{np.sum(inlier_mask)}inliers"
    success = image_utils.save_image(img_with_epilines, filename, save_path)
    
    if success:
        print(f"Saved epipolar visualization: {filename}")
    else:
        print(f"Failed to save epipolar visualization: {filename}")


def plot_point_cloud(points, title="Point Cloud", save_path=None):
    """
    Plota uma nuvem de pontos 3D com matplotlib.

    Args:
        points: array (N, 3) contendo pontos [x, y, z].
        title: título do gráfico.
        save_path: caminho do arquivo para salvar a figura (ex: "cloud.png").
                   Se None, mostra a figura na tela.
    """
    if isinstance(points, list):
        points = np.array(points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap="viridis", s=2)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # fecha para não abrir janela
    else:
        plt.show()
        
        
def save_point_cloud_ply(points: np.ndarray,
                         colors: np.ndarray = None,
                         save_path: str = "cloud.ply") -> None:
    """
    Save a 3D point cloud in PLY format (MeshLab compatible).

    Args:
        points: (N, 3) array with 3D points
        colors: (N, 3) array with RGB values [0-255] (optional)
        save_path: output file path (.ply)
    """
    if isinstance(points, list):
        points = np.array(points)
    if colors is not None and isinstance(colors, list):
        colors = np.array(colors)

    if colors is None:
        # default: white
        colors = np.ones_like(points) * 255

    assert points.shape[0] == colors.shape[0], "Points and colors must have same length"

    n_points = points.shape[0]

    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(save_path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors.astype(np.uint8)):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    print(f"Saved point cloud with {n_points} points to {save_path}")


def visualize_point_cloud(file_path):
    """
    Reads a PLY file and visualizes it using Open3D.
    Supports both point clouds and triangle meshes.
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.has_points():
            print(f"Visualizing point cloud from: {file_path}")
            o3d.visualization.draw_geometries([pcd])
            return

        print(f"Error: Could not load valid point cloud or mesh from {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
