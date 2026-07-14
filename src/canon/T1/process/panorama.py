import cv2
import numpy as np
from canon.utils import image_utils
from canon.T1.process import feature_extraction

def find_best_center_image(images: list) -> int:
    """
    Encontra o índice da melhor imagem de referência para o panorama.
    A melhor imagem é a que tem o maior número de correspondências com as outras.
    """
    num_images = len(images)
    match_counts = [0] * num_images
    kp_descs = [feature_extraction.SIFT(img, nfeatures=1000) for img in images]

    bf = cv2.BFMatcher(cv2.NORM_L2)

    for i in range(num_images):
        for j in range(i + 1, num_images):
            _, des1 = kp_descs[i]
            _, des2 = kp_descs[j]

            if des1 is None or des2 is None:
                continue

            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = image_utils.david_loew_ratio_test(matches)
            
            match_counts[i] += len(good_matches)
            match_counts[j] += len(good_matches)
    
    best_image_idx = np.argmax(match_counts)
    print(f"A melhor imagem central para o panorama é a imagem {best_image_idx + 1}.")
    return best_image_idx

def stitch_panorama_tree_based(images: list) -> np.ndarray:
    if not images:
        return None

    center_idx = find_best_center_image(images)
    center_image = images[center_idx]
    
    H_matrices = {}
    
    for i, img in enumerate(images):
        if i == center_idx:
            H_matrices[i] = np.identity(3)
            continue
        
        kp1, des1 = feature_extraction.SIFT(center_image, nfeatures=5000, contrastThreshold=0.02, edgeThreshold=15)
        kp2, des2 = feature_extraction.SIFT(img, nfeatures=5000, contrastThreshold=0.02, edgeThreshold=15)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = image_utils.david_loew_ratio_test(matches)

        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            H_matrices[i] = H
        else:
            print(f"Não há correspondências suficientes para a imagem {i+1}, pulando.")
            H_matrices[i] = None
    
    # Determine final canvas size
    h_ref, w_ref = center_image.shape[:2]
    all_corners = np.float32([[0, 0], [0, h_ref], [w_ref, h_ref], [w_ref, 0]]).reshape(-1, 1, 2)
    
    for i, img in enumerate(images):
        if i == center_idx or H_matrices.get(i) is None:
            continue
            
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H_matrices[i])
        all_corners = np.concatenate((all_corners, transformed_corners), axis=0)
        
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
    
    # Calculate translation matrix to ensure all content is visible
    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Create final canvas and stitch all images
    final_canvas = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        if H_matrices.get(i) is None:
            # Adiciona a mensagem de depuração aqui
            print(f"AVISO: Imagem {i+1} não foi inserida no panorama final.")
            continue
            
        H = H_matrices[i]
        final_H = H_translation @ H
        
        warped_img = cv2.warpPerspective(img, final_H, (xmax - xmin, ymax - ymin))
        
        # Blend the images
        mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) > 0
        final_canvas[mask] = warped_img[mask]
        
    return final_canvas

def crop_black_borders_completely(img: np.ndarray) -> np.ndarray:
    """Removes black borders from an image."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return img[y:y+h, x:x+w]

# Main script

if __name__ == "__main__":
    images_data = image_utils.load_images("T1/raw/PanoramaWebDataset")
    panorama_images = [images_data[str(i)] for i in range(1, 6)]

    final_panorama_with_borders = stitch_panorama_tree_based(panorama_images)

    if final_panorama_with_borders is not None:
        final_panorama = crop_black_borders_completely(final_panorama_with_borders)
        image_utils.save_image(final_panorama, "panorama_final_optimized")
        print("Final panorama saved successfully.")
    else:
        print("Failed to create panorama.")