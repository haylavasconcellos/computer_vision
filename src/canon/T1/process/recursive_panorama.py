import numpy as np
import cv2
import sys
from canon.utils import image_utils
from canon.T1.process import feature_extraction
from canon.T1.process.blender import Blender


class ImageStitching:
    """containts the utilities required to stitch images"""

    def __init__(self, query_photo, train_photo):
        super().__init__()
        width_query_photo = query_photo.shape[1]
        width_train_photo = train_photo.shape[1]
        lowest_width = min(width_query_photo, width_train_photo)
        smoothing_window_percent = 0.10 # consider increasing or decreasing[0.00, 1.00] 
        self.smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000))

    def give_gray(self, image):
        """receives an image array and returns grayscaled image

        Args:
            image (numpy array): array of images

        Returns:
            image (numpy array): same as image input
            photo_gray (numpy array): grayscaled images
        """
        photo_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image, photo_gray



    @staticmethod
    def _sift_detector(image):
        """Applies SIFT algorithm to the given image

        Args:
            image (numpy array): input image

        Returns:
            keypoints, features
        """
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(image, None)

        return keypoints, features

    def create_and_match_keypoints(self, features_train_image, features_query_image):
        """Creates and Matches keypoints from the SIFT features using Brute Force matching
        by checking the L2 norm of the feature vector

        Args:
            features_train_image: SIFT features of train image
            features_query_image: SIFT features of query image

        Returns:
            matches (List): matches in features of train and query image
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        best_matches = bf.match(features_train_image, features_query_image)
        raw_matches = sorted(best_matches, key=lambda x: x.distance)

        return raw_matches

    def compute_homography(
        self, keypoints_train_image, keypoints_query_image, matches, reprojThresh
    ):
        """Computes the Homography to map images to a single plane,
        uses RANSAC algorithm to find the best matches iteratively.

        Args:
            keypoints_train_image: keypoints found using SIFT in train image
            keypoints_query_image: keypoints found using SIFT in query image
            matches: matches found using Brute Force
            reprojThresh: threshold for error

        Returns:
            M (Tuple): (matches, Homography matrix, status)
        """
        keypoints_train_image = np.float32(
            [keypoint.pt for keypoint in keypoints_train_image]
        )
        keypoints_query_image = np.float32(
            [keypoint.pt for keypoint in keypoints_query_image]
        )

        if len(matches) >= 4:
            points_train = np.float32(
                [keypoints_train_image[m.queryIdx] for m in matches]
            )
            points_query = np.float32(
                [keypoints_query_image[m.trainIdx] for m in matches]
            )

            H, status = cv2.findHomography(
                points_train, points_query, cv2.RANSAC, reprojThresh
            )

            return (matches, H, status)

        else:
            print(f"Minimum match count not satisfied cannot get homopgrahy")
            return None

    
def recurse_tree(image_list, blender):
    """Versão em árvore balanceada para juntar imagens."""
    n = len(image_list)
    if n == 1:
        return image_list[0]
    elif n == 2:
        result, _ = forward(image_list[0], image_list[1], blender)
        return result
    else:
        mid = n // 2
        left = recurse_tree(image_list[:mid], blender)
        right = recurse_tree(image_list[mid:], blender)
        result, _ = forward(left, right, blender)
        return result
    
def forward(query_photo, train_photo, blenderr):
    """Runs a forward pass using the ImageStitching() class in utils.py.
    Takes in a query image and train image and runs entire pipeline to return
    a panoramic image.

    Args:
        query_photo (numpy array): query image
        train_photo (nnumpy array): train image

    Returns:
        result image (numpy array): RGB result image
    """
    image_stitching = ImageStitching(query_photo, train_photo)
    _, query_photo_gray = image_stitching.give_gray(query_photo)  # left image
    _, train_photo_gray = image_stitching.give_gray(train_photo)  # right image

    keypoints_train_image, features_train_image = image_stitching._sift_detector(
        train_photo_gray
    )
    keypoints_query_image, features_query_image = image_stitching._sift_detector(
        query_photo_gray
    )

    matches = image_stitching.create_and_match_keypoints(
        features_train_image, features_query_image
    )

    mapped_feature_image = cv2.drawMatches(
                        train_photo,
                        keypoints_train_image,
                        query_photo,
                        keypoints_query_image,
                        matches[:100],
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    M = image_stitching.compute_homography(
        keypoints_train_image, keypoints_query_image, matches, reprojThresh=4
    )

    if M is None:
        return "Error cannot stitch images"

    (matches, homography_matrix, status) = M

    result = blenderr.blend_images(
        query_photo, train_photo, homography_matrix
    )

    result_rgb = np.uint8(result)
    mapped_feature_image_rgb = np.uint8(mapped_feature_image)

    return result_rgb, mapped_feature_image_rgb


def main():
    # Carrega imagens usando seu utilitário
    images_data = image_utils.load_images("T1/raw/PanoramaWebDataset")
    panorama_images = [images_data[str(i)] for i in range(1, 7)]
    
    if len(panorama_images) < 2:
        print("⚠️ É necessário pelo menos duas imagens para criar panorama.")
        return

    # Reduz tamanho das imagens para acelerar o processamento
    scale_percent = 50  # reduz para 50% do tamanho original, ajuste conforme necessário
    resized_images = []
    for img in panorama_images:
        w = int(img.shape[1] * scale_percent / 100)
        h = int(img.shape[0] * scale_percent / 100)
        resized_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)

    # Converte de BGR para RGB
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in resized_images]

    # Cria panorama usando a função recursiva
    blender = Blender("feathering")
    panorama = recurse_tree(images, blender)

    # Converte de volta para BGR para salvar com OpenCV
    panorama_bgr = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)

    # Salva o panorama final
    image_utils.save_image(panorama_bgr, "panorama_final")
    print("✅ Panorama final salvo.")


if __name__ == "__main__":
    main()