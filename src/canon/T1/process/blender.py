import cv2
import numpy as np


class Blender:
    """Class to blend two images using different blending methods"""

    def __init__(self, blending_method="feathering"):
        self.method = blending_method

    def _create_linear_mask(self, query_image, train_image, version):
        """Creates the mask using query and train images for blending the images,
        using a gaussian smoothing window/kernel

        Args:
            query_image (numpy array)
            train_image (numpy array)
            version (str) == 'left_image' or 'right_image'

        Returns:
            masks
        """
        height_query_photo = query_image.shape[0]
        width_query_photo = query_image.shape[1]
        width_train_photo = train_image.shape[1]
        height_panorama = height_query_photo
        width_panorama = width_query_photo + width_train_photo
        lowest_width = min(width_query_photo, width_train_photo)
        smoothing_window_percent = 0.10 # consider increasing or decreasing[0.00, 1.00] 
        smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000))

        offset = int(smoothing_window_size / 2)
        barrier = query_image.shape[1] - int(smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == "left_image":
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, barrier + offset :] = 1
        return cv2.merge([mask, mask, mask])

    def _blending_feathering(self, query_image, train_image, homography_matrix):
        """blends both query and train image via the homography matrix,
        and ensures proper blending and smoothing using masks created in create_masks()
        to give a seamless panorama.

        Args:
            query_image (numpy array)
            train_image (numpy array)
            homography_matrix (numpy array): Homography to map images to a single plane

        Returns:
            panoramic image (numpy array)
        """
        height_img1 = query_image.shape[0]
        width_img1 = query_image.shape[1]
        width_img2 = train_image.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self._create_linear_mask(query_image, train_image, version="left_image")
        panorama1[0 : query_image.shape[0], 0 : query_image.shape[1], :] = query_image
        panorama1 *= mask1
        mask2 = self._create_linear_mask(query_image, train_image, version="right_image")
        panorama2 = (
            cv2.warpPerspective(
                train_image, homography_matrix, (width_panorama, height_panorama)
            )
            * mask2
        )
        result = panorama1 + panorama2

        # remove extra blackspace
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1

        final_result = result[min_row:max_row, min_col:max_col, :]

        return final_result

    def _build_laplacian_pyramid(self, img, levels):
        gp = [img.astype(np.float32)]
        for _ in range(levels):
            gp.append(cv2.pyrDown(gp[-1]))
        lp = [gp[-1]]
        for i in range(levels-1, -1, -1):
            size = (gp[i].shape[1], gp[i].shape[0])
            ge = cv2.pyrUp(gp[i+1], dstsize=size)
            lp.append(cv2.subtract(gp[i], ge))
        return lp[::-1]  # from fine to coarse
    
    def _blending_multiband(self, query_image, train_image, homography_matrix, levels=5):
        """
        Performs multi-band (pyramid) blending of two images into a panorama.

        Args:
            query_image (numpy array): Left/base image
            train_image (numpy array): Right image to warp and blend
            homography_matrix (numpy array): Homography to align train_image
            levels (int): Number of pyramid levels for blending

        Returns:
            final_result (numpy array): Seamlessly blended panorama
        """
        # Step 1: Warp train_image to panorama space
        height_img1, width_img1 = query_image.shape[:2]
        height_panorama, width_panorama = height_img1, width_img1 + train_image.shape[1]

        warped_train = cv2.warpPerspective(train_image, homography_matrix,
                                        (width_panorama, height_panorama))

        # Step 2: Place query_image in panorama space
        panorama = np.zeros((height_panorama, width_panorama, 3), dtype=np.float32)
        panorama[0:height_img1, 0:width_img1, :] = query_image.astype(np.float32)

        # Step 3: Create blending mask
        mask = self._create_linear_mask(query_image, train_image, version="left_image").astype(np.float32)

        # Step 4: Build Gaussian pyramids for masks
        gp_mask = [mask]
        for _ in range(levels):
            gp_mask.append(cv2.pyrDown(gp_mask[-1]))

        # Step 5: Build Laplacian pyramids for images
        lp_panorama = self._build_laplacian_pyramid(panorama, levels)
        lp_warped_train = self._build_laplacian_pyramid(warped_train, levels)

        # Step 6: Blend pyramids using masks
        blended_pyramid = []
        for l1, l2, m in zip(lp_panorama, lp_warped_train, gp_mask):
            blended = l1 * m + l2 * (1 - m)
            blended_pyramid.append(blended)

        # Step 7: Reconstruct blended image
        result = blended_pyramid[-1]
        for i in range(levels - 1, -1, -1):
            size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
            result = cv2.pyrUp(result, dstsize=size)
            result = cv2.add(result, blended_pyramid[i])

        # Step 8: Crop black borders
        result = np.clip(result, 0, 255).astype(np.uint8)
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]

        return final_result

    def blend_images(self, query_image, train_image, homography_matrix):
        """Blends two images using the specified method.

        Args:
            query_image (numpy array): Left/base image
            train_image (numpy array): Right image to warp and blend
            homography_matrix (numpy array): Homography to align train_image
            method (str): Blending method - "feathering" or "multiband"

        Returns:
            final_result (numpy array): Blended panorama
        """
        if self.method == "feathering":
            return self._blending_feathering(query_image, train_image, homography_matrix)
        elif self.method == "multiband":
            return self._blending_multiband(query_image, train_image, homography_matrix)
        else:
            raise ValueError("Invalid blending method. Choose 'feathering' or 'multiband'.")