#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class ImageProcessingUtils:

    def __init__(self):
        pass

    def read_image(self, file_path):
        """
        Read an image from a file path.
        """
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Could not read image at {file_path}")
        return image

    def show_image(self, image):
        """
        Show an image.
        """
        if image is None:
            print("Warning: Cannot show a None image.")
            return
        try:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"Error displaying image: {e}")


    def rescale_intensity(self, image):
        """
        Rescale the intensity of an image to 0-255.
        """
        if image is None:
            print("Warning: Cannot rescale intensity of a None image.")
            return None
        min_val = image.min()
        max_val = image.max()
        if max_val == min_val:
            # Avoid division by zero if the image is flat
            return image.astype("uint8")
        scaled_image = (image - min_val) / (max_val - min_val) * 255
        return scaled_image.astype("uint8")

    def resize_image(self, image, new_size=(224, 224)):
        """
        Resize an image to a new size.
        """
        if image is None:
            print("Warning: Cannot resize a None image.")
            return None
        resized_image = cv2.resize(image, new_size)
        return resized_image

    def patch_and_stack(self, image_tensor, patch_size=(16, 16)):
        """
        Patch and stack an image tensor (C, H, W).
        Input: torch tensor (C, H, W)
        Output: numpy array (num_patches, patch_height * patch_width * C)
        """
        if not isinstance(image_tensor, np.ndarray):
             # Convert tensor to numpy array if it's not already
            if hasattr(image_tensor, 'numpy'):
                image = image_tensor.numpy()
            else:
                print("Error: Input must be a NumPy array or a PyTorch tensor.")
                return None
        else:
            image = image_tensor

        # Ensure image is in C, H, W format for calculation
        if image.shape[0] != 3 and image.shape[2] == 3: # If H, W, C format, transpose
             image = np.transpose(image, (2, 0, 1))
        elif image.shape[0] != 3:
             print(f"Error: Expected input shape (C, H, W) or (H, W, C), but got {image.shape}")
             return None

        c, h, w = image.shape
        patch_h, patch_w = patch_size

        # Calculate the number of patches
        if h % patch_h != 0 or w % patch_w != 0:
            print(f"Warning: Image dimensions ({h}, {w}) are not perfectly divisible by patch size ({patch_h}, {patch_w}). Cropping might occur or consider padding.")

        n_patches_h = h // patch_h
        n_patches_w = w // patch_w

        # Create patches
        patches = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = image[
                    :,
                    i * patch_h : (i + 1) * patch_h,
                    j * patch_w : (j + 1) * patch_w,
                ].flatten() # Flatten includes channels
                patches.append(patch)

        # Stack patches: Resulting shape (num_patches, patch_h * patch_w * c)
        if not patches:
            print("Warning: No patches were created. Check image and patch dimensions.")
            return np.array([])

        stacked_patches = np.array(patches)
        return stacked_patches 