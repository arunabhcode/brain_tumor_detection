#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import cv2
from logger import print  # Import custom print
from filesystem import FilesystemUtils
from preprocess import ImageProcessingUtils


class DatasetUtils:
    """
    This class will create the train and test dataloaders using torch
    """

    def __init__(
        self,
        fs_utils: FilesystemUtils,
        img_utils: ImageProcessingUtils,
        batch_size=32,
        img_size=(224, 224),
    ):
        self.fs_utils = fs_utils
        self.img_utils = img_utils
        self.batch_size = batch_size
        self.img_size = img_size
        # Define patch size consistently, maybe pass as arg or get from img_utils/config
        self.patch_size = (16, 16)

    def create_dataset(self):
        """
        Create custom dataset class for brain tumor images
        """

        class BrainTumorDataset(torch.utils.data.Dataset):
            def __init__(
                self, file_paths, img_utils, img_size=(224, 224), patch_size=(16, 16)
            ):
                self.file_paths = file_paths
                self.img_utils = img_utils
                self.img_size = img_size
                self.patch_size = patch_size  # Store patch size

            def __len__(self):
                return len(self.file_paths)

            def __getitem__(self, idx):
                img_path = self.file_paths[idx]

                # Read and process image
                image = self.img_utils.read_image(img_path)
                if image is None:
                    # Handle case where image reading failed
                    # Return None or raise error, or return a placeholder?
                    # For now, print warning and return None (will cause issues in DataLoader)
                    # A better approach: Filter out bad paths beforehand or handle Nones in collate_fn
                    print(f"Warning: Skipping image {img_path} due to read error.")
                    # Need to return something the DataLoader can handle, perhaps skip?
                    # This requires a custom collate_fn or filtering `file_paths`.
                    # Let's return a dummy tensor and label for now, assuming filtering happens later.
                    dummy_patch_dim = (
                        self.patch_size[0] * self.patch_size[1] * 3
                    )  # Assuming 3 channels
                    num_patches = (self.img_size[0] // self.patch_size[0]) * (
                        self.img_size[1] // self.patch_size[1]
                    )
                    return (
                        torch.zeros((num_patches, dummy_patch_dim)),
                        -1,
                    )  # Dummy label -1

                image = self.img_utils.resize_image(image, self.img_size)
                image = self.img_utils.rescale_intensity(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_np = np.transpose(image, (2, 0, 1))  # Change to CxHxW
                image_tensor = torch.from_numpy(image_np).float() / 255.0

                # Patching expects numpy array or tensor (C, H, W)
                patch_image_np = self.img_utils.patch_and_stack(
                    image_tensor, self.patch_size
                )
                if patch_image_np is None or patch_image_np.size == 0:
                    print(f"Warning: Skipping image {img_path} due to patching error.")
                    dummy_patch_dim = self.patch_size[0] * self.patch_size[1] * 3
                    num_patches = (self.img_size[0] // self.patch_size[0]) * (
                        self.img_size[1] // self.patch_size[1]
                    )
                    return torch.zeros((num_patches, dummy_patch_dim)), -1

                patch_image_tensor = torch.from_numpy(patch_image_np).float()

                # Create label (0 if no tumor, 1 if tumor)
                label = 0 if "notumor" in img_path.lower() else 1

                return patch_image_tensor, image, label

        return BrainTumorDataset

    def create_dataloaders(self):
        """
        Create train and test dataloaders based on file paths
        """
        # Get all data files
        all_files = self.fs_utils.get_data_files()
        if not all_files:
            print("Error: No data files found. Cannot create dataloaders.")
            return None, None

        # Split into train and test files
        train_files = [f for f in all_files if "Training" in f]
        test_files = [f for f in all_files if "Testing" in f]

        if not train_files:
            print("Warning: No training files found.")
        if not test_files:
            print("Warning: No testing files found.")

        print(
            f"Found {len(train_files)} training files and {len(test_files)} testing files"
        )

        # Create datasets
        BrainTumorDataset = self.create_dataset()
        train_dataset = BrainTumorDataset(
            train_files, self.img_utils, self.img_size, self.patch_size
        )
        test_dataset = BrainTumorDataset(
            test_files, self.img_utils, self.img_size, self.patch_size
        )

        # Determine num_workers based on platform
        num_workers = 2 if torch.cuda.is_available() else 0  # Simple check
        pin_memory = torch.cuda.is_available()

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # Consider adding a collate_fn to handle potential None values from __getitem__
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # Consider adding a collate_fn
        )

        return train_loader, test_loader
