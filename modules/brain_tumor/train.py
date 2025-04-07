#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import torch
import cv2
import numpy as np


class FilesystemUtils:

    def __init__(self):
        self.data_ext = ".jpg"

    def get_repo_root(self):
        """
        Get the root directory of the repository.
        """
        # Get the current working directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        # Traverse up the directory tree until we find the root directory
        while True:
            if os.path.exists(os.path.join(self.current_dir, ".git")):
                return self.current_dir
            parent_dir = os.path.dirname(self.current_dir)
            if parent_dir == self.current_dir:
                break
            self.current_dir = parent_dir
        return None

    def get_data_dir(self):
        """
        Get the data directory.
        """
        self.data_dir = os.path.join(self.get_repo_root(), "data")
        return self.data_dir

    def get_data_files(self):
        """
        Get all data files in the data directory.
        """
        self.data_dir = self.get_data_dir()
        print(f"data_dir = {self.data_dir + "/**/*" + self.data_ext}")
        self.data_files = glob.glob(
            self.data_dir + "/**/*" + self.data_ext, recursive=True
        )
        return self.data_files


class ImageProcessingUtils:

    def __init__(self):
        pass

    def read_image(self, file_path):
        """
        Read an image from a file path.
        """
        image = cv2.imread(file_path)
        return image

    def show_image(self, image):
        """
        Show an image.
        """
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rescale_intensity(self, image):
        """
        Rescale the intensity of an image.
        """
        min_val = image.min()
        max_val = image.max()
        scaled_image = (image - min_val) / (max_val - min_val) * 255
        return scaled_image.astype("uint8")

    def resize_image(self, image, new_size=(224, 224)):
        """
        Resize an image to a new size.
        """
        resized_image = cv2.resize(image, new_size)
        return resized_image

    def patch_and_stack(self, image, patch_size=(16, 16)):
        """
        Patch and stack an image.
        """
        # Get the dimensions of the image
        h, w, c = image.shape

        # Calculate the number of patches
        n_patches_h = h // patch_size[0]
        n_patches_w = w // patch_size[1]

        # Create patches
        patches = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = image[
                    i * patch_size[0] : (i + 1) * patch_size[0],
                    j * patch_size[1] : (j + 1) * patch_size[1],
                ]
                patches.append(patch.flatten())

        # Stack patches
        stacked_patches = np.array(patches).reshape(
            n_patches_h * n_patches_w, patch_size[0] * patch_size[1] * c
        )
        return stacked_patches


class DatasetUtils:
    """
    This class will create the train and test dataloaders using torch
    """

    def __init__(self, fs_utils, img_utils, batch_size=32, img_size=(224, 224)):
        self.fs_utils = fs_utils
        self.img_utils = img_utils
        self.batch_size = batch_size
        self.img_size = img_size

    def create_dataset(self):
        """
        Create custom dataset class for brain tumor images
        """

        class BrainTumorDataset(torch.utils.data.Dataset):
            def __init__(self, file_paths, img_utils, img_size=(224, 224)):
                self.file_paths = file_paths
                self.img_utils = img_utils
                self.img_size = img_size

            def __len__(self):
                return len(self.file_paths)

            def __getitem__(self, idx):
                img_path = self.file_paths[idx]

                # Read and process image
                image = self.img_utils.read_image(img_path)
                image = self.img_utils.resize_image(image, self.img_size)
                image = self.img_utils.rescale_intensity(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.transpose(image, (2, 0, 1))  # Change to CxHxW
                image = torch.from_numpy(image).float() / 255.0
                patch_image_tensor = self.img_utils.patch_and_stack(image)
                patch_image_tensor = torch.from_numpy(patch_image_tensor)

                # Create label (0 if no tumor, 1 if tumor)
                label = 0 if "notumor" in img_path.lower() else 1

                return image, label

        return BrainTumorDataset

    def create_dataloaders(self):
        """
        Create train and test dataloaders based on file paths
        """
        # Get all data files
        all_files = self.fs_utils.get_data_files()

        # Split into train and test files
        train_files = [f for f in all_files if "Training" in f]
        test_files = [f for f in all_files if "Testing" in f]

        print(
            f"Found {len(train_files)} training files and {len(test_files)} testing files"
        )

        # Create datasets
        BrainTumorDataset = self.create_dataset()
        train_dataset = BrainTumorDataset(train_files, self.img_utils, self.img_size)
        test_dataset = BrainTumorDataset(test_files, self.img_utils, self.img_size)

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, test_loader


class PatchEmbeddingFull(torch.nn.Module):
    """
    Class for the patch embedding layer.
    """

    def __init__(self, in_channels=3, out_channels=768, patch_size=(16, 16)):
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        # Define the convolutional layer for patch embedding
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x):
        # Define the forward pass
        x = self.conv(x)
        return x


class PatchEmbedding(torch.nn.Module):
    """
    Class for the patch embedding layer.
    """

    def __init__(self, in_channels=16 * 16 * 3, out_channels=768):
        super(PatchEmbedding, self).__init__()
        # Define the convolutional layer for patch embedding
        self.embed_matrix = torch.nn.Linear(
            in_channels,
            out_channels,
        )

    def forward(self, x):
        # Define the forward pass
        x = self.embed_matrix(x)
        return x


class PositionEmbedding(torch.nn.Module):
    """
    Class for the position embedding layer.
    """

    def __init__(self, seq_length=14, embed_dim=768):
        super(PositionEmbedding, self).__init__()

        pe = torch.zeros(seq_length, embed_dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_length, embed_dim)
        # Register the positional embedding matrix as a buffer
        # so that it is not considered a model parameter
        # and does not require gradients
        # Define the positional embedding matrix
        # self.register_buffer("pe", pe)
        self.pe = torch.nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        # Define the forward pass
        x = x + self.pe[:, : x.size(1)]
        return x


# TODO: Implement two approaches of multi head attention, one with separate attention and multi head attention and one with all in one
# TODO: Implement the complete Encoder block
# Training and testing the model

# class VisionTransformer(torch.nn.Module):
#     """
#     Class for the Vision Transformer model.
#     """

#     def __init__(self, seq_length=14, num_classes=2):
#         super(VisionTransformer, self).__init__()
#         # Define y
#         #
#         # our Vision Transformer architecture here
#         self.embedding_dim = 768
#         self.n_heads = 12

#     def forward(self, x):
#         # Define the forward pass
#         pass


# # Example usage
# fs_inst = FilesystemUtils()
# dp_inst = ImageProcessingUtils()
# data_files = fs_inst.get_data_files()
# print(dp_inst.read_image(data_files[1]).shape)

# # Create dataset utils and dataloaders
# ds_utils = DatasetUtils(fs_inst, dp_inst, batch_size=16)
# train_loader, test_loader = ds_utils.create_dataloaders()
# print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

if __name__ == "__main__":
    # Example usage
    pe_inst = PositionEmbedding(seq_length=2, embed_dim=4)
