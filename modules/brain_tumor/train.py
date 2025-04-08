#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import torch
import cv2
import numpy as np
import sys
import inspect
import builtins as __builtins__  # Use a different name to avoid conflict


# --- Custom Print Function ---
def custom_print(*args, **kwargs):
    """Custom print function that includes filename and line number."""
    # Get the frame of the caller
    frame = inspect.currentframe().f_back
    filename = os.path.basename(frame.f_code.co_filename)
    lineno = frame.f_lineno

    # Format the prefix
    prefix = f"[{filename}:{lineno}]"

    # Call the original print function with the prefix
    __builtins__.print(prefix, *args, **kwargs)


# Overload the built-in print
print = custom_print
# --- End Custom Print Function ---


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

    def get_model_dir(self):
        """
        Get the model directory.
        """
        self.model_dir = os.path.join(self.get_repo_root(), "bin")
        return self.model_dir

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
        c, h, w = image.shape

        # Calculate the number of patches
        n_patches_h = h // patch_size[0]
        n_patches_w = w // patch_size[1]

        # Create patches
        patches = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = image[
                    :,
                    i * patch_size[0] : (i + 1) * patch_size[0],
                    j * patch_size[1] : (j + 1) * patch_size[1],
                ].flatten()
                patches.append(patch)

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

                return patch_image_tensor, label

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
        super(PatchEmbeddingFull, self).__init__()
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
        # print(x.shape)
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


class SelfAttention(torch.nn.Module):
    """
    Class for the self attention layer.
    """

    def __init__(self, embed_dim=768 // 12):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.W_q = torch.nn.Linear(embed_dim, embed_dim)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Define the forward pass
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        attention_weights = torch.softmax(
            (q @ k.transpose(1, 2)) / np.sqrt(self.embed_dim), dim=-1
        )
        attention = attention_weights @ v
        return attention


class MultiHeadAttention(torch.nn.Module):
    """
    Class for the multi head attention layer.
    """

    def __init__(self, seq_length=14, embed_dim=768, num_heads=12):
        super(MultiHeadAttention, self).__init__()
        # instatiate a list of self attention layers
        self.self_attention_layers = torch.nn.ModuleList(
            [SelfAttention(embed_dim // num_heads) for _ in range(num_heads)]
        )
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.W_o = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        batch_size, seq_length, _ = x.shape

        # Split embedding dim into num_heads pieces
        head_dim = self.embed_dim // self.num_heads
        x_split = x.reshape(batch_size, seq_length, self.num_heads, head_dim)

        # Process each head separately through attention layers
        head_outputs = []
        for i, layer in enumerate(self.self_attention_layers):
            head_i = x_split[:, :, i, :]  # Get i-th head
            head_output = layer(head_i)  # Apply attention
            head_outputs.append(head_output)

        # Concatenate all head outputs
        concat_heads = torch.cat(head_outputs, dim=-1)

        # Apply output projection
        output = self.W_o(concat_heads)

        return output


class FeedForward(torch.nn.Module):
    """
    Class for the feed forward layer.
    """

    def __init__(self, embed_dim=768, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = torch.nn.Linear(embed_dim * 4, embed_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # Define the forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# TODO: Implement one more approach of multi head attention with all in one multiplications instead of separate Attention layers
# TODO: Implement the complete Encoder block
# Training and testing the model


class ViTEncoderMean(torch.nn.Module):
    """
    Class for the Vision Transformer Encoder model with a classification head.
    Uses MEAN POOLING of output tokens for classification.
    """

    def __init__(
        self,
        seq_length=196,
        embed_dim=768,
        num_heads=12,
        num_classes=2,
        patch_input_dim=768,
    ):
        super(ViTEncoderMean, self).__init__()
        # Note: This assumes input x to forward is already patch-embedded and flattened (B, num_patches, patch_input_dim)
        self.patch_embedding = PatchEmbedding(
            in_channels=patch_input_dim, out_channels=embed_dim
        )
        self.position_embedding = PositionEmbedding(seq_length, embed_dim)
        self.multi_head_attention = MultiHeadAttention(seq_length, embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim)
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.classifier = torch.nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_patches, patch_input_dim)
        embedded_x = self.patch_embedding(x)
        x_with_pos = embedded_x + self.position_embedding(embedded_x)

        # Pass through transformer encoder layers
        attn_output = self.multi_head_attention(self.layer_norm1(x_with_pos))
        x = x_with_pos + attn_output
        ff_output = self.feed_forward(self.layer_norm2(x))
        encoded_output = x + ff_output

        # Perform classification using mean pooling
        mean_output = encoded_output.mean(dim=1)
        logits = self.classifier(mean_output)
        return logits


class ViTEncoderCLS(torch.nn.Module):
    """
    Class for the Vision Transformer Encoder model with a classification head.
    Uses the standard [CLS] token approach for classification.
    Input is expected to be an image tensor (B, C, H, W).
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        in_channels=3,
        embed_dim=768,
        num_heads=12,
        num_classes=2,
        dropout=0.1,
    ):
        super(ViTEncoderCLS, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Calculate number of patches
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )
        self.seq_length = self.num_patches + 1  # Add 1 for CLS token

        # Patch embedding (using Conv2D)
        self.patch_embedding = PatchEmbeddingFull(in_channels, embed_dim, patch_size)

        # CLS token
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (for CLS token + patches)
        self.position_embedding = PositionEmbedding(self.seq_length, embed_dim)

        # Transformer Encoder parts
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.multi_head_attention = MultiHeadAttention(
            self.seq_length, embed_dim, num_heads
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, dropout)

        # Classification head
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers and cls_token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # You might want more sophisticated initialization for other layers (e.g., Xavier/Kaiming for Conv/Linear)

    def forward(self, x):
        # x shape: (batch_size, in_channels, img_height, img_width)
        batch_size = x.shape[0]

        # 1. Patch Embedding
        patches = self.patch_embedding(x)
        patches = patches.flatten(2)
        patches = patches.transpose(1, 2)

        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patches), dim=1)

        # 3. Add Positional Embedding
        tokens = self.position_embedding(tokens)

        # 4. Transformer Encoder Block
        # MHA + Dropout + Add & Norm
        attn_input = self.layer_norm1(tokens)
        attn_output = self.multi_head_attention(attn_input)
        attn_output = self.dropout(attn_output)
        x = tokens + attn_output

        # FeedForward + Dropout + Add & Norm
        ff_input = self.layer_norm2(x)
        ff_output = self.feed_forward(ff_input)
        encoded_output = x + ff_output

        # 5. Classification
        cls_output = encoded_output[:, 0]
        logits = self.classifier(cls_output)

        return logits


NUM_EPOCHS = 10

model = ViTEncoderMean()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Example usage
fs_inst = FilesystemUtils()
dp_inst = ImageProcessingUtils()
ds_utils = DatasetUtils(fs_inst, dp_inst, batch_size=16)
train_loader, test_loader = ds_utils.create_dataloaders()
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

for epoch in range(NUM_EPOCHS):
    model.train()  # Set the model to training mode
    for i, (images, labels) in enumerate(train_loader):

        outputs = model(images)
        labels = labels.float().unsqueeze(1)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, NUM_EPOCHS, i + 1, len(train_loader), loss.item()
                )
            )

    model.eval()  # Set the model to evaluation mode
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the test images: {} %".format(
                100 * correct / total
            )
        )

# Save the model checkpoint
# torch.save(model.state_dict(), fs_inst.get_model_dir() + "/model.ckpt")
