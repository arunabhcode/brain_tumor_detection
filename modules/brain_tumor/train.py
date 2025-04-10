#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np  # Keep numpy for potential use in main script
import os

# Import custom modules
# Changed from relative to absolute imports
from logger import print
from preprocess import ImageProcessingUtils
from dataset import DatasetUtils
from vit import ViTEncoderMean
from filesystem import FilesystemUtils  # Added missing import
from introspect import Introspect  # Added missing import
# --- Configuration ---
config_dict = {}
config_dict["IMG_SIZE"] = (224, 224)
config_dict["PATCH_SIZE"] = (16, 16)
config_dict["IN_CHANNELS"] = 3
config_dict["EMBED_DIM"] = 768  # Standard ViT-Base embedding dimension
config_dict["NUM_HEADS"] = 12  # Standard ViT-Base head count
config_dict["NUM_CLASSES"] = 1  # For BCEWithLogitsLoss, output is 1 logit
config_dict["DROPOUT"] = 0.1
config_dict["BATCH_SIZE"] = 16  # Reduced batch size from original example
config_dict["NUM_EPOCHS"] = 10

# Determine patch input dimension and number of patches
patch_h, patch_w = config_dict["PATCH_SIZE"]
img_h, img_w = config_dict["IMG_SIZE"]
patch_input_dim = patch_h * patch_w * config_dict["IN_CHANNELS"]
num_patches = (img_h // patch_h) * (img_w // patch_w)

# --- Model, Loss, Optimizer ---

# Choose the desired ViT model
# Option 1: ViTEncoderMean (expects pre-patched input)
model = ViTEncoderMean(
    num_patches=num_patches,
    patch_input_dim=patch_input_dim,
    embed_dim=config_dict["EMBED_DIM"],
    num_heads=config_dict["NUM_HEADS"],
    num_classes=config_dict["NUM_CLASSES"],
    dropout=config_dict["DROPOUT"],
)


# Loss function (Binary Cross Entropy with Logits is suitable for 1 output neuron)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)  # Consider AdamW for Transformers

# --- Data Loading ---
fs_inst = FilesystemUtils()
dp_inst = ImageProcessingUtils()

# Initial logging of model summary
ist_inst = Introspect()
ist_inst.initialize(config_dict=config_dict)
ist_inst.log_model_summary(model)


ds_utils = DatasetUtils(fs_inst, dp_inst, batch_size=config_dict["BATCH_SIZE"], img_size=config_dict["IMG_SIZE"])
# Make sure DatasetUtils is initialized with consistent patch_size if needed
# ds_utils.patch_size = config_dict["PATCH_SIZE"] # Explicitly set if needed by DatasetUtils

train_loader, test_loader = ds_utils.create_dataloaders()

if train_loader is None or test_loader is None:
    print("Failed to create dataloaders. Exiting.")
    exit()  # Or handle appropriately

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# --- Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

for epoch in range(config_dict["NUM_EPOCHS"]):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Skip potentially problematic batches (e.g., from dataset __getitem__ errors)
        if inputs is None or labels is None or -1 in labels:
            print(f"Skipping problematic batch {i}")
            continue

        inputs = inputs.to(device)
        labels = (
            labels.float().unsqueeze(1).to(device)
        )  # Ensure labels are float and have shape (B, 1) for BCE

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        ist_inst.log_training_loss(running_loss/(i + 1))  # Log average loss

        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(
                f"Epoch [{epoch + 1}/{config_dict['NUM_EPOCHS']}], Step [{i + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, Avg Loss: {running_loss / (i + 1):.4f}"
            )

    # --- Evaluation Loop ---
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, labels in test_loader:
            # Skip potentially problematic batches
            if inputs is None or labels is None or -1 in labels:
                print(f"Skipping problematic test batch")
                continue

            inputs = inputs.to(device)
            labels = labels.to(device)  # Keep labels as integers for comparison

            outputs = model(inputs)  # Shape (B, 1)

            # Convert logits to probabilities and then to predictions (0 or 1)
            predicted = (
                (torch.sigmoid(outputs) > 0.5).float().squeeze()
            )  # Apply sigmoid, threshold at 0.5

            total += labels.size(0)
            correct += (
                (predicted == labels.float()).sum().item()
            )  # Compare with float labels
            # Log image predictions
            ist_inst.log_image_predictions(inputs, predicted, labels)

    accuracy = 100 * correct / total if total > 0 else 0
    ist_inst.log_accuracy(accuracy)  # Log accuracy
    print(
        f"Epoch {epoch + 1} completed. "
        f"Accuracy on test images: {accuracy:.2f} % ({correct}/{total})"
    )


# --- Save Model ---
model_save_dir = fs_inst.get_model_dir()
if model_save_dir:
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"Created model directory: {model_save_dir}")
    model_save_path = os.path.join(model_save_dir, "vit_mean_brain_tumor.ckpt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model state dictionary saved to {model_save_path}")
else:
    print("Could not determine model directory. Model not saved.")

print("Training finished.")
ist_inst.finalize()