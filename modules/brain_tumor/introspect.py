#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wandb
from logger import print

PROJECT_NAME = "brain-tumor"
ENTITY_NAME = "interview-co"

class Introspect:
    """
    Class for introspecting the model.
    """

    def __init__(self):
        """
        Initialize the introspection class.
        """
        wandb.login()
        print("Introspect initialized")

    def initialize(self, config_dict=None):
        """
        Initialize the introspection process.
        """
        wandb.init(
            project=PROJECT_NAME,
            entity=ENTITY_NAME, 
            config=config_dict,
        )

    def finalize(self):
        """
        Finalize the introspection process.
        """
        wandb.finish()
    
    def log_accuracy(self, accuracy):
        """
        Log the accuracy to Weights & Biases.
        """
        wandb.log({"accuracy": accuracy})

    def log_training_loss(self, training_loss):
        """
        Log the training_loss): to Weights & Biases.
        """
        wandb.log({"training_loss:": training_loss})

    def log_test_loss(self, test_loss):
        """
        Log the test_loss to Weights & Biases.
        """
        wandb.log({"test_loss": test_loss})

    def log_image_predictions(self, images, predictions, labels):
        """
        Log image predictions to Weights & Biases.
        """
        table = wandb.Table(columns=["Image", "Prediction", "Label"])
        for image, pred, label in zip(images, predictions, labels):
            table.add_data(wandb.Image(image.cpu().numpy() * 255), pred.cpu(), label.cpu())
        wandb.log({"predictions": table}, commit=False)

    def log_model_summary(self, model):
        """
        Log the model summary to Weights & Biases.
        """
        wandb.watch(model, log="all", log_graph=True)
    
