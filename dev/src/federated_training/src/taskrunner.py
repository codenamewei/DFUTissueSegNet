# Copyright (C) 2020-2024 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from logging import getLogger
import numpy as np

from typing import Iterator, Tuple

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import Metric

import torch.optim as optim
import torch.nn.functional as F
import torch
import os
import torch
import numpy as np
from woundlib.thirdpartymodel.segmentation_models_pytorch.utils import losses, base, train
from woundlib.thirdpartymodel.segmentation_models_pytorch.utils import metrics as metricsutil
from woundlib.thirdpartymodel.segmentation_models_pytorch.decoders.unet import model
from woundlib.thirdpartymodel.segmentation_models_pytorch import encoders

import random
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from datetime import datetime


logger = getLogger(__name__)

ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
LR = 0.0001 # learning rate
WEIGHT_DECAY = 1e-5

class TemplateTaskRunner(PyTorchTaskRunner):
    """Template Task Runner for PyTorch.

    This class should be used as a template to create a custom Task Runner for your specific model and training workload.
    After generating this template, you should:
    1. Define your model, optimizer, and loss function as you would in PyTorch. PyTorchTaskRunner inherits from torch.nn.Module.
    2. Implement the `train_` and `validate_` functions to define a single train and validate epoch of your workload.
    3. Modify the `plan.yaml` file to use this Task Runner.

    The `plan.yaml` modifications should be done under the `<workspace>/plan/plan.yaml` section:
    ```
    task_runner:
        defaults : plan/defaults/task_runner.yaml
        template: src.taskrunner.TemplateTaskRunner # Modify this line appropriately if you change the class name
        settings:
            # Add additional arguments that you wish to pass through `__init__`
    ```

    Define the `forward` method of this class as a forward pass through the model.
    """

    def __init__(self, num_classes: int, device="cpu", **kwargs):
        """Initialize the Task Runner.

        Args:
            device: The hardware device to use for training (Default = "cpu").
            **kwargs: Additional arguments that may be defined in `plan.yaml`
        """
        super().__init__(device=device, **kwargs)

        self.collaborator_name = os.getenv("COLLABORATOR_NAME", "collaborator")
        logger.info(f"[INFO] TaskRunner initialized for: {self.collaborator_name}")

        self.num_classes = num_classes

        # create segmentation model with pretrained encoder
        self.model = model.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            # aux_params=aux_params,
            classes=self.num_classes,
            activation=ACTIVATION,
            decoder_attention_type='pscse',
        )

        self.epoch = 0
        preprocessing_fn = encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        self.model.to(device)


        # Optimizer
        optimizer = torch.optim.Adam([
            dict(params=self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        ])

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                    factor=0.1,
                                    mode='min',
                                    patience=10,
                                    min_lr=0.00001#,
                                    #verbose=True,
                                    )

        seed = random.randint(0, 5000)

        logger.info(f'seed: {seed}')

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Loss function
        dice_loss = losses.DiceLoss()
        focal_loss = losses.FocalLoss()
        total_loss = base.SumOfLosses(dice_loss, focal_loss)

        # Metrics
        metrics = [
            metricsutil.IoU(threshold=0.5),
            metricsutil.Fscore(threshold=0.5),
        ]

        # create epoch runners =========================================================
        # it is a simple loop of iterating over dataloader`s samples
        self.train_epoch = train.TrainEpoch(
            self.model,
            loss=total_loss,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True,
        )

        self.valid_epoch = train.ValidEpoch(
            self.model,
            loss=total_loss,
            metrics=metrics,
            device=device,
            verbose=True,
        )

        # Train ========================================================================
        # train model for N epochs
        self.best_viou = 0.0
        self.best_vloss = 1_000_000.
        self.save_model = False # Initially start with False
        self.cnt_patience = 0

        self.store_train_loss, self.store_val_loss = [], []
        self.store_train_iou, self.store_val_iou = [], []
        self.store_train_dice, self.store_val_dice = [], []



    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass.

        Returns:
            The output of the model's forward pass.
        """
        return self.model(x)

    def train_(
        self, train_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
    ) -> Metric:
        """Single Training epoch.

        Args:
            train_dataloader: Train dataset batch generator. Yields (samples, targets) tuples
                              of size = `self.train_dataloader.batch_size`.

        Returns:
            Metric: An object containing the name of the metric and its value as an np.ndarray.
        """
        # Implement training logic here and return a Metric object with the training loss.
        # Replace the following placeholder with actual training code.

        logger.info("Train current epoch: {self.epoch}...")
        train_logs = self.train_epoch.run(train_dataloader)

        # Store losses and metrics
        train_loss_key = list(train_logs.keys())[0] # first key is for loss

        self.store_train_loss.append(train_logs[train_loss_key])
        self.store_train_iou.append(train_logs["iou_score"])
        self.store_train_dice.append(train_logs["fscore"])

   
    
        return Metric(name="crossentropy_loss", value=np.array(train_loss_key))

    def validate_(
        self, validation_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
    ) -> Metric:
        """Single validation epoch.

        Args:
            validation_dataloader: Validation dataset batch generator. Yields (samples, targets) tuples.
                                   of size = `self.validation_dataloader.batch_size`.

        Returns:
            Metric: An object containing the name of the metric and its value as an np.ndarray.
        """
        # Implement validation logic here and return a Metric object with the validation accuracy.
        # Replace the following placeholder with actual validation code.


        logger.info("Validate current epoch: {self.epoch}...")

        valid_logs = self.valid_epoch.run(validation_dataloader)

        # Store losses and metrics
        val_loss_key = list(valid_logs.keys())[0] # first key is for loss

        self.store_val_loss.append(valid_logs[val_loss_key])
        self.store_val_iou.append(valid_logs["iou_score"])
        self.store_val_dice.append(valid_logs["fscore"])

        # Track best performance, and save the model's state
        if  self.best_vloss > valid_logs[val_loss_key]:
            self.best_vloss = valid_logs[val_loss_key]
            logger.info(f'Validation loss reduced. Saving the model at epoch: {self.epoch:04d}')
            cnt_patience = 0 # reset patience
            # best_model_epoch = self.epoch
            this_epoch_save_model = True

        # Compare iou score
        elif self.best_viou < valid_logs['iou_score']:
            self.best_viou = valid_logs['iou_score']
            logger.info(f'Validation IoU increased. Saving the model at epoch: {self.epoch:04d}.')
            cnt_patience = 0 # reset patience
            # best_model_epoch = self.epoch
            this_epoch_save_model = True

        else: cnt_patience += 1

        # Learning rate scheduler
        self.scheduler.step(valid_logs[sorted(valid_logs.keys())[0]]) # monitor validation loss

        if self.save_model and this_epoch_save_model:

            save_dir = "save"
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir,  f"{self.collaborator_name}.pth")#f"{self.collaborator_name}_round{self.round_num}.pth")
            torch.save(self.model.state_dict(), model_path)

        self.epoch += 1


        return Metric(name="accuracy", value=np.array(valid_logs["iou_score"])) # FIXME , not sure if its true
        #return Metric(name="accuracy", value=np.array(accuracy))
