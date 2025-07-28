#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : cw
# Created Date: 2025-07-28
# Updated Date: 2025-07-28
# ---------------------------------------------------------------------------
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

import sys

sys.path.append("../../federated_training")

from dataloader import DFUTissueSegNetDataLoader

rootmodelpath = "/home/chiawei/Documents/work/dfu/DFUTissueSegNet_metadata"
checkpointpath = os.path.join(rootmodelpath, "checkpoints")
predictionpath = os.path.join(rootmodelpath, "predictions")
plotpath = os.path.join(rootmodelpath, "plots")
datasetpath = os.path.join(rootmodelpath, "dataset_MiT_v3+aug-added")


allpaths = [checkpointpath, predictionpath, plotpath, datasetpath]

for path in allpaths:

    if not os.path.exists(path):

        raise FileNotFoundError(f"{path} not found")


# Parameters
BASE_MODEL = 'MiT+pscse'
ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 16
n_classes = 4
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001 # learning rate
EPOCHS = 500
WEIGHT_DECAY = 1e-5
SAVE_WEIGHTS_ONLY = True
RESIZE = (True, (256,256)) # if resize needed
TO_CATEGORICAL = True
SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = False

PERIOD = 10 # periodically save checkpoints
RAW_PREDICTION = False # if true, then stores raw predictions (i.e. before applying threshold)
RETRAIN = False

# For early stopping
EARLY_STOP = True # True to activate early stopping
PATIENCE = 50 # for early stopping

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


## Helper function: save a model

def save(model_path, epoch, model_state_dict, optimizer_state_dict):

    state = {
        'epoch': epoch + 1,
        'state_dict': deepcopy(model_state_dict),
        'optimizer': deepcopy(optimizer_state_dict),
        }

    torch.save(state, model_path)

## Loss, optimizer, metrics, and callbacks

# Loss function
dice_loss = losses.DiceLoss()
focal_loss = losses.FocalLoss()
total_loss = base.SumOfLosses(dice_loss, focal_loss)

# Metrics
metrics = [
    metricsutil.IoU(threshold=0.5),
    metricsutil.Fscore(threshold=0.5),
]

## Model Run

# Create a function to read names from a text file, and add extensions
def read_names(txt_file, ext=".png"):
  with open(txt_file, "r") as f: names = f.readlines()

  names = [name.strip("\n") for name in names] # remove newline

  # Names are without extensions. So, add extensions
  names = [name + ext for name in names]

  return names


## Training

save_dir_pred_root = predictionpath
os.makedirs(save_dir_pred_root, exist_ok = True)

aux_params=dict(
    classes=n_classes,
    activation=ACTIVATION,
    dropout=0.1, # dropout ratio, default is None
)

# create segmentation model with pretrained encoder
model = model.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    # aux_params=aux_params,
    classes=n_classes,
    activation=ACTIVATION,
    decoder_attention_type='pscse',
)

preprocessing_fn = encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

model.to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
])

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                              factor=0.1,
                              mode='min',
                              patience=10,
                              min_lr=0.00001#,
                              #verbose=True,
                              )

seed = random.randint(0, 5000)

print(f'seed: {seed}')

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


# Create a unique model name
model_name = BASE_MODEL + '_padded_aug_' + ENCODER + '_sup_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print(model_name)

# Checkpoint directory
checkpoint_loc = os.path.join(checkpointpath, model_name)

# Create checkpoint directory if does not exist
if not os.path.exists(checkpoint_loc): os.makedirs(checkpoint_loc)

# if SAVE_BEST_MODEL_ONLY: checkpoint_path = os.path.join(checkpoint_loc, 'best_model.pth')
# else: checkpoint_path = os.path.join(checkpoint_loc, "cp-{epoch:04d}.pth")

data_loader = DFUTissueSegNetDataLoader(data_path = datasetpath, num_classes = n_classes, batch_size = BATCH_SIZE)


train_loader = data_loader.get_train_loader()
valid_loader = data_loader.get_valid_loader()

# create epoch runners =========================================================
# it is a simple loop of iterating over dataloader`s samples
train_epoch = train.TrainEpoch(
    model,
    loss=total_loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = train.ValidEpoch(
    model,
    loss=total_loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# Train ========================================================================
# train model for N epochs
best_viou = 0.0
best_vloss = 1_000_000.
save_model = False # Initially start with False
cnt_patience = 0

store_train_loss, store_val_loss = [], []
store_train_iou, store_val_iou = [], []
store_train_dice, store_val_dice = [], []

for epoch in range(EPOCHS):

    print('\nEpoch: {}'.format(epoch))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # Store losses and metrics
    train_loss_key = list(train_logs.keys())[0] # first key is for loss
    val_loss_key = list(valid_logs.keys())[0] # first key is for loss

    store_train_loss.append(train_logs[train_loss_key])
    store_val_loss.append(valid_logs[val_loss_key])
    store_train_iou.append(train_logs["iou_score"])
    store_val_iou.append(valid_logs["iou_score"])
    store_train_dice.append(train_logs["fscore"])
    store_val_dice.append(valid_logs["fscore"])

    # Track best performance, and save the model's state
    if  best_vloss > valid_logs[val_loss_key]:
        best_vloss = valid_logs[val_loss_key]
        print(f'Validation loss reduced. Saving the model at epoch: {epoch:04d}')
        cnt_patience = 0 # reset patience
        best_model_epoch = epoch
        save_model = True

    # Compare iou score
    elif best_viou < valid_logs['iou_score']:
        best_viou = valid_logs['iou_score']
        print(f'Validation IoU increased. Saving the model at epoch: {epoch:04d}.')
        cnt_patience = 0 # reset patience
        best_model_epoch = epoch
        save_model = True

    else: cnt_patience += 1

    # Learning rate scheduler
    scheduler.step(valid_logs[sorted(valid_logs.keys())[0]]) # monitor validation loss

    # Save the model
    if save_model:
        save(os.path.join(checkpoint_loc, 'best_model' + '.pth'),
            epoch+1, model.state_dict(), optimizer.state_dict())
        save_model = False

    # Early stopping
    if EARLY_STOP and cnt_patience >= PATIENCE:
      print(f"Early stopping at epoch: {epoch:04d}")
      break

    # Periodic checkpoint save
    if not SAVE_BEST_MODEL:
      if (epoch+1) % PERIOD == 0:
        save(os.path.join(checkpoint_loc, f"cp-{epoch+1:04d}.pth"),
            epoch+1, model.state_dict(), optimizer.state_dict())
        print(f'Checkpoint saved for epoch {epoch:04d}')

if not EARLY_STOP and SAVE_LAST_MODEL:
    print('Saving last model')
    save(os.path.join(checkpoint_loc, 'last_model' + '.pth'),
        epoch+1, model.state_dict(), optimizer.state_dict())

print(best_model_epoch)

## Plot loss curves

# Plot loss curves =============================================================
fig, ax = plt.subplots(1,3, figsize=(12, 3))

ax[0].plot(store_train_loss, 'r')
ax[0].plot(store_val_loss, 'b')
ax[0].set_title('Loss curve')
ax[0].legend(['training', 'validation'])

ax[1].plot(store_train_iou, 'r')
ax[1].plot(store_val_iou, 'b')
ax[1].set_title('IoU curve')
ax[1].legend(['training', 'validation'])

ax[2].plot(store_train_iou, 'r')
ax[2].plot(store_val_iou, 'b')
ax[2].set_title('Dice curve')
ax[2].legend(['training', 'validation'])

fig.tight_layout()

save_fig_dir = plotpath
if not os.path.exists(save_fig_dir): os.makedirs(save_fig_dir)

fig.savefig(os.path.join(save_fig_dir, model_name + '.png'))

## Inference

# =================================== Inference ================================
# Load model====================================================================
checkpoint = torch.load(os.path.join(checkpoint_loc, 'best_model.pth'))
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
