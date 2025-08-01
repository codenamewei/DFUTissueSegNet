# /home/chiawei/temp/wound_tissue_segmentation
# - checkpoints
# - predictions
# - plots
# mkdir -p dataset_MiT_v3+aug-added/{PNGImages,SegmentationClass,test_images,test_labels}

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import cv2
import numpy as np


import random
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
import copy
#import sys


#sys.path.append("../../Codes")

from woundlib.thirdpartymodel.segmentation_models_pytorch.utils import losses, base, train
from woundlib.thirdpartymodel.segmentation_models_pytorch.utils import metrics as metricsutil
from woundlib.thirdpartymodel.segmentation_models_pytorch.decoders.unet import model
from woundlib.thirdpartymodel.segmentation_models_pytorch import encoders

rootmodelpath = "/home/chiawei/data"#"/home/chiawei/Documents/work/dfu/DFUTissueSegNet_metadata/"
checkpointpath = os.path.join(rootmodelpath, "dfusegnet-pred",  "checkpoints")
predictionpath = os.path.join(rootmodelpath, "dfusegnet-pred", "predictions")
plotpath = os.path.join(rootmodelpath, "dfusegnet-pred", "plots")
datasetpath = os.path.join(rootmodelpath, "dfusegnet")
repodatapath = datasetpath#"../../DFUTissue/Labeled"


label_color_keyvalue = {
    0: [0, 0, 0],         # background
    1: [0, 255, 0],       # granulation tissue
    2: [0, 0, 255],       # callus
    3: [255, 0, 0],       # fibrin
    4: [255, 255, 0],     # necrotic tissue (yellow)
    5: [255, 0, 255],     # eschar (magenta)
    6: [0, 255, 255],     # neodermis (cyan)
    7: [128, 0, 128],     # tendon (purple)
    8: [255, 165, 0]      # dressing (orange)
}

# Checkpoint directory
checkpoint_loc = os.path.join(rootmodelpath, "checkpoints", "MiT+pscse_padded_aug_mit_b3_sup_2025-07-18_00-00-35")

allpaths = [checkpointpath, predictionpath, plotpath, datasetpath]

for path in allpaths:

    if not os.path.exists(path):

        raise FileNotFoundError(f"{path} not found")

## Dataloader 

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            list_IDs,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
            to_categorical:bool=False,
            resize=(True, (256, 256)), # To resize, the first value has to be True
            n_classes:int=6,
            default_img=None,
            default_mask=None,
    ):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.to_categorical = to_categorical
        self.resize = resize
        self.n_classes = n_classes
        self.default_img = default_img
        self.default_mask = default_mask
        
    def __getitem__(self, i):


        try:
            # Read image and mask
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)
        except Exception as e:
            print(f"********** Error loading {self.ids[i]}. Using default. *********")
            image = self.default_img.copy()
            mask = self.default_mask.copy()
    
        # ✅ Always resize, including default fallback
        if self.resize[0]:
            image = cv2.resize(image, self.resize[1], interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.resize[1], interpolation=cv2.INTER_NEAREST)
    
        mask = np.expand_dims(mask, axis=-1)
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
    
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
    
        # Convert to one-hot if requested
        if self.to_categorical:
            mask = torch.from_numpy(mask)
            mask = F.one_hot(mask.long(), num_classes=self.n_classes)
            mask = mask.type(torch.float32)
            mask = mask.numpy()
            mask = np.squeeze(mask)
            mask = np.moveaxis(mask, -1, 0)
    
        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [

        albu.OneOf(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
            ],
            p=0.8,
        ),

        albu.OneOf(
            [
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0), # scale only
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0), # rotate only
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0), # shift only
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0), # affine transform
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Perspective(p=0.2),
                albu.GaussNoise(p=0.2),
                albu.Sharpen(p=0.2),
                albu.Blur(blur_limit=3, p=0.2),
                albu.MotionBlur(blur_limit=3, p=0.2),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.CLAHE(p=0.25),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                albu.RandomGamma(p=0.25),
                albu.HueSaturationValue(p=0.25),
            ],
            p=0.3,
        ),

    ]

    return albu.Compose(train_transform, p=0.9) # 90% augmentation probability

## Augmentation

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(512, 512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# Parameters
BASE_MODEL = 'MiT+pscse'
ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 16
n_classes = 4
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                              min_lr=0.00001)

seed = random.randint(0, 5000)

print(f'seed: {seed}')

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

x_train_dir = x_valid_dir = os.path.join(datasetpath, "PNGImages")
y_train_dir = y_valid_dir = os.path.join(datasetpath, "SegmentationClass")

x_test_dir = os.path.join(datasetpath, "test_images")
y_test_dir = os.path.join(datasetpath, "test_labels")

# Read train, test, and val names
#dir_txt = '/content/drive/MyDrive/Wound_tissue_segmentation/Dataset/dataset_MiT_v3+aug-added'
list_IDs_train = read_names(os.path.join(repodatapath, 'labeled_train_names.txt'), ext='.png')
list_IDs_val = read_names(os.path.join(repodatapath, 'labeled_val_names.txt'), ext='.png')
list_IDs_test = read_names(os.path.join(repodatapath, 'test_names.txt'), ext='.png')

random.seed(seed) # seed for random number generator
random.shuffle(list_IDs_train) # shuffle train names

print('No. of training images: ', len(list_IDs_train))
print('No. of validation images: ', len(list_IDs_val))
print('No. of test images: ', len(list_IDs_test))

# Create a unique model name
model_name = BASE_MODEL + '_padded_aug_' + ENCODER + '_sup_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print(model_name)

# Default images
DEFAULT_IMG_TRAIN = cv2.imread(os.path.join(x_train_dir, list_IDs_train[0]))[:,:,::-1]
DEFAULT_MASK_TRAIN = cv2.imread(os.path.join(y_train_dir, list_IDs_train[0]), 0)
DEFAULT_IMG_VAL = cv2.imread(os.path.join(x_valid_dir, list_IDs_val[0]))[:,:,::-1]
DEFAULT_MASK_VAL = cv2.imread(os.path.join(y_valid_dir, list_IDs_val[0]), 0)


# Create checkpoint directory if does not exist
#if not os.path.exists(checkpoint_loc): os.makedirs(checkpoint_loc)

# if SAVE_BEST_MODEL_ONLY: checkpoint_path = os.path.join(checkpoint_loc, 'best_model.pth')
# else: checkpoint_path = os.path.join(checkpoint_loc, "cp-{epoch:04d}.pth")

# Dataloader ===================================================================
train_dataset = Dataset(
    list_IDs_train,
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    to_categorical=TO_CATEGORICAL,
    resize=(RESIZE),
    n_classes=n_classes,
    default_img=DEFAULT_IMG_TRAIN,
    default_mask=DEFAULT_MASK_TRAIN,
)

valid_dataset = Dataset(
    list_IDs_val,
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    resize=(RESIZE),
    to_categorical=TO_CATEGORICAL,
    n_classes=n_classes,
    default_img=DEFAULT_IMG_VAL,
    default_mask=DEFAULT_MASK_VAL,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

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



## Inference

# =================================== Inference ================================
# Load model====================================================================
checkpoint = torch.load(os.path.join(checkpointpath, 'collaborator.pth'))
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Test dataloader ==============================================================
test_dataset = Dataset(
    list_IDs_test,
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    resize=(RESIZE),
    to_categorical=False, # don't convert to onehot now
    n_classes=n_classes,
)

test_dataloader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=6)

# Prediction ===================================================================
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import scipy.io as sio

import warnings
warnings.filterwarnings("ignore")

save_pred = True
threshold = 0.5
ep = 1e-6
raw_pred = []

HARD_LINE = True

# Save directory
# save_dir_pred = '/content/drive/MyDrive/Wound_tissue_segmentation/predictions/' + model_name
# save_dir_pred_pal = '/content/drive/MyDrive/Wound_tissue_segmentation/predictions_palette/' + model_name
# save_dir_pred_pal_cat = '/content/drive/MyDrive/Wound_tissue_segmentation/predictions_palette_cat/' + model_name
save_dir_pred = os.path.join(rootmodelpath, "predictions", model_name)
save_dir_pred_pal = os.path.join(rootmodelpath, "predictions_palette", model_name)
save_dir_pred_pal_cat = os.path.join(rootmodelpath, "predictions_palette_cat", model_name)

save_dir_pred_overlay = os.path.join(save_dir_pred_pal_cat, "overlay_dir")
save_dir_pred_contour = os.path.join(save_dir_pred_pal_cat, "contour_dir")

if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)
if not os.path.exists(save_dir_pred_pal): os.makedirs(save_dir_pred_pal)
if not os.path.exists(save_dir_pred_pal_cat): os.makedirs(save_dir_pred_pal_cat)

# --- Overlay silhouette and contours ---
if not os.path.exists(save_dir_pred_overlay): os.makedirs(save_dir_pred_overlay)
if not os.path.exists(save_dir_pred_contour): os.makedirs(save_dir_pred_contour)

# Create a dictionary to store metrics
metric = {} # Nested metric format: metric[image_name][label] = [precision, recall, dice, iou]

# fig, ax = plt.subplots(5,2, figsize=(10,15))
iter_test_dataloader = iter(test_dataloader)

palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

stp, stn, sfp, sfn = 0, 0, 0, 0

for i in range(len(list_IDs_test)):

    tp, tn, fp, fn = 0, 0, 0, 0

    name = os.path.splitext(list_IDs_test[i])[0] # remove extension
    metric[name] = {}

    i_mp, i_mr, i_mdice, i_miou = [], [], [], []

    image, gt_mask = next(iter_test_dataloader)

    pr_mask = model.predict(image.to(DEVICE))

    if TO_CATEGORICAL:
        pr_mask = torch.argmax(pr_mask, dim=1)

    gt_mask = gt_mask.squeeze().cpu().numpy()
    gt_mask = np.asarray(gt_mask, dtype=np.int64)
    pred = pr_mask.squeeze().cpu().numpy()

    if RAW_PREDICTION:
        raw_pred.append(pred)

    if save_pred:
        # --- Save raw prediction ---
        cv2.imwrite(os.path.join(save_dir_pred, list_IDs_test[i]), np.squeeze(pred).astype(np.uint8))
        print(f"Compare: Write to {os.path.join(save_dir_pred, list_IDs_test[i])}")

        debug_image_path = os.path.join(x_test_dir, list_IDs_test[i])
        raw_color_img = cv2.imread(debug_image_path)
        prediction_sized_raw_color_img= cv2.resize(raw_color_img.copy(), (256, 256))
        # print(f"[DEBUGGING] {os.path.exists(debug_image_path)}")
        # print(f"[DEBUGGING] {debug_image_path}")

        # --- Palette masks ---
        pal_gt_mask = np.squeeze(gt_mask).astype(np.uint8)
        pal_gt_mask = Image.fromarray(pal_gt_mask).convert("P")
        pal_gt_mask.putpalette(np.array(palette, dtype=np.uint8))

        pal_pred = np.squeeze(pred).astype(np.uint8)
        pal_pred = Image.fromarray(pal_pred).convert("P")
        pal_pred.putpalette(np.array(palette, dtype=np.uint8))

        pal_pred.save(os.path.join(save_dir_pred_pal, list_IDs_test[i]))

        concat_pals = Image.new("RGB", (pal_gt_mask.width*2, pal_gt_mask.height), "white")
        concat_pals.paste(pal_gt_mask, (0, 0))
        concat_pals.paste(pal_pred, (pal_gt_mask.width, 0))
        concat_pals.save(os.path.join(save_dir_pred_pal_cat, list_IDs_test[i]))        
        
        # Convert original image tensor to numpy [H, W, 3]
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    
        img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)

    
        # Initialize silhouette image
        silhouette_base_image = prediction_sized_raw_color_img.copy()
        silhouette = np.zeros_like(img_np)

        for label_val in np.unique(pred):
            if label_val == 0: continue  # skip background
            silhouette[pred == label_val] = label_color_keyvalue[label_val]
    
        blended = cv2.addWeighted(silhouette_base_image, 0.3, silhouette, 0.7, 0)
        cv2.imwrite(os.path.join(save_dir_pred_overlay, list_IDs_test[i]), blended)
        print(f"Silhouette: Write to {os.path.join(save_dir_pred_overlay, list_IDs_test[i])}")

        # Initialize silhouette image
        # silhouette_base_image = np.ones((256, 256,3), dtype = np.uint8) * 255#prediction_sized_raw_color_img.copy()
        # silhouette = np.zeros_like(img_np)

        # for label_val in np.unique(pred):
        #     if label_val == 0: continue  # skip background
        #     color = [255, 0, 0]  # red overlay
        #     silhouette[pred == label_val] = color

        # blended = cv2.addWeighted(silhouette_base_image, 0.7, silhouette, 0.3, 0)
        # cv2.imwrite(os.path.join(save_dir_pred_overlay, list_IDs_test[i]), blended)
        # print(f"Write to {os.path.join(save_dir_pred_overlay, list_IDs_test[i])}")

        # Contour drawing
        contour_base_image = prediction_sized_raw_color_img.copy()
        for label_val in np.unique(pred):
            if label_val == 0: continue
            mask_uint8 = (pred == label_val).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_base_image, contours, -1, (0, 255, 0), 2)  # green contour

        cv2.imwrite(os.path.join(save_dir_pred_contour, list_IDs_test[i]), contour_base_image)
        print(f"Contour: Write to {os.path.join(save_dir_pred_contour, list_IDs_test[i])}")
