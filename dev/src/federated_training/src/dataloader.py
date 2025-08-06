#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : cw
# Created Date: 2025-07-28
# Updated Date: 2025-08-05
# ---------------------------------------------------------------------------
from logging import getLogger

import numpy as np
import cv2
import random
import os
from torch.utils.data import Dataset as BaseDataset
from openfl.federated import PyTorchDataLoader
from torch.utils.data import DataLoader
from woundlib.thirdpartymodel.segmentation_models_pytorch import encoders
import albumentations as albu
import torch
import torch.nn.functional as F
import psutil

logger = getLogger(__name__)


class DFUTissueSegNetDataLoader(PyTorchDataLoader):
    """PyTorch data loader for DFUTissueSegNet dataset."""

    def __init__(self, data_path, num_classes, batch_size=32, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data. If None, initialize for model creation only.
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
             init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        self.feature_shape = [256, 256]
        self.num_classes = num_classes

        self.data_path = data_path# "/home/chiawei/dev/DFUTissueSegNet/DFUTissue/Labeled"#FIXMEdata_path

        self.RESIZE = (True, self.feature_shape)


        X_train, y_train, X_valid, y_valid = None, None, None, None

        self.X_train = X_train
        self.y_train = y_train

        self.X_valid = X_valid
        self.y_valid = y_valid

        self.generate_data_loader()

    def get_feature_shape(self):
        """Returns the shape of the feature array
        """
        return self.feature_shape

    def get_num_classes(self):
        """Returns the number of classes
        """
        return self.num_classes
    
    def get_train_loader(self, batch_size=None, num_batches=None):
        """Returns the data loader for the training data.

        Args:
            batch_size (int, optional): The batch size for the data loader
                (default is None).
            num_batches (int, optional): The number of batches for the data
                loader (default is None).

        Returns:
            DataLoader: The DataLoader object for the training data.
        """
        return self.train_loader

    def get_valid_loader(self, batch_size=None):
        """Returns the data loader for the validation data.

        Args:
            batch_size (int, optional): The batch size for the data loader
                (default is None).

        Returns:
            DataLoader: The DataLoader object for the validation data.
        """
        return self.valid_loader

    def generate_data_loader(self):

        if not self.data_path:
            return

        list_IDs_train = _read_names(os.path.join(self.data_path, 'labeled_train_names.txt'), ext='.png')
        list_IDs_val = _read_names(os.path.join(self.data_path, 'labeled_val_names.txt'), ext='.png')
        #list_IDs_test = _read_names(os.path.join(self.data_path, 'test_names.txt'), ext='.png')


        seed = random.randint(0, 5000)

        logger.info(f'Seed Number: {seed}')


        random.seed(seed) # seed for random number generator
        random.shuffle(list_IDs_train) # shuffle train names

        logger.info("[CW DEBUGGING] Load Data...")
        process = psutil.Process(os.getpid())
        logger.info(f"Memory used: {process.memory_info().rss / 1e6:.2f} MB")



        logger.info(f'No. of training images: {len(list_IDs_train)}')
        logger.info(f'No. of validation images: {len(list_IDs_val)}')
        #logger.info('No. of test images: ', len(list_IDs_test))

        x_train_dir = x_valid_dir = os.path.join(self.data_path, "PNGImages")
        y_train_dir = y_valid_dir = os.path.join(self.data_path, "SegmentationClass")

        # x_test_dir = os.path.join(self.data_path, "test_images")
        # y_test_dir = os.path.join(self.data_path, "test_labels")

        self.X_train = np.array([
            cv2.resize(cv2.imread(os.path.join(x_train_dir, img_name))[:, :, ::-1], tuple(self.feature_shape))
            for img_name in list_IDs_train
        ])

        self.X_valid = np.array([
            cv2.resize(cv2.imread(os.path.join(x_valid_dir, img_name))[:, :, ::-1], tuple(self.feature_shape))
            for img_name in list_IDs_val
        ])


        # Default images
        DEFAULT_IMG_TRAIN = cv2.imread(os.path.join(x_train_dir, list_IDs_train[0]))[:,:,::-1]
        DEFAULT_MASK_TRAIN = cv2.imread(os.path.join(y_train_dir, list_IDs_train[0]), 0)
        DEFAULT_IMG_VAL = cv2.imread(os.path.join(x_valid_dir, list_IDs_val[0]))[:,:,::-1]
        DEFAULT_MASK_VAL = cv2.imread(os.path.join(y_valid_dir, list_IDs_val[0]), 0)

        preprocessing_fn = encoders.get_preprocessing_fn('mit_b3', "imagenet")#ENCODER, ENCODER_WEIGHTS)



        # Dataloader ===================================================================

        train_dataset = Dataset(
            list_IDs_train,
            x_train_dir,
            y_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            to_categorical=True,
            resize=(self.RESIZE),
            n_classes=self.num_classes,
            default_img=DEFAULT_IMG_TRAIN,
            default_mask=DEFAULT_MASK_TRAIN,
        )

        valid_dataset = Dataset(
            list_IDs_val,
            x_valid_dir,
            y_valid_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            resize=(self.RESIZE),
            to_categorical=True,
            n_classes=self.num_classes,
            default_img=DEFAULT_IMG_VAL,
            default_mask=DEFAULT_MASK_VAL,
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=False,
pin_memory=False)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=False,
pin_memory=False)        


# Create a function to read names from a text file, and add extensions
def _read_names(txt_file, ext=".png"):
    with open(txt_file, "r") as f: names = f.readlines()

    names = [name.strip("\n") for name in names] # remove newline

    # Names are without extensions. So, add extensions
    names = [name + ext for name in names]

    return names

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
            logger.info(f"********** Error loading {self.ids[i]}. Using default. *********")
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
