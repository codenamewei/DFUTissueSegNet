import cv2
import numpy as np

from segmentation_models_pytorch.decoders.unet import model
from segmentation_models_pytorch import encoders
import torch
import os

from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import albumentations as albu

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

# Create a function to read names from a text file, and add extensions
def read_names(txt_file, ext=".png"):
  with open(txt_file, "r") as f: names = f.readlines()

  names = [name.strip("\n") for name in names] # remove newline

  # Names are without extensions. So, add extensions
  names = [name + ext for name in names]

  return names

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
                
def get_image_with_contour(rgb : np.ndarray, mask: np.ndarray, contour_color: set = (0, 0, 255)) -> np.ndarray:

    # Make a copy of the original image for drawing
    rgb_contour = rgb.copy()
    
    # Ensure masks are uint8 binary
    mask_uint8 = (mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the RGB copy
    cv2.drawContours(rgb_contour, contours, -1, color=contour_color, thickness=2)   # blue

    return rgb_contour

def torch_tensor_to_opencv_image(tensor: torch.tensor):

    # Step 1: Remove batch dimension -> (3, 256, 256)
    image = tensor.squeeze(0)

    # Step 2: Convert from CHW to HWC -> (256, 256, 3)
    image = image.permute(1, 2, 0).numpy()

    # Step 3: Convert from RGB to BGR
    image = image[:, :, ::-1]

    # Step 4: Scale to [0, 255] and convert to uint8
    return (image * 255).astype(np.uint8)
if __name__ == "__main__":


    ENCODER = 'mit_b3'
    ENCODER_WEIGHTS = 'imagenet'
    n_classes = 4
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    LR = 0.0001 # learning rate
    WEIGHT_DECAY = 1e-5
    repodatapath = "../DFUTissue/Labeled"
    rootmodelpath = "/home/chiawei/temp/wound_tissue_segmentation"
    RESIZE = (True, (256,256)) # if resize needed
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TO_CATEGORICAL = True
    colors = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]

    datasetpath = os.path.join(rootmodelpath, "dataset_MiT_v3+aug-added")

    x_test_dir = os.path.join(datasetpath, "test_images")
    y_test_dir = os.path.join(datasetpath, "test_labels")


    list_IDs_test = read_names(os.path.join(repodatapath, 'test_names.txt'), ext='.png')


    # Checkpoint directory
    checkpoint_loc = "/home/chiawei/temp/wound_tissue_segmentation/checkpoints/MiT+pscse_padded_aug_mit_b3_sup_2025-06-25_00-52-09"

    # create segmentation model with pretrained encoder
    model = model.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        # aux_params=aux_params,
        classes=n_classes,
        activation=ACTIVATION,
        decoder_attention_type='pscse',
    )

    # Optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
    ])


    checkpoint = torch.load(os.path.join(checkpoint_loc, 'best_model.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Test dataloader ==============================================================
    preprocessing_fn = encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

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
    

    iter_test_dataloader = iter(test_dataloader)

    for id in list_IDs_test:

        tensor_image, gt_mask = next(iter_test_dataloader)

        pr_mask = model.predict(tensor_image.to(DEVICE))

        if TO_CATEGORICAL:
            pr_mask = torch.argmax(pr_mask, dim=1)

            pred_mask = pr_mask.squeeze().cpu().numpy()

            #temp
            unique_mask_values = list(np.unique(pred_mask))
            unique_mask_values.remove(0)



            print("Unique values in the mask:", unique_mask_values)

            cv_image = cv2.imread(f"{x_test_dir}/{id}")
            resized_image = cv2.resize(cv_image, (256, 256))

            image = resized_image.copy()

            for i, current_mask_value in enumerate(unique_mask_values):

                boolean_mask = (pred_mask == current_mask_value)    



                image = get_image_with_contour(image, boolean_mask, colors[i])

            outimagepath = f"sample/{id}"
            cv2.imwrite(outimagepath, image)


            print(f"Write to {outimagepath} completed")
        