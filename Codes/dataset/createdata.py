import os
import shutil

def copy_dataset(dfutissue_root, output_root):
    image_src = os.path.join(dfutissue_root, 'Labeled/Original/Images/TrainVal')
    mask_src = os.path.join(dfutissue_root, 'Labeled/Original/Masks/TrainVal')

    for split in ['train', 'val']:
        with open(os.path.join(dfutissue_root, 'train_test_val_list', f'{split}.txt')) as f:
            ids = [line.strip() for line in f]

        for img_id in ids:
            img = f'{img_id}.png'
            shutil.copy(os.path.join(image_src, img), os.path.join(output_root, 'PNGImages', img))
            shutil.copy(os.path.join(mask_src, img), os.path.join(output_root, 'SegmentationClass', img))

    # Repeat for test
    with open(os.path.join(dfutissue_root, 'train_test_val_list', 'test.txt')) as f:
        ids = [line.strip() for line in f]

    for img_id in ids:
        img = f'{img_id}.png'
        shutil.copy(os.path.join(image_src, img), os.path.join(output_root, 'test_images', img))
        shutil.copy(os.path.join(mask_src, img), os.path.join(output_root, 'test_labels', img))

copy_dataset("../../DFUTissue/Labelled", "/home/chiawei/temp/Wound_tissue_segmentation/dataset_MiT_v3+aug-added")#"../../dataset_MiT_v3+aug-added")
