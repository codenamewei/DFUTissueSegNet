import os
import shutil
import logging

def copy_dataset(dfutissue_root, output_root):
    image_src = os.path.join(dfutissue_root, 'Labeled/Original/Images/TrainVal')
    mask_src = os.path.join(dfutissue_root, 'Labeled/Original/Annotations/TrainVal')

    for split in ['labeled_train_names.txt', 'labeled_val_names.txt']:

        labeledtxtpath = os.path.join(dfutissue_root, "Labeled", split)

        if not os.path.exists(labeledtxtpath):

            raise FileNotFoundError(f"File Not Found: {labeledtxtpath}")
        
        with open(labeledtxtpath) as f:
            ids = [line.strip() for line in f]

        for img_id in ids:
            img = f'{img_id}.png'
            image_path = os.path.join(image_src, img)
            label_path = os.path.join(mask_src, img)

            if not os.path.exists(image_path):

                raise FileNotFoundError(f"Image not found: {image_path}")
            
            elif not os.path.exists(label_path):

                raise FileNotFoundError(f"Label not found: {label_path}")
            
            
            shutil.copy(image_path, os.path.join(output_root, 'PNGImages', img))
            shutil.copy(label_path, os.path.join(output_root, 'SegmentationClass', img))

    image_src = os.path.join(dfutissue_root, 'Labeled/Original/Images/Test')
    mask_src = os.path.join(dfutissue_root, 'Labeled/Original/Annotations/Test')


    # Repeat for test
    with open(os.path.join(dfutissue_root,  "Labeled", 'test_names.txt')) as f:
        ids = [line.strip() for line in f]

    for img_id in ids:
        img = f'{img_id}.png'

        image_path = os.path.join(image_src, img)
        label_path = os.path.join(mask_src, img)
        
        if not os.path.exists(image_path):

            raise FileNotFoundError(f"Image not found: {image_path}")
        
        elif not os.path.exists(label_path):

            raise FileNotFoundError(f"Label not found: {label_path}")
        
        shutil.copy(image_path, os.path.join(output_root, 'test_images', img))
        shutil.copy(label_path, os.path.join(output_root, 'test_labels', img))

#copy_dataset("../../DFUTissue", "../../dataset_MiT_v3+aug-added")
#copy_dataset("/home/chiawei/temp/DFUTissueSegNet/DFUTissue", "/home/chiawei/temp/Wound_tissue_segmentation/dataset_MiT_v3+aug-added")

repo_path = "/home/chiawei/Documents/work/dfu/DFUTissueSegNet"
data_path = "/home/chiawei/Documents/work/dfu/DFUTissueSegNet_metadata"
copy_dataset(os.path.join(repo_path, "DFUTissue"), os.path.join(data_path, "dataset_MiT_v3+aug-added"))
logging.info("Completed copy data from {repo_path} to {data_path}")