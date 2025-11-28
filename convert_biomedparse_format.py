import os
import sys
from PIL import Image
import numpy as np

MODALITY= "MRI"
SITE = "Brain"

# OUTPUT_PATH = "/media/Datacenter_storage/Ji/valdo_dataset/biomedparse_valdo_all_sequence"
# OUTPUT_PATH = "/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/valdo_biomedparse_3slices"
OUTPUT_PATH = "/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/valdo_biomedparse_cmbonly_t2sonly_gan"

def handle_mask(root_path, task):
    train_mask_root_path = os.path.join(root_path, "masks", task)
    
    if task == "val":
        task = "test"
    if not os.path.exists(os.path.join(OUTPUT_PATH, f"{task}_mask")):
        os.makedirs(os.path.join(OUTPUT_PATH, f"{task}_mask"))
    for mask in os.listdir(train_mask_root_path):
        mask_data = Image.open(os.path.join(train_mask_root_path, mask))

        unique_len = len(np.unique(np.array(mask_data)))
        if unique_len == 1 and task == "train":
            print(mask)
            continue
        mask_data = mask_data.convert('L')
        mask = mask.split(".")[0]
        mask = mask.replace('_', '-')

        mask_name = f"{mask}_{MODALITY}_{SITE}.png"
        mask_path_biomedparse = os.path.join(OUTPUT_PATH, f"{task}_mask", mask_name)
        mask_data.save(mask_path_biomedparse)
        print("Saved: ", {mask_path_biomedparse})


def handle_image(root_path, task):
    train_image_root_path = os.path.join(root_path, "images", task)
    if task == "val":
        task = "test"
    if not os.path.exists(os.path.join(OUTPUT_PATH, task)):
        os.makedirs(os.path.join(OUTPUT_PATH, task))
    for img in os.listdir(train_image_root_path):
        img_data = Image.open(os.path.join(train_image_root_path, img))
        img_data = img_data.convert("RGB")
        img = img.split(".")[0]
        img = img.replace('_', '-')
        img_name = f"{img}_{MODALITY}_{SITE}.png"
        img_path_biomedparse = os.path.join(OUTPUT_PATH, task, img_name)
        print(img_path_biomedparse)
        img_data.save(img_path_biomedparse)
        print(f"Saved: ", {img_path_biomedparse})

def main(root_path):
    handle_image(root_path, task="train")
    handle_mask(root_path, task="train")
    handle_image(root_path, task="val")
    handle_mask(root_path, task="val")

if __name__ == "__main__":
    # root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final"
    root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_GAN_CMBONLY_T2SONLY"
    main(root_path)