import os
import sys
from PIL import Image
import numpy as np

MODALITY= "MRI"
SITE = "Brain"

# OUTPUT_PATH = "/media/Datacenter_storage/Ji/valdo_dataset/biomedparse_valdo_all_sequence"
OUTPUT_PATH = "/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/valdo_biomedparse_csf_gan_norm_1115"

def handle_mask(root_path, task):
    mask_root_path = os.path.join(root_path, "masks", task)
    
    if task == "val":
        task = "test"
    if not os.path.exists(os.path.join(OUTPUT_PATH, f"{task}_mask")):
        os.makedirs(os.path.join(OUTPUT_PATH, f"{task}_mask"))
    for mask in os.listdir(mask_root_path):
        mask_data = Image.open(os.path.join(mask_root_path, mask))

        unique_len = len(np.unique(np.array(mask_data)))
        print(np.unique(np.array(mask_data)))
        if np.any(np.array(mask_data) == 1):
            print("np.unique(np.array(mask_data)) == {1}")
            print("CMB is included")        
            arr = np.array(mask_data)
            bin_arr  = (np.isclose(arr, 1)).astype(np.uint8)
            mask = mask.split(".")[0]
            mask = mask.replace('_', '-')
            mask_name = f"{mask}_{MODALITY}_{SITE}_cmb.png"
            mask_path_biomedparse_cmb = os.path.join(OUTPUT_PATH, f"{task}_mask", mask_name)
            Image.fromarray(bin_arr, mode='L').save(mask_path_biomedparse_cmb)
            print("Saved: ", {mask_path_biomedparse_cmb})
        if np.any(np.array(mask_data) == 2):
            print("np.unique(np.array(mask_data)) == {2}")
            print("CSF is included")
            arr = np.array(mask_data)
            bin_arr = (np.isclose(arr, 2)).astype(np.uint8)

            mask = mask.split(".")[0]
            mask = mask.replace('_', '-')
            mask_name = f"{mask}_{MODALITY}_{SITE}_csf.png"
            mask_path_biomedparse_csf = os.path.join(OUTPUT_PATH, f"{task}_mask", mask_name)
            Image.fromarray(bin_arr, mode='L').save(mask_path_biomedparse_csf)
            print("Saved: ", {mask_path_biomedparse_csf})
         
        if np.any(np.array(mask_data) == 0):
            print("np.unique(np.array(mask_data)) == {0}")
            print("This doesn't have cmb: ", mask)


def handle_image(root_path, task):
    image_root_path = os.path.join(root_path, "images", task)
    if task == "val":
        task = "test"
    if not os.path.exists(os.path.join(OUTPUT_PATH, task)):
        os.makedirs(os.path.join(OUTPUT_PATH, task))
    for img in os.listdir(image_root_path):
        img_data = Image.open(os.path.join(image_root_path, img))
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
    # root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/csf_png"
    root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/bias_field_correction_resampled_win_normalization_1021_min_max_3ch_stacked_yolo_cmbonly_GAN"
    main(root_path)