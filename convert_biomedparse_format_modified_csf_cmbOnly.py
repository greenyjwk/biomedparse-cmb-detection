# import os
# import sys
# from PIL import Image
# import numpy as np

# MODALITY= "MRI"
# SITE = "Brain"

# # OUTPUT_PATH = "/media/Datacenter_storage/Ji/valdo_dataset/biomedparse_valdo_all_sequence"
# OUTPUT_PATH = "/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/valdo_biomedparse_csf_cmbonly"

# def handle_mask(root_path, task):
#     mask_root_path = os.path.join(root_path, "masks", task)
    
#     if task == "val":
#         task = "test"
#     if not os.path.exists(os.path.join(OUTPUT_PATH, f"{task}_mask")):
#         os.makedirs(os.path.join(OUTPUT_PATH, f"{task}_mask"))
#     for mask in os.listdir(mask_root_path):
#         mask_temp = mask
#         mask_data = Image.open(os.path.join(mask_root_path, mask))
#         print(np.unique(np.array(mask_data)))
#         if np.any(np.array(mask_data) == 1):
#             # CMB Mask Saving
#             print("np.unique(np.array(mask_data)) == {1}")
#             print("CMB is included")        
#             arr = np.array(mask_data)
#             bin_arr  = (np.isclose(arr, 1)).astype(np.uint8)
#             mask = mask.split(".")[0]
#             mask = mask.replace('_', '-')
#             mask_name = f"{mask}_{MODALITY}_{SITE}_cmb.png"
#             mask_path_biomedparse_cmb = os.path.join(OUTPUT_PATH, f"{task}_mask", mask_name)
#             Image.fromarray(bin_arr, mode='L').save(mask_path_biomedparse_cmb)
#             print("Saved: ", {mask_path_biomedparse_cmb})

#             # CSF Mask Saving
#             arr = np.array(mask_data)
#             bin_arr = (np.isclose(arr, 2)).astype(np.uint8)

#             mask = mask_temp.split(".")[0]
#             mask = mask.replace('_', '-')
#             mask_name = f"{mask}_{MODALITY}_{SITE}_csf.png"
#             mask_path_biomedparse_csf = os.path.join(OUTPUT_PATH, f"{task}_mask", mask_name)
#             Image.fromarray(bin_arr, mode='L').save(mask_path_biomedparse_csf)
#             print("Saved: ", {mask_path_biomedparse_csf})

#         # if np.any(np.array(mask_data) == 2):
#         #     print("np.unique(np.array(mask_data)) == {2}")
#         #     print("CSF is included")
#         #     arr = np.array(mask_data)
#         #     bin_arr = (np.isclose(arr, 2)).astype(np.uint8)

#         #     mask = mask.split(".")[0]
#         #     mask = mask.replace('_', '-')
#         #     mask_name = f"{mask}_{MODALITY}_{SITE}_csf.png"
#         #     mask_path_biomedparse_csf = os.path.join(OUTPUT_PATH, f"{task}_mask", mask_name)
#         #     Image.fromarray(bin_arr, mode='L').save(mask_path_biomedparse_csf)
#         #     print("Saved: ", {mask_path_biomedparse_csf})         
#         # if np.any(np.array(mask_data) == 0):
#         #     print("np.unique(np.array(mask_data)) == {0}")
#         #     print("This doesn't have cmb: ", mask)


# def handle_image(root_path, task):
#     image_root_path = os.path.join(root_path, "images", task)
#     if task == "val":
#         task = "test"
#     if not os.path.exists(os.path.join(OUTPUT_PATH, task)):
#         os.makedirs(os.path.join(OUTPUT_PATH, task))
#     for img in os.listdir(image_root_path):
#         img_data = Image.open(os.path.join(image_root_path, img))
#         img_data = img_data.convert("RGB")
#         img = img.split(".")[0]
#         img = img.replace('_', '-')
#         img_name = f"{img}_{MODALITY}_{SITE}.png"
#         img_path_biomedparse = os.path.join(OUTPUT_PATH, task, img_name)
#         print(img_path_biomedparse)
#         img_data.save(img_path_biomedparse)
#         print(f"Saved: ", {img_path_biomedparse})

# def main(root_path):
#     handle_image(root_path, task="train")
#     handle_mask(root_path, task="train")
#     handle_image(root_path, task="val")
#     handle_mask(root_path, task="val")

# if __name__ == "__main__":
#     # root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final"
#     root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/csf_png_final"
#     main(root_path)

import os
from pathlib import Path
from PIL import Image
import numpy as np

MODALITY = "MRI"
SITE = "Brain"

# OUTPUT_PATH = "/media/Datacenter_storage/Ji/valdo_dataset/biomedparse_valdo_all_sequence"
OUTPUT_PATH = "/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/valdo_t2s_resampling_GAN"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def norm_task(task: str) -> str:
    # "val" → "test"
    return "test" if task == "val" else task

def handle_mask(root_path: str, task: str):
    task_out = norm_task(task)
    mask_root_path = Path(root_path) / "masks" / task
    out_dir = Path(OUTPUT_PATH) / f"{task_out}_mask"
    ensure_dir(out_dir)

    for mask_file in sorted(mask_root_path.iterdir()):
        if not mask_file.is_file():
            continue
        try:
            mask_img = Image.open(mask_file).convert("L")
        except Exception as e:
            print(f"Skip (cannot open): {mask_file} ({e})")
            continue

        arr = np.array(mask_img)

        # --- Only proceed if value 0 exists ---
        if np.any(arr == 1):
            stem = mask_file.stem.replace("_", "-")

            # CMB mask (value == 1)
            cmb_bin = (arr == 1).astype(np.uint8)
            if np.any(cmb_bin):
                cmb_name = f"{stem}_{MODALITY}_{SITE}_brain+microbleeds.png"
                cmb_path = out_dir / cmb_name
                Image.fromarray(cmb_bin, mode="L").save(cmb_path)
                print("Saved:", cmb_path)

            # CSF mask (value == 2)
            csf_bin = (arr == 2).astype(np.uint8)
            if np.any(csf_bin):
                csf_name = f"{stem}_{MODALITY}_{SITE}_cerebrospinal+fluid.png"
                csf_path = out_dir / csf_name
                Image.fromarray(csf_bin, mode="L").save(csf_path)
                print("Saved:", csf_path)
        else:
            print(f"Skipped (no pixel value 0): {mask_file}")

def handle_image(root_path: str, task: str):
    task_out = norm_task(task)
    image_root_path = Path(root_path) / "images" / task
    out_dir = Path(OUTPUT_PATH) / task_out
    ensure_dir(out_dir)

    for img_file in sorted(image_root_path.iterdir()):
        if not img_file.is_file():
            continue
        try:
            img = Image.open(img_file).convert("RGB")
        except Exception as e:
            print(f"Skip (cannot open): {img_file} ({e})")
            continue

        stem = img_file.stem.replace("_", "-")
        out_name = f"{stem}_{MODALITY}_{SITE}.png"
        out_path = out_dir / out_name
        img.save(out_path)
        print("Saved:", out_path)

def main(root_path: str):
    for split in ["train", "val"]:
        handle_image(root_path, task=split)
        handle_mask(root_path, task=split)

if __name__ == "__main__":
    root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/bias_field_correction_resampled_win_normalization_1021_min_max_t2s_yolo_resampling_GAN"
    main(root_path)