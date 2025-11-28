# import sys
# import os
# import torch
# import numpy as np
# import pandas as pd
# import huggingface_hub
# from PIL import Image
# from scipy import ndimage
# from scipy.ndimage import label
# from datetime import datetime
# from inference_utils.processing_utils import read_rgb
# from modeling.BaseModel import BaseModel
# from modeling import build_model
# from utilities.distributed import init_distributed
# from utilities.arguments import load_opt_from_config_files
# from utilities.constants import BIOMED_CLASSES
# from inference_utils.inference import interactive_infer_image
# from pathlib import Path


# TRUE_POSITIVE = 0
# FALSE_NEGATIVE = 0
# FALSE_POSITIVE = 0
# SEGMENT_THRESHOLD = 0.5
# SMALL_FILTER = 1

# # Set GPU device
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# torch.cuda.set_device(0)

# # Load model
# # opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
# opt = load_opt_from_config_files(["/media/Datacenter_storage/Ji/BiomedParse/configs/biomedparse_inference.yaml"])
# opt = init_distributed(opt)

# # All Sequence
# # pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_32/00004536/default/model_state_dict.pt"

# # T2S Only
# # pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_39/00025920/default/model_state_dict.pt"

# # T2S Only from original pretrained biomedparse
# # pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_41/00010812/default/model_state_dict.pt"

# # T2S Only from original dataset(No GAN, resampling train/val set)
# # pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_41/00012720/default/model_state_dict.pt"

# # T2S Only from original dataset with GAN augmentation with resampling
# pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_43/00024080/default/model_state_dict.pt"

# # T2 Only
# # pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_37/00012960/default/model_state_dict.pt"

# # T1 Only
# # pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_38/00025920/default/model_state_dict.pt"

# model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
# with torch.no_grad():
#     model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

# # Login to HF (if needed)
# HF_TOKEN = ''
# huggingface_hub.login(HF_TOKEN)

# def plot_segmentation_masks(segmentation_masks):
#     combined_mask = np.zeros_like(segmentation_masks[0], dtype=np.uint8)
#     for mask in segmentation_masks:
#         binary_mask = (mask >= SEGMENT_THRESHOLD).astype(np.uint8)
        
#         combined_mask |= binary_mask    # |= is bitwise or, so it concatenates.
#     labeled_mask, num_clusters = label(combined_mask)
#     # print("The number of predictions: ", num_clusters)

#     '''
#     mask index 1 cluster has size of 2 
#     mask index 2 cluster has size of 4
#     mask index 3 cluster has size of 6
#     mask index 4 cluster has size of 1 
#     (ex)[2, 4, 6, 1]
#     '''
    
#     size_list = ndimage.sum(combined_mask, labeled_mask, index=range(1, num_clusters+1))
#     print("size_list              : ", size_list)
#     print("labeled_mask.shape     : ", labeled_mask.shape)
#     print("np.unique(labeled_mask): ", np.unique(labeled_mask))
#     small_size_index = [idx+1 for idx, value in enumerate(size_list) if value < SMALL_FILTER]
#     print("small_size_index", small_size_index)

#     for idx in small_size_index:
#         labeled_mask[labeled_mask==idx] = 0 # for small size label, being overlapped with 0

#     return labeled_mask, num_clusters

# def inference_rgb(file_path, text_prompts):
#     image = read_rgb(file_path)
#     pred_masks = interactive_infer_image(model, Image.fromarray(image), text_prompts)
#     pred_mask, num_clusters = plot_segmentation_masks(pred_masks)
#     return pred_mask, num_clusters

# def compute_iou(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     return intersection / union if union > 0 else 0

# def yolo_gt_seg_pred_overlap_check(
#     gt_path,
#     pred_mask,
#     matched_excel_filename,
#     unmatched_excel_filename,
#     false_positive_excel_filename,
#     img_path,
# ):
#     global TRUE_POSITIVE, FALSE_NEGATIVE, FALSE_POSITIVE

#     gt_box_list = []
#     gt_path_list = []
#     if os.path.exists(gt_path):
#         with open(gt_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) < 5:
#                     continue
#                 gt_box_list.append([float(x) for x in parts[1:]])  # YOLO format
#                 gt_path_list.append(gt_path)
#     width = pred_mask.shape[0]

#     '''
#     previous version uses max features to find the toal number of preds.
#     but we use unique preds, becuase we mask out the small preds.
#     '''
#     # num_features = pred_mask.max()
#     unique_features = np.unique(pred_mask)
#     print("num_features", unique_features)
#     print()
#     print()
#     matched_preds = set()
#     matched_gts = set()

#     for gt_idx, gt_box in enumerate(gt_box_list):
#         x_center, y_center, box_w, box_h = [gt_box[i] * width for i in range(4)]
#         x1 = int(x_center - box_w / 2)
#         x2 = int(x_center + box_w / 2)
#         y1 = int(y_center - box_h / 2)
#         y2 = int(y_center + box_h / 2)
#         x1, x2 = np.clip([x1, x2], 0, width)
#         y1, y2 = np.clip([y1, y2], 0, width)

#         gt_mask = np.zeros_like(pred_mask, dtype=np.uint8)
#         gt_mask[y1:y2, x1:x2] = 1

#         found_match = False
#         for i in unique_features:
#             if i == 0 or i in matched_preds:
#                 continue
#             # pred_segment = (pred_mask == (i + 1)).astype(np.uint8)
#             pred_segment = (pred_mask == i).astype(np.uint8)    # i is unique pred_mask_index
#             iou = compute_iou(gt_mask, pred_segment)
#             if iou > 0.01:
#                 TRUE_POSITIVE += 1
#                 matched_preds.add(i)
#                 matched_gts.add(gt_idx)
#                 found_match = True
#                 break
#         if not found_match:
#             FALSE_NEGATIVE += 1

#     # Generate timestamp for filename
    
#     unmatched_gts = []
#     for i, gt_path in enumerate(gt_path_list):
#         if i not in matched_gts:
#             unmatched_gts.append(gt_path)
#             append_to_excel(unmatched_excel_filename, gt_path, gt_box_list[i])
#         elif i in matched_gts:
#             append_to_excel(matched_excel_filename, gt_path, gt_box_list[i])

#     print()
#     print("unmatched_gts  ", unmatched_gts)
#     print()
#     false_positive_indices = [i for i in unique_features if i not in matched_preds and i != 0]
#     for pred_idx in false_positive_indices:
#         pred_area = int(np.sum(pred_mask == pred_idx))
#         append_false_positive_to_excel(
#             false_positive_excel_filename,
#             img_path,
#             pred_idx,
#             pred_area,
#         )

#     FALSE_POSITIVE += len(false_positive_indices)

# def cal_box_area(gt_path, gt_coord):

#     # mask_path_root = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_T2SONLY/masks/val"
#     # mask_path = os.path.join(mask_path_root, os.path.basename(gt_path).replace( '.txt', '.png'))
#     # print(mask_path)
#     mask_path = gt_path.replace('labels', 'masks').replace( '.txt', '.png')
#     mask = np.array(Image.open(mask_path).convert('L'))

#     height, width = mask.shape
#     x_center, y_center, box_w, box_h = gt_coord

#     x_center *= width
#     y_center *= height
#     box_w *= width
#     box_h *= height

#     x1 = int(x_center - box_w / 2)
#     x2 = int(x_center + box_w / 2)
#     y1 = int(y_center - box_h / 2)
#     y2 = int(y_center + box_h / 2)

#     # x1, x2 = np.clip([x1, x2], 0, width)
#     # y1, y2 = np.clip([y1, y2], 0, height)

#     buffer = 2
#     x1, x2, y1, y2 = x1-buffer, x2+buffer, y1-buffer, y2+buffer
#     region = mask[y1:y2, x1:x2]
#     foreground_area = np.sum(region==255)
#     print("Foreground area: ", foreground_area)
#     return foreground_area


# def append_to_excel(file_path, gt_path, gt_coord):
#     foreground_area = cal_box_area(gt_path, gt_coord)
#     if os.path.exists(file_path):
#         df = pd.read_excel(file_path)
#     else:
#         df = pd.DataFrame(columns=["gt_path", "cmb_area"])
#     df = pd.concat([df, pd.DataFrame({"gt_path": [gt_path], "cmb_area": [foreground_area]})], ignore_index=True)
#     df.to_excel(file_path, index=False)


# def append_false_positive_to_excel(file_path, img_path, pred_idx, pred_area):
#     if os.path.exists(file_path):
#         df = pd.read_excel(file_path)
#     else:
#         df = pd.DataFrame(columns=["img_path", "prediction_label", "prediction_area_pixels"])
#     df = pd.concat(
#         [
#             df,
#             pd.DataFrame(
#                 {
#                     "img_path": [img_path],
#                     "prediction_label": [pred_idx],
#                     "prediction_area_pixels": [pred_area],
#                 }
#             ),
#         ],
#         ignore_index=True,
#     )
#     df.to_excel(file_path, index=False)


# # === Main Loop ===
# # # All Sequence - Mayo
# # img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_stacked_win_norm_yolo_dataset/images/test"
# # gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_stacked_win_norm_yolo_dataset/labels/test"
# # text_prompt = ["cerebral microbleeds"]

# # All Sequence - Valdo
# # img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_GAN_CMBONLY_ALLSEQ_WIN_NORM/images/val"
# # gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_GAN_CMBONLY_ALLSEQ_WIN_NORM/labels/val"
# # text_prompt = ["cerebral microbleeds"]

# # T2S Only - Valdo
# # img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_GAN_CMBONLY_T2S_WIN_NORM/images/val"
# # gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_GAN_CMBONLY_T2S_WIN_NORM/labels/val"
# # text_prompt = ["cerebral microbleeds"]


# # # T2S Only - Valdo - seuqneital experiment
# img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/bias_field_correction_resampled_win_normalization_1021_min_max_t2s_yolo_resampling_GAN/images/val"
# gt_root_path  = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/bias_field_correction_resampled_win_normalization_1021_min_max_t2s_yolo_resampling_GAN/labels/val"
# text_prompt = ["brain microbleeds"]

# # # T2S Only - Valdo - seuqneital experiment
# # img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/sequential_T2/images/val"
# # gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/sequential_T2/labels/val"
# # text_prompt = ["cerebral microbleeds"]


# # T1 Only - Valdo - seuqneital experiment
# # img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/sequential_T1/images/val"
# # gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/sequential_T1/labels/val"
# # text_prompt = ["brain microbleeds"]

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dir = f"/media/Datacenter_storage/Ji/BiomedParse/runs/matched_unmatched/{timestamp}"
# matched_excel_filename = f"{output_dir}/matched_gts.xlsx"
# unmatched_excel_filename = f"{output_dir}/unmatched_gts.xlsx"
# false_positive_excel_filename = f"{output_dir}/false_positive_preds.xlsx"
# Path(output_dir).mkdir(parents=True, exist_ok=True)

# cnt = 0
# for img_path in os.listdir(img_root_path):
#     img_full_path = os.path.join(img_root_path, img_path)
#     gt_path = img_path.replace("_MRI_Brain.png", "").replace("-slice-", "_slice_").replace('png', 'txt')
#     full_gt_path = os.path.join(gt_root_path, gt_path)
#     pred_mask, num_clusters = inference_rgb(img_full_path, text_prompt)
#     yolo_gt_seg_pred_overlap_check(
#         full_gt_path,
#         pred_mask,
#         matched_excel_filename,
#         unmatched_excel_filename,
#         false_positive_excel_filename,
#         img_full_path,
#     )

#     cnt += 1
#     # if cnt % 50  == 0:
#     if cnt % 10  == 0:
#         print("TRUE_POSTIVE", TRUE_POSITIVE)
#         print("FALSE_NEGATIVE", FALSE_NEGATIVE)
#         print("FALSE_POSITIVE", FALSE_POSITIVE)
#         print("Image Processed:", cnt)
#         print()

# print("TRUE_POSITIVE:", TRUE_POSITIVE)
# print("FALSE_NEGATIVE:", FALSE_NEGATIVE)
# print("FALSE_POSITIVE:", FALSE_POSITIVE)


import sys
import os
import torch
import numpy as np
import pandas as pd
import huggingface_hub
from PIL import Image
from scipy import ndimage
from scipy.ndimage import label
from datetime import datetime
from inference_utils.processing_utils import read_rgb
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from pathlib import Path


TRUE_POSITIVE = 0
FALSE_NEGATIVE = 0
FALSE_POSITIVE = 0
SEGMENT_THRESHOLD = 0.5
SMALL_FILTER = 1

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
torch.cuda.set_device(0)

# Load model
# opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = load_opt_from_config_files(["/media/Datacenter_storage/Ji/BiomedParse/configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

# All Sequence
# pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_32/00004536/default/model_state_dict.pt"

# T2S Only
# pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_39/00025920/default/model_state_dict.pt"

# T2S Only from original pretrained biomedparse
# pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_41/00010812/default/model_state_dict.pt"

# T2S Only from original dataset(No GAN, resampling train/val set)
pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_41/00012720/default/model_state_dict.pt"

# T2S Only from original dataset with GAN augmentation with resampling
# pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_43/00024080/default/model_state_dict.pt"

# T2 Only
# pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_37/00012960/default/model_state_dict.pt"

# T1 Only
# pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_38/00025920/default/model_state_dict.pt"

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

# Login to HF (if needed)
HF_TOKEN = ''
huggingface_hub.login(HF_TOKEN)

def plot_segmentation_masks(segmentation_masks):
    combined_mask = np.zeros_like(segmentation_masks[0], dtype=np.uint8)
    for mask in segmentation_masks:
        binary_mask = (mask >= SEGMENT_THRESHOLD).astype(np.uint8)
        
        combined_mask |= binary_mask    # |= is bitwise or, so it concatenates.
    labeled_mask, num_clusters = label(combined_mask)
    # print("The number of predictions: ", num_clusters)

    '''
    mask index 1 cluster has size of 2 
    mask index 2 cluster has size of 4
    mask index 3 cluster has size of 6
    mask index 4 cluster has size of 1 
    (ex)[2, 4, 6, 1]
    '''
    
    size_list = ndimage.sum(combined_mask, labeled_mask, index=range(1, num_clusters+1))
    print("size_list              : ", size_list)
    print("labeled_mask.shape     : ", labeled_mask.shape)
    print("np.unique(labeled_mask): ", np.unique(labeled_mask))
    small_size_index = [idx+1 for idx, value in enumerate(size_list) if value < SMALL_FILTER]
    print("small_size_index", small_size_index)

    for idx in small_size_index:
        labeled_mask[labeled_mask==idx] = 0 # for small size label, being overlapped with 0

    return labeled_mask, num_clusters

def inference_rgb(file_path, text_prompts):
    image = read_rgb(file_path)
    pred_masks = interactive_infer_image(model, Image.fromarray(image), text_prompts)
    pred_mask, num_clusters = plot_segmentation_masks(pred_masks)
    return pred_mask, num_clusters

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def yolo_gt_seg_pred_overlap_check(
    gt_path,
    pred_mask,
    matched_excel_filename,
    unmatched_excel_filename,
    false_positive_excel_filename,
    img_path,
):
    global TRUE_POSITIVE, FALSE_NEGATIVE, FALSE_POSITIVE

    gt_box_list = []
    gt_path_list = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                gt_box_list.append([float(x) for x in parts[1:]])  # YOLO format
                gt_path_list.append(gt_path)
    width = pred_mask.shape[0]

    '''
    previous version uses max features to find the toal number of preds.
    but we use unique preds, becuase we mask out the small preds.
    '''
    # num_features = pred_mask.max()
    unique_features = np.unique(pred_mask)
    print("num_features", unique_features)
    print()
    print()
    matched_preds = set()
    matched_gts = set()

    for gt_idx, gt_box in enumerate(gt_box_list):
        x_center, y_center, box_w, box_h = [gt_box[i] * width for i in range(4)]
        x1 = int(x_center - box_w / 2)
        x2 = int(x_center + box_w / 2)
        y1 = int(y_center - box_h / 2)
        y2 = int(y_center + box_h / 2)
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, width)

        gt_mask = np.zeros_like(pred_mask, dtype=np.uint8)
        gt_mask[y1:y2, x1:x2] = 1

        found_match = False
        for i in unique_features:
            if i == 0 or i in matched_preds:
                continue
            # pred_segment = (pred_mask == (i + 1)).astype(np.uint8)
            pred_segment = (pred_mask == i).astype(np.uint8)    # i is unique pred_mask_index
            iou = compute_iou(gt_mask, pred_segment)
            if iou > 0.01:
                TRUE_POSITIVE += 1
                matched_preds.add(i)
                matched_gts.add(gt_idx)
                found_match = True
                break
        if not found_match:
            FALSE_NEGATIVE += 1

    # Generate timestamp for filename
    
    unmatched_gts = []
    for i, gt_path in enumerate(gt_path_list):
        if i not in matched_gts:
            unmatched_gts.append(gt_path)
            append_to_excel(unmatched_excel_filename, gt_path, gt_box_list[i])
        elif i in matched_gts:
            append_to_excel(matched_excel_filename, gt_path, gt_box_list[i])

    print()
    print("unmatched_gts  ", unmatched_gts)
    print()
    false_positive_indices = [i for i in unique_features if i not in matched_preds and i != 0]
    for pred_idx in false_positive_indices:
        pred_area = int(np.sum(pred_mask == pred_idx))
        append_false_positive_to_excel(
            false_positive_excel_filename,
            img_path,
            pred_idx,
            pred_area,
        )

    FALSE_POSITIVE += len(false_positive_indices)

def cal_box_area(gt_path, gt_coord):

    # mask_path_root = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_T2SONLY/masks/val"
    # mask_path = os.path.join(mask_path_root, os.path.basename(gt_path).replace( '.txt', '.png'))
    # print(mask_path)
    mask_path = gt_path.replace('labels', 'masks').replace( '.txt', '.png')
    mask = np.array(Image.open(mask_path).convert('L'))

    height, width = mask.shape
    x_center, y_center, box_w, box_h = gt_coord

    x_center *= width
    y_center *= height
    box_w *= width
    box_h *= height

    x1 = int(x_center - box_w / 2)
    x2 = int(x_center + box_w / 2)
    y1 = int(y_center - box_h / 2)
    y2 = int(y_center + box_h / 2)

    # x1, x2 = np.clip([x1, x2], 0, width)
    # y1, y2 = np.clip([y1, y2], 0, height)

    buffer = 2
    x1, x2, y1, y2 = x1-buffer, x2+buffer, y1-buffer, y2+buffer
    region = mask[y1:y2, x1:x2]
    foreground_area = np.sum(region==255)
    print("Foreground area: ", foreground_area)
    return foreground_area


def append_to_excel(file_path, gt_path, gt_coord):
    foreground_area = cal_box_area(gt_path, gt_coord)
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=["gt_path", "cmb_area"])
    df = pd.concat([df, pd.DataFrame({"gt_path": [gt_path], "cmb_area": [foreground_area]})], ignore_index=True)
    df.to_excel(file_path, index=False)


def append_false_positive_to_excel(file_path, img_path, pred_idx, pred_area):
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=["img_path", "prediction_label", "prediction_area_pixels"])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "img_path": [img_path],
                    "prediction_label": [pred_idx],
                    "prediction_area_pixels": [pred_area],
                }
            ),
        ],
        ignore_index=True,
    )
    df.to_excel(file_path, index=False)


# === Main Loop ===
# # All Sequence - Mayo
# img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_stacked_win_norm_yolo_dataset/images/test"
# gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_stacked_win_norm_yolo_dataset/labels/test"
# text_prompt = ["cerebral microbleeds"]

# All Sequence - Valdo
# img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_GAN_CMBONLY_ALLSEQ_WIN_NORM/images/val"
# gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_GAN_CMBONLY_ALLSEQ_WIN_NORM/labels/val"
# text_prompt = ["cerebral microbleeds"]

# T2S Only - Valdo
# img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_GAN_CMBONLY_T2S_WIN_NORM/images/val"
# gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/valdo_png_final_GAN_CMBONLY_T2S_WIN_NORM/labels/val"
# text_prompt = ["cerebral microbleeds"]


# # T2S Only - Valdo - seuqneital experiment
img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/bias_field_correction_resampled_win_normalization_1021_min_max_t2s_yolo_resampling_GAN/images/val"
gt_root_path  = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/bias_field_correction_resampled_win_normalization_1021_min_max_t2s_yolo_resampling_GAN/labels/val"
text_prompt = ["brain microbleeds"]

# # T2S Only - Valdo - seuqneital experiment
# img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/sequential_T2/images/val"
# gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/sequential_T2/labels/val"
# text_prompt = ["cerebral microbleeds"]


# T1 Only - Valdo - seuqneital experiment
# img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/sequential_T1/images/val"
# gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/sequential_T1/labels/val"
# text_prompt = ["brain microbleeds"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/media/Datacenter_storage/Ji/BiomedParse/runs/matched_unmatched/{timestamp}"
matched_excel_filename = f"{output_dir}/matched_gts.xlsx"
unmatched_excel_filename = f"{output_dir}/unmatched_gts.xlsx"
false_positive_excel_filename = f"{output_dir}/false_positive_preds.xlsx"
Path(output_dir).mkdir(parents=True, exist_ok=True)

cnt = 0
for img_path in os.listdir(img_root_path):
    img_full_path = os.path.join(img_root_path, img_path)
    gt_path = img_path.replace("_MRI_Brain.png", "").replace("-slice-", "_slice_").replace('png', 'txt')
    full_gt_path = os.path.join(gt_root_path, gt_path)
    pred_mask, num_clusters = inference_rgb(img_full_path, text_prompt)
    yolo_gt_seg_pred_overlap_check(
        full_gt_path,
        pred_mask,
        matched_excel_filename,
        unmatched_excel_filename,
        false_positive_excel_filename,
        img_full_path,
    )

    cnt += 1
    # if cnt % 50  == 0:
    if cnt % 10  == 0:
        print("TRUE_POSTIVE", TRUE_POSITIVE)
        print("FALSE_NEGATIVE", FALSE_NEGATIVE)
        print("FALSE_POSITIVE", FALSE_POSITIVE)
        print("Image Processed:", cnt)
        print()

print("TRUE_POSITIVE:", TRUE_POSITIVE)
print("FALSE_NEGATIVE:", FALSE_NEGATIVE)
print("FALSE_POSITIVE:", FALSE_POSITIVE)
