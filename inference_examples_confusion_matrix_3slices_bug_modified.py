import os
import torch
import numpy as np
import huggingface_hub
from PIL import Image
from scipy.ndimage import label
from inference_utils.processing_utils import read_rgb
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

TRUE_POSITVE = 0
FALSE_NEGATIVE = 0
FALSE_POSITIVE = 0

SEGMENT_THRESHOLD = 0.9
# Set GPU device
gpu = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
torch.cuda.set_device(0)

# Load model
# opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = load_opt_from_config_files(["/media/Datacenter_storage/Ji/BiomedParse/configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)
pretrained_pth = '/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/3slices/00012960/default/model_state_dict.pt'
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
        
        combined_mask |= binary_mask
    labeled_mask, num_clusters = label(combined_mask)
    # print("The number of predictions: ", num_clusters)
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

def yolo_gt_seg_pred_overlap_check(gt_path, pred_mask):
    global TRUE_POSITVE, FALSE_NEGATIVE, FALSE_POSITIVE

    gt_box_list = []
    with open(gt_path, 'r') as f:
        for line in f:
            
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            print(parts[1:])
            
            # gt_box_list.append([float(x) for x in parts[1:]])  # YOLO format
            box_strings = parts[1:]
            box_floats = []

            for value in box_strings:
                if value.startswith("np.float"):
                    value = value.split("(")[1].split(")")[0]
                number = float(value)
                box_floats.append(number)
            gt_box_list.append(box_floats)

        # print("The number of ground truth: ", len(gt_box_list))
    width = pred_mask.shape[0]
    num_features = pred_mask.max()
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
        for i in range(num_features):
            if i in matched_preds:
                continue
            pred_segment = (pred_mask == (i + 1)).astype(np.uint8)
            iou = compute_iou(gt_mask, pred_segment)
            if iou > 0.01:  # threshold for match
                TRUE_POSITVE += 1
                matched_preds.add(i)
                matched_gts.add(gt_idx)
                found_match = True
                break

        if not found_match:
            FALSE_NEGATIVE += 1

    FALSE_POSITIVE += (num_features - len(matched_preds))

# === Main Loop ===
# img_root_path = "/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/valdo_biomedparse_T2S_cmbOnly/test"
# gt_root_path = "/media/Datacenter_storage/Ji/valdo_dataset/valdo_t2s_cmbOnly/labels/val"

# Valdo
img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/3slices_png/images/val"
gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_VALDO/3slices_png/labels/val"
text_prompt = ['cmb']

cnt = 0
for img_path in os.listdir(img_root_path):
    img_full_path = os.path.join(img_root_path, img_path)
    # print(f"Processing: {img_full_path}")
    gt_path = img_path.replace("_MRI_Brain.png", "").replace("-slice-", "_slice_")
    gt_path = gt_path.replace("png", "txt")
    full_gt_path = os.path.join(gt_root_path, gt_path)
    pred_mask, num_clusters = inference_rgb(img_full_path, text_prompt)
    yolo_gt_seg_pred_overlap_check(full_gt_path, pred_mask)

    cnt += 1
    if cnt % 50  == 0:
        print("TRUE_POSTIVE", TRUE_POSITVE)
        print("FALSE_NEGATIVE", FALSE_NEGATIVE)
        print("FALSE_POSITIVE", FALSE_POSITIVE)
        print("Image Processed:", cnt)
        print()
        print()

print("TRUE_POSITVE:", TRUE_POSITVE)
print("FALSE_NEGATIVE:", FALSE_NEGATIVE)
print("FALSE_POSITIVE:", FALSE_POSITIVE)