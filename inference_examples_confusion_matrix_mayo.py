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

TRUE_POSITIVE = 0
FALSE_NEGATIVE = 0
FALSE_POSITIVE = 0
SEGMENTATION_THRESHOLD = 0.9

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.cuda.set_device(0)

# Load model
opt = load_opt_from_config_files(["/media/Datacenter_storage/Ji/BiomedParse/configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)
pretrained_pth = "/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/run_33/00003240/default/model_state_dict.pt"
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

# Login to HF (if needed)
HF_TOKEN = ''
huggingface_hub.login(HF_TOKEN)

def plot_segmentation_masks(segmentation_masks):
    combined_mask = np.zeros_like(segmentation_masks[0], dtype=np.uint8)
    for mask in segmentation_masks:
        binary_mask = (mask >= SEGMENTATION_THRESHOLD).astype(np.uint8)
        
        combined_mask |= binary_mask
    labeled_mask, num_clusters = label(combined_mask)
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
    global TRUE_POSITIVE, FALSE_NEGATIVE, FALSE_POSITIVE

    gt_box_list = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                gt_box_list.append([float(x) for x in parts[1:]])  # YOLO format

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
                TRUE_POSITIVE += 1
                matched_preds.add(i)
                matched_gts.add(gt_idx)
                found_match = True
                break

        if not found_match:
            FALSE_NEGATIVE += 1

    FALSE_POSITIVE += (num_features - len(matched_preds))

# MAYO
# All sequence
# img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_allsequence_png_GAN/images/test"
# gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_allsequence_png_GAN/labels/test"

# All Sequence
img_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_t2s_win_norm_yolo_dataset/images/test"
gt_root_path = "/media/Datacenter_storage/PublicDatasets/cerebral_microbleeds_MAYO/mayo_t2s_win_norm_yolo_dataset/labels/test"
text_prompt = ["cerebral microbleeds"]

cnt = 0
for img_path in os.listdir(img_root_path):
    img_full_path = os.path.join(img_root_path, img_path)
    
    gt_path = img_path.replace("png", "txt")
    full_gt_path = os.path.join(gt_root_path, gt_path)
    pred_mask, num_clusters = inference_rgb(img_full_path, text_prompt)
    yolo_gt_seg_pred_overlap_check(full_gt_path, pred_mask)
    
    cnt += 1
    if cnt % 50  == 0:
        print("TRUE_POSITIVE", TRUE_POSITIVE)
        print("FALSE_NEGATIVE", FALSE_NEGATIVE)
        print("FALSE_POSITIVE", FALSE_POSITIVE)
        print("Image Processed:", cnt)
        print()

print("TRUE_POSITIVE:", TRUE_POSITIVE)
print("FALSE_NEGATIVE:", FALSE_NEGATIVE)
print("FALSE_POSITIVE:", FALSE_POSITIVE)