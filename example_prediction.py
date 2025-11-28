from PIL import Image
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
import numpy as np
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats

opt = load_opt_from_config_files(["/media/Datacenter_storage/Ji/BiomedParse/configs/biomedparse_inference.yaml"])
# opt = load_opt_from_config_files(["/media/Datacenter_storage/Ji/BiomedParse/configs/biomed_seg_lang_v1.yaml"])
opt = init_distributed(opt)

# Load model from pretrained weights
# pretrained_pth = ':microsoft/BiomedParse'
pretrained_pth = '/media/Datacenter_storage/Ji/BiomedParse/output/biomed_seg_lang_v1.yaml_conf~/t2s_only/00006480/default/model_state_dict.pt'
# pretrained_pth = 'pretrained/biomed_parse.pt'
# pretrained_pth = '/media/Datacenter_storage/Ji/BiomedParse/pretrained/biomedparse_v1.pt'

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

uid= "111"
slice= "009"
# Load image and run inference
# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
# image = Image.open('examples/Part_1_516_pathology_breast.png', formats=['png'])
# image = Image.open('/media/Datacenter_storage/Ji/BiomedParse/examples/sub-108_slice_016.png', formats=['png'])
image = Image.open(f"/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/valdo_biomedparse_T2S_cmbOnly/test/sub-{uid}-slice-{slice}_MRI_Brain.png", formats=['png'])
image = image.convert('RGB')
# text prompts querying objects in the image. Multiple ones can be provided.
# prompts = ['neoplastic cells', 'inflammatory cells']
prompts = ['cmb']

# load ground truth mask
gt_masks = []
for prompt in prompts:
    # gt_mask = Image.open(f"examples/Part_1_516_pathology_breast_{prompt.replace(' ', '+')}.png", formats=['png'])
    gt_mask = Image.open(f"/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/valdo_biomedparse_T2S_cmbOnly/test_mask/sub-{uid}-slice-{slice}_MRI_Brain_cmb.png", formats=['png'])
    gt_mask = 1*(np.array(gt_mask.convert('RGB'))[:,:,0] > 0)
    gt_masks.append(gt_mask)

pred_mask = interactive_infer_image(model, image, prompts)

# prediction with ground truth mask
for i, pred in enumerate(pred_mask):
    gt = gt_masks[i]
    dice = (1*(pred>0.5) & gt).sum() * 2.0 / (1*(pred>0.5).sum() + gt.sum())
    print(f'Dice score for {prompts[i]}: {dice:.4f}')
    # p_value = check_mask_stats(np.array(image), pred*255, 'Pathology', prompts[i])
    # print(f'p-value for {prompts[i]}: {p_value:.4f}')