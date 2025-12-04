import os
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as nd
from torch import nn
from PIL import Image
from scipy.ndimage import distance_transform_edt


class Loss_Distance(nn.Module):
    def __init__(self, batched_inputs, sigma_param):
        super(Loss_Distance, self).__init__()
        # self.outputs = outputs['batched_masks']
        self.batched_inputs = batched_inputs
        self.root_path = "/media/Datacenter_storage/Ji/BiomedParse"
        self.csf_mask_path = os.path.join(self.root_path, batched_inputs[0]["grounding_info"][0]["mask_file"])
        self.cmb_mask_path = self.csf_mask_path.replace("cerebrospinal+fluid", "brain_microbleeds")
        self.sigma = sigma_param

    # def forward(self, cmb_prob: torch.Tensor) -> torch.Tensor:
    #     csf_mask_path = "/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/loss_dev/train_mask/sub-207-slice-108_MRI_Brain_cerebrospinal+fluid.png"
    #     cmb_mask_path = csf_mask_path.replace("cerebrospinal+fluid", "brain+microbleeds")
    #     gt_csf = Image.open(csf_mask_path).convert('L')
    #     gt_cmb = Image.open(cmb_mask_path).convert('L')
    #     gt_csf = np.array(gt_csf)
    #     gt_cmb = np.array(gt_cmb)

    #     gt_csf = torch.from_numpy(gt_csf)
    #     gt_cmb = torch.from_numpy(gt_cmb)

    #     gt_csf = gt_csf.to(cmb_prob.device)
    #     gt_cmb = gt_cmb.to(cmb_prob.device)

    #     gt_csf = gt_csf.unsqueeze(0).unsqueeze(0).float()
    #     gt_cmb = gt_cmb.unsqueeze(0).unsqueeze(0).float()

        
    #     dist_map = compute_csf_distance_map(gt_csf).to(cmb_prob.device) # Distance Map for CSF
    #     # print("=== Distance Maps ===")
    #     # print(dist_map)
    #     # print()

    #     sigma = 5.0
    #     weight = torch.exp(-dist_map / sigma)  # Distance map based weight
    #     # print("=== Weight Maps ===")
    #     # print(weight)
    #     # print()

        
    #     bg_mask = (gt_cmb==0).float().to(cmb_prob.device)
    #     # print("bg_mask")
    #     # print(bg_mask)
    #     # print()

    #     cmb_prob = torch.sigmoid(cmb_prob)
    #     cmb_prob = cmb_prob * (cmb_prob > 0.6).float()
    #     cmb_prob = F.interpolate(cmb_prob, size=bg_mask.shape[2:], mode='bilinear')
    #     fp_soft = cmb_prob * bg_mask # selecting only fp cases.

    #     loss_map = fp_soft * weight
    #     # print("=== Loss Map ===")
    #     # print(loss_map)
    #     # print()

    #     max_values = clustering(loss_map)
    #     # print("=== Max values for each cluster ===")
    #     # print(max_values)
    #     # print()
    #     # print("=== Sum of max values for all clusters: FINAL DISTANCE LOSS ===")
    #     # print(sum(max_values))

    #     reduction = "sum"
    #     if reduction == "mean":
    #         denom = (bg_mask * (weight > 0).float()).sum()
    #         denom = torch.clamp(denom, min=1.0)
    #     elif reduction == "sum":
    #         return sum(max_values)
    #     else:
    #         return loss_map

    def forward(self, cmb_prob: torch.Tensor) -> torch.Tensor:
        csf_mask_path = "/media/Datacenter_storage/Ji/BiomedParse/biomedparse_datasets/loss_dev/train_mask/sub-207-slice-108_MRI_Brain_cerebrospinal+fluid.png"
        cmb_mask_path = csf_mask_path.replace("cerebrospinal+fluid", "brain+microbleeds")
        gt_csf = Image.open(csf_mask_path).convert('L')
        gt_cmb = Image.open(cmb_mask_path).convert('L')
        gt_csf = torch.from_numpy(np.array(gt_csf)).to(cmb_prob.device).unsqueeze(0).unsqueeze(0).float()
        gt_cmb = torch.from_numpy(np.array(gt_cmb)).to(cmb_prob.device).unsqueeze(0).unsqueeze(0).float()

        # distance map from CSF (fixed wrt model => fine)
        dist_map = compute_csf_distance_map(gt_csf).to(cmb_prob.device)

        weight = torch.exp(-dist_map / self.sigma)   # Bx1xHxW

        # background mask outside CMB
        bg_mask = (gt_cmb == 0).float().to(cmb_prob.device)

        # model prediction
        cmb_prob = torch.sigmoid(cmb_prob)
        # (optional) REMOVE hard threshold for better gradients:
        # cmb_prob = cmb_prob * (cmb_prob > 0.6).float()

        # resize prediction to match mask
        cmb_prob = F.interpolate(cmb_prob, size=bg_mask.shape[2:], mode='bilinear', align_corners=False)

        # false-positive "soft" map
        fp_soft = cmb_prob * bg_mask  # only region where GT is not CMB

        # distance-weighted loss
        loss_map = fp_soft * weight

        # final scalar loss (differentiable)
        # loss_distance = loss_map.mean()
        loss_distance = loss_map.sum() / torch.clamp((weight > 0).sum(), min=1.0)

        # loss_distance = loss_map.sum()
        return loss_distance


def clustering(loss_map):
    numpy_arr = loss_map.squeeze().detach().cpu().numpy()
    labeled, num = nd.label(numpy_arr)
    max_values = []
    for i in range(1, num + 1):
        region = numpy_arr[labeled == i]
        max_values.append(region.max())
    return max_values


def compute_csf_distance_map(csf_mask: torch.Tensor) -> torch.Tensor:
    device = csf_mask.device
    csf_np = csf_mask.detach().cpu().numpy().astype(np.float32)

    B = csf_np.shape[0]
    dist_maps = []

    for b in range(B):
        csf_slice = csf_np[b,0]
        dt = distance_transform_edt(1.0 - csf_slice)
        dist_maps.append(dt)

    dist_maps = np.stack(dist_maps, axis=0)
    dist_maps = torch.from_numpy(dist_maps).to(device=device, dtype=torch.float32)
    dist_maps = dist_maps.unsqueeze(1)
    return dist_maps


    # def _one_hot_encoder(self, input_tensor):
    #     tensor_list = []
    #     for i in range(self.n_classes):
    #         temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
    #         tensor_list.append(temp_prob.unsqueeze(1))
    #     output_tensor = torch.cat(tensor_list, dim=1)
    #     return output_tensor.float()

    # def _dice_loss(self, score, target):
    #     target = target.float()
    #     smooth = 1e-5
    #     intersect = torch.sum(score * target)
    #     y_sum = torch.sum(target * target)
    #     z_sum = torch.sum(score * score)
    #     loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    #     loss = 1 - loss
    #     return loss

    # def forward(self, inputs, target, weight=None, softmax=False):
    #     if softmax:
    #         inputs = torch.softmax(inputs, dim=1)
    #     target = self._one_hot_encoder(target)
    #     if weight is None:
    #         weight = [1] * self.n_classes
    #     assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
    #     class_wise_dice = []
    #     loss = 0.0
    #     for i in range(0, self.n_classes):
    #         dice = self._dice_loss(inputs[:, i], target[:, i])
    #         class_wise_dice.append(1.0 - dice.item())
    #         loss += dice * weight[i]
    #     return loss / self.n_classes