import os
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as nd
from torch import nn
from PIL import Image
from scipy.ndimage import distance_transform_edt


class Loss_Distance(nn.Module):
    def __init__(self, root_path, sigma_param):
        super(Loss_Distance, self).__init__()
        self.root_path = root_path
        self.sigma = sigma_param

    def forward(self, cmb_prob: torch.Tensor, csf_mask_path) -> torch.Tensor:
        csf_mask_path = os.path.join(self.root_path, csf_mask_path)
        cmb_mask_path = csf_mask_path.replace("cerebrospinal+fluid", "brain+microbleeds")
        gt_csf = Image.open(csf_mask_path).convert('L')
        gt_cmb = Image.open(cmb_mask_path).convert('L')
        gt_csf = torch.from_numpy(np.array(gt_csf)).to(cmb_prob.device).unsqueeze(0).unsqueeze(0).float()
        gt_cmb = torch.from_numpy(np.array(gt_cmb)).to(cmb_prob.device).unsqueeze(0).unsqueeze(0).float()

        # distance map from CSF (fixed wrt model => fine)
        distance_map = compute_csf_distance_map(gt_csf).to(cmb_prob.device)

        '''
        The larger distance map value, the larger weight.
        '''
        weight = torch.exp(-distance_map / self.sigma)   # Bx1xHxW

        # background mask outside CMB
        bg_mask = (gt_cmb == 0).float().to(cmb_prob.device)

        # model prediction
        cmb_prob = torch.sigmoid(cmb_prob)
        cmb_prob = cmb_prob * (cmb_prob > 0.2).float()

        # resize prediction to match mask
        cmb_prob = F.interpolate(cmb_prob, size=bg_mask.shape[2:], mode='bilinear', align_corners=False)

        '''
        In order to have only false positive cases, 
        the cmb prediction that overlappes with GT CMB are removed.
        '''
        fp_soft = cmb_prob * bg_mask  # only region where GT is not CMB

        # distance-weighted loss
        loss_map = fp_soft * weight

        # final scalar loss (differentiable)
        # loss_distance = loss_map.mean()
        loss_distance = loss_map.sum() / torch.clamp((weight > 0).sum(), min=1.0)
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
        '''
        1.0 - csf_slice instead of just using csf_slice. 
        we want background(cmb potential) as target and csf as background.
        So we want to the distance in perspective of non-csf region.
        '''
        dt = distance_transform_edt(1.0 - csf_slice)
        dist_maps.append(dt)

    dist_maps = np.stack(dist_maps, axis=0)
    dist_maps = torch.from_numpy(dist_maps).to(device=device, dtype=torch.float32)
    dist_maps = dist_maps.unsqueeze(1)
    return dist_maps