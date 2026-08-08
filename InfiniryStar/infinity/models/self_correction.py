# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import cv2
import torch
import torch.nn.functional as F
import numpy as np

from infinity.schedules.dynamic_resolution import get_first_full_spatial_size_scale_index


def labels2image(all_indices, label_type='int_label', scale_schedule=None):
    summed_codes, recons_imgs = self.vae.decode_from_indices(all_indices, scale_schedule, label_type)
    recons_img = recons_imgs[0]
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)[:,:,::-1]
    return recons_img

def features2image(raw_features):
    recons_imgs = self.vae.decode(raw_features.squeeze(-3))
    recons_img = recons_imgs[0]
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)[:,:,::-1]
    return recons_img

class SelfCorrection(object):
    def __init__(self, vae, args):
        self.noise_apply_layers = args.noise_apply_layers
        self.noise_apply_requant = args.noise_apply_requant
        self.noise_apply_strength = args.noise_apply_strength
        if not isinstance(self.noise_apply_strength, list):
            self.noise_apply_strength = str(self.noise_apply_strength)
            self.noise_apply_strength = list(map(float, self.noise_apply_strength.split(',')))
        if len(self.noise_apply_strength) == 1:
            self.noise_apply_strength = self.noise_apply_strength[0]
        self.apply_spatial_patchify = args.apply_spatial_patchify
        self.vae = vae
        print(f'self.noise_apply_strength: {self.noise_apply_strength}')

    def apply_noise_requant(self, bit_indices, quantized, args, device, si, lfq=None, noise_apply_strength=None):
        if lfq is None:
            lfq = self.vae.quantizer.lfq
        if noise_apply_strength is None:
            noise_apply_strength = self.noise_apply_strength
        if isinstance(noise_apply_strength, list):
            noise_apply_strength = np.random.randint(0, max(1, 100 * noise_apply_strength[si]+1)) * 0.01
        else:
            noise_apply_strength = np.random.randint(0, max(1, 100 * noise_apply_strength+1)) * 0.01
        mask = torch.rand(*bit_indices.shape, device=device) < noise_apply_strength
        pred_bit_indices = bit_indices.clone()
        if args.num_of_label_value == 2:
            pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
        else:
            noise = torch.randint(0, args.num_of_label_value, bit_indices.shape, dtype=bit_indices.dtype, device=device)
            pred_bit_indices[mask] = noise[mask]
        if self.noise_apply_requant:
            quantized = lfq.indices_to_codes(pred_bit_indices, label_type = 'bit_label')
        return pred_bit_indices, quantized
    
    def visualize(self, vae_scale_schedule, inp_B3HW, gt_all_bit_indices, pred_all_bit_indices):
        gt_img = (inp_B3HW.squeeze(-3) + 1) / 2 * 255
        gt_img = gt_img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)[:,:,::-1]
        recons_img_2 = labels2image(gt_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        recons_img_3 = labels2image(pred_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        cat_image = np.concatenate([gt_img, recons_img_2, recons_img_3], axis=1)
        save_path = osp.abspath('non_teacher_force.jpg')
        cv2.imwrite(save_path, cat_image)
        print(f'Save to {save_path}')
        import pdb; pdb.set_trace()
        print(cat_image.shape)
        