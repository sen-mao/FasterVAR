import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image
import os
from sklearn.decomposition import PCA

scale_schedule = [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 6, 6), (1, 8, 8), (1, 12, 12), (1, 16, 16), (1, 20, 20), (1, 24, 24), (1, 32, 32), (1, 40, 40), (1, 48, 48), (1, 64, 64)]
exit_points = {1: 1, 2: 5, 4: 21, 6: 57, 8: 121, 12: 265, 16: 521, 20: 921, 24: 1497, 32: 2521, 40: 4121, 48: 6425, 64: 7464}


def plot_ffn_features(features, layer_idx=-1):
    cond, uncond = features.unbind(dim=0)
    for feati, feat in zip(('cond', 'uncond'), (cond, uncond)):
        hw, c = feat.shape
        h = w = int(hw ** 0.5)
        dim = 3 if int(hw ** 0.5) > 3 else 1
        pca = PCA(n_components=dim)
        fea_pca = pca.fit_transform(feat.to(torch.float32).cpu())
        fea_min = np.min(fea_pca, axis=0)
        fea_max = np.max(fea_pca, axis=0)
        fea_pca = (fea_pca - fea_min) / ((fea_max - fea_min) + 1e-6)
        fea_pca = fea_pca.reshape(h, w, dim)
        image = (fea_pca * 255).astype(np.uint8)
        if dim == 1: image = np.squeeze(image, axis=-1)
        image = np.array(Image.fromarray(image).resize((h * 16, w * 16), resample=Image.Resampling.NEAREST))

        outdir = f'ffn_feat/scale{h}_layer{layer_idx}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = f'{outdir}/{feati}.png'
        if dim == 3:
            Image.fromarray(image, 'RGB').save(outdir)
        else:
            Image.fromarray(image, 'L').save(outdir)
    return