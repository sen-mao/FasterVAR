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

# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight, attn_weight @ value

def plot_attention_weights(query, key, value, scale=None, oup=None, layer_idx=-1):
    assert layer_idx != -1
    ql = int(query.shape[0]/2)
    cond_query, uncond_query = query[:ql], query[ql:]
    kl = int(key.shape[0] / 2)
    cond_key, uncond_key = key[:kl], key[kl:]
    vl = int(value.shape[0] / 2)
    cond_value, uncond_vale = value[:vl], value[vl:]

    branch = ('cond', 'uncond')
    attention_maps = {'cond': [], 'uncond': []}
    outputs = {'cond': None, 'uncond': None}
    for i, (q, k, v) in enumerate(zip((cond_query, uncond_query), (cond_key, uncond_key), (cond_value, uncond_vale))):
        ss, head_num, head_dim = q.shape
        s = int(ss ** 0.5)
        outputs[branch[i]] = torch.zeros_like(q)
        for hi in range(head_num):
            attn_maps, output = scaled_dot_product_attention(q[:,hi,:], k[:,hi,:], v[:,hi,:], scale=scale)
            attention_maps[branch[i]] += [attn_maps]
            outputs[branch[i]][:, hi, :] = output

            ## cross-attention map
            images = []
            _, n = attn_maps.shape
            for ni in range(n):
                attn_map = attn_maps[:, ni]
                attn_map = attn_map.T.reshape(s, s)
                # attn_map = (attn_map - attn_map.min()) / ((attn_map.max() - attn_map.min()) + 1e-6)
                # image = 255 * attn_map
                image = 255 * attn_map / attn_map.max()
                image = image.cpu().numpy().astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((s*16, s*16), resample=Image.Resampling.NEAREST))
                images.append(image)

            images = np.concatenate(images, axis=1)

            outdir = f'cross_attention_map/scale{s}_layer{layer_idx}'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outdir = f'{outdir}/head{hi}_{branch[i]}.png'
            Image.fromarray(images, 'L').save(outdir)

            ## save feature of cross-attention
            hw, c = output.shape
            h = w = int(hw ** 0.5)
            dim = 3 if int(ql ** 0.5) > 3 else 1
            pca = PCA(n_components=dim)
            fea_pca = pca.fit_transform(output.to(torch.float32).cpu())
            fea_min = np.min(fea_pca, axis=0)
            fea_max = np.max(fea_pca, axis=0)
            fea_pca = (fea_pca - fea_min) / ((fea_max - fea_min) + 1e-6)
            fea_pca = fea_pca.reshape(h, w, dim)
            image = (fea_pca * 255).astype(np.uint8)
            if dim == 1: image = np.squeeze(image, axis=-1)
            image = np.array(Image.fromarray(image).resize((h * 16, w * 16), resample=Image.Resampling.NEAREST))

            outdir = f'cross_attention_feat/scale{h}_layer{layer_idx}'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outdir = f'{outdir}/head{hi}_{branch[i]}.png'
            if dim == 3:
                Image.fromarray(image, 'RGB').save(outdir)
            else:
                Image.fromarray(image, 'L').save(outdir)
    return


