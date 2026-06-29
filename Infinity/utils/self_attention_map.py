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
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight, attn_weight @ value

def plot_attention_weights(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, layer_idx=-1):
    # if int(query.shape[2]**0.5) != 48:
    #     return
    attn_weight, output = scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    assert layer_idx != -1
    _, head_num, ql, kl = attn_weight.shape

    for hi in range(head_num):
        cond, uncond = attn_weight[:, hi].unbind(dim=0)
        for feati, feat in zip(('cond', 'uncond'), (cond, uncond)):
            feat = feat.reshape(-1,)
            feat = feat.detach().cpu().numpy()
            fea_min = np.min(feat, axis=0)
            fea_max = np.max(feat, axis=0)
            feat = (feat - fea_min) / ((fea_max - fea_min) + 1e-6)
            feat = feat.reshape(ql, kl)
            image = (feat * 255).astype(np.uint8)
            image = np.array(Image.fromarray(image))  # .resize((h, w), resample=Image.Resampling.NEAREST)

            ## save as scale*scale
            outdir = f'self_attention_map/scale{int(ql**0.5)}_layer{layer_idx}'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outdir = f'{outdir}/head{hi}_{feati}.png'
            Image.fromarray(image, 'L').save(outdir)

            ## reshape to (scale, scale)
            outdir = f'self_attention_map_scale_scale/scale{int(ql ** 0.5)}_layer{layer_idx}'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            sacle_i = scale_schedule.index((1, int(ql ** 0.5), int(ql ** 0.5)))
            for si, scale in enumerate(scale_schedule[:sacle_i+1]):
                sti = 0 if si == 0 else exit_points[scale_schedule[si-1][-1]]
                endi = exit_points[scale[-1]]
                image_scale = image[:, sti:endi]

                s, n = image_scale.shape
                s = int(s**0.5)
                imgs = image_scale.T.reshape(n, s, s)
                cols = math.ceil(math.sqrt(n))
                rows = math.ceil(n / cols)
                # 创建白底大图（uint8, 255 表示白色）
                canvas = np.ones((rows * s, cols *s), dtype=np.uint8) * 255
                for idx in range(n):
                    row = idx // cols
                    col = idx % cols
                    canvas[row * s:(row + 1) * s, col * s:(col + 1) * s] = imgs[idx]

                outdiri = f'{outdir}/head{hi}_{feati}_scale{scale[-1]}.png'
                Image.fromarray(canvas, 'L').save(outdiri)

        ## save feature of self-attention
        cond_out, uncond_out = output[:, hi].unbind(dim=0)
        for outi, out in zip(('cond_out', 'uncond_out'), (cond_out, uncond_out)):
            hw, c = out.shape
            h = w = int(hw**0.5)
            dim = 3 if int(ql ** 0.5) > 3 else 1
            pca = PCA(n_components=dim)
            fea_pca = pca.fit_transform(out.to(torch.float32).cpu())
            fea_min = np.min(fea_pca, axis=0)
            fea_max = np.max(fea_pca, axis=0)
            fea_pca = (fea_pca - fea_min) / ((fea_max - fea_min) + 1e-6)
            fea_pca = fea_pca.reshape(h, w, dim)
            image = (fea_pca * 255).astype(np.uint8)
            if dim == 1: image = np.squeeze(image, axis=-1)
            image = np.array(Image.fromarray(image).resize((h*16, w*16), resample=Image.Resampling.NEAREST))

            outdir = f'self_attention_feat/scale{h}_layer{layer_idx}'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outdir = f'{outdir}/head{hi}_{outi}.png'
            if dim == 3:
                Image.fromarray(image, 'RGB').save(outdir)
            else:
                Image.fromarray(image, 'L').save(outdir)
    return


