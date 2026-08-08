# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

def interpolate(tensor, size, mode, quantizer, is_semantic_scale):
    """
    arguments:
        tensor: (B,C,T,H,W)
        size: (C1,T,H1,W1)
        mode: str
        quantizer: quantizer
        is_semantic_scale: bool
    return:
        tensor: (B,*size)
    """
    B, C, T, H, W = tensor.shape
    C1, T, H1, W1 = size
    if quantizer.other_args.use_learnable_dim_proj:
        if is_semantic_scale:
            if C > C1:
                proj = quantizer.semantic_proj_down
            elif C < C1:
                proj = quantizer.semantic_proj_up
        else:
            if C > C1:
                proj = quantizer.detail_proj_down
            elif C < C1:
                proj = quantizer.detail_proj_up
        if C != C1:
            tensor = tensor.permute(0,2,3,4,1) #  (B,C,T,H,W) -> (B,T,H,W,C)
            tensor = proj(tensor) # (B,T,H,W,C1)
            tensor = tensor.permute(0,4,1,2,3) # (B,T,H,W,C1) -> (B,C1,T,H,W)
        tensor = F.interpolate(tensor, size=(T, H1, W1), mode=mode) # (B,C1,T,H,W) -> (B,C1,T,H1,W1)
        return tensor
    else:
        tensor = tensor.permute(0,2,1,3,4) # (B,C,T,H,W) -> (B,T,C,H,W)
        tensor = F.interpolate(tensor, size=(C1, H1, W1), mode=mode)
        tensor = tensor.permute(0,2,1,3,4) # (B,T,C1,H1,W1) -> (B,C1,T,H1,W1)
    return tensor

def get_scale_pack_info(scale_schedule, first_full_spatial_size_scale_index, args):
    meta = {}
    sid2clipid_innsid = {}
    clipid_innsid2sid = {}
    scales_per_clip = first_full_spatial_size_scale_index + 1
    compress_frames_inner_clip = args.frames_inner_clip
    total_clips = len(scale_schedule) // scales_per_clip
    context_clips = args.context_frames // args.frames_inner_clip
    for si in range(len(scale_schedule)):
        clipid = si // scales_per_clip
        if clipid == 0:
            frame_ss, frame_ee = 0, scale_schedule[scales_per_clip*(clipid+1)-1][0] # compressed_frame_ss and compressed_frame_ee
        else:
            frame_ss = scale_schedule[0][0] + (clipid-1) * compress_frames_inner_clip
            frame_ee = frame_ss + scale_schedule[scales_per_clip*(clipid+1)-1][0]
            if context_clips < total_clips-1:
                assert scale_schedule[si][0] == compress_frames_inner_clip
        sid2clipid_innsid[si] = (clipid, si % scales_per_clip)
        clipid_innsid2sid[(clipid, si % scales_per_clip)] = si
        # add clip ind for ref
        if si <= first_full_spatial_size_scale_index:
            meta[si] = {
                'clipid': clipid,
                'frame_ss': frame_ss,
                'frame_ee': frame_ee,
                'left_ref': [-1],
                'right_ref': [-1],
            }
        else:
            meta[si] = {
                'clipid': clipid,
                'frame_ss': frame_ss,
                'frame_ee': frame_ee,
                'left_ref': [clipid-1], # list(range(clipid-1, -1, -1)),
                'right_ref': [-1],
            }
            meta[si]['left_ref'] = meta[si]['left_ref'][:context_clips]
        # append inner scale ind to clip ind, (frame pack)
        if args.context_from_largest_no > 0:
            meta[si]['left_ref'] = [(meta[si]['left_ref'][i], max(0, scales_per_clip - args.context_from_largest_no - args.context_interval*i)) for i in range(len(meta[si]['left_ref']))]
            meta[si]['right_ref'] = [(meta[si]['right_ref'][i], max(0, scales_per_clip - args.context_from_largest_no - args.context_interval*i)) for i in range(len(meta[si]['right_ref']))]
    for si in meta:
        if args.context_from_largest_no > 0:
            meta[si]['left_ref_sids'], meta[si]['right_ref_sids'] = [], []
            for clipid, innsid in (meta[si]['left_ref']):
                if clipid != -1:
                    meta[si]['left_ref_sids'].append(clipid_innsid2sid[(clipid, innsid)])
            for fid, innsid in (meta[si]['right_ref']):
                if fid != -1:
                    meta[si]['right_ref_sids'].append(clipid_innsid2sid[(clipid, innsid)])
            meta[si]['ref_sids'] = meta[si]['left_ref_sids'] + meta[si]['right_ref_sids']
        else:
            meta[si]['ref_sids'] = list(range(si))
    return meta


def video_encode(
    vae,
    inp_B3HW,
    vae_features=None,
    self_correction=None,
    device='cuda',
    args=None,
    infer_mode=False,
    rope2d_freqs_grid=None,
    dynamic_resolution_h_w=None,
    tokens_remain=9999999,
    text_lens=[],
    **kwargs,
):
    return video_encode_global_bsc(
        vae,
        inp_B3HW,
        vae_features,
        self_correction,
        device,
        args,
        infer_mode,
        rope2d_freqs_grid,
        dynamic_resolution_h_w,
        tokens_remain,
        text_lens,
        **kwargs,
    )

def video_encode_global_bsc(
    vae,
    inp_B3HW,
    vae_features=None,
    self_correction=None,
    device='cuda',
    args=None,
    infer_mode=False,
    rope2d_freqs_grid=None,
    dynamic_resolution_h_w=None,
    tokens_remain=9999999,
    text_lens=[],
    **kwargs,
):
    if vae_features is None:
        raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW, scale_schedule=None, slice=True)
        raw_features_list = [raw_features]
        x_recon_raw = vae.decode(raw_features, slice=True)
        x_recon_raw = torch.clamp(x_recon_raw, min=-1, max=1)
        print(f'raw_features.shape: {raw_features.shape}')
    else:
        raw_features_list = vae_features
    # raw_features_list: list of [1,d,t,h,w]:
    gt_all_bit_indices = []
    pred_all_bit_indices = []
    var_input_list = []
    sequece_packing_scales = [] # with trunk
    flatten_packing_scales = []
    h_div_w_template_list = np.array(list(dynamic_resolution_h_w.keys()))
    visual_rope_cache_list = []
    noise_list = []
    scale_pack_info_list = []
    image_scale_repetition = json.loads(args.image_scale_repetition)
    video_scale_repetition = json.loads(args.video_scale_repetition)
    scales_in_one_clip = dynamic_resolution_h_w[h_div_w_template_list[0]][args.pn]['scales_in_one_clip']
    other_info_by_scale = []
    tokens_remain = tokens_remain-sum(text_lens)
    examples = len(raw_features_list)
    assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
    with torch.amp.autocast('cuda', enabled = False):
        for example_ind, raw_features in enumerate(raw_features_list):
            t, h, w = raw_features.shape[-3:]
            h_div_w = h / w
            mapped_h_div_w_template = h_div_w_template_list[np.argmin(np.abs(h_div_w-h_div_w_template_list))]
            min_t = min(dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'].keys())
            image_scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'][min_t]
            scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'][t]
            
            if args.apply_spatial_patchify:
                vae_scale_schedule = [(pt, ph + (ph % 2), pw + (pw % 2)) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule
            first_full_spatial_size_scale_index = len(image_scale_schedule) - 1
            scale_pack_info = get_scale_pack_info(vae_scale_schedule, first_full_spatial_size_scale_index, args)
            scale_pack_info_list.append(scale_pack_info)

            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2) # [B, d, t, h, w]
            else:
                codes_out = raw_features # [B, d, t, h, w]
            # print(f'{raw_features.shape=}, {scale_schedule=}')
            v_d = codes_out.shape[1]
            B, C, T, H, W = codes_out.shape
            if args.noise_input:
                noise = torch.randn((B, v_d, *vae_scale_schedule[0]), device=device, dtype=raw_features.dtype)
            else:
                noise = torch.zeros((B, v_d, *vae_scale_schedule[0]), device=device, dtype=raw_features.dtype)
            if infer_mode: noise_list.append(noise)
            next_var_input = noise
            valid_scales = len(vae_scale_schedule)
            assert len(image_scale_repetition) == len(image_scale_schedule), f'{len(image_scale_repetition)} != {len(image_scale_schedule)}'
            real_si = 0
            noise_apply_strength = self_correction.noise_apply_strength
            if args.noise_apply_random_one:
                image_scale_cnt = len(image_scale_schedule)
                video_scale_cnt = len(vae_scale_schedule)
                keep_image_si = random.randint(0, image_scale_cnt-1)
                if video_scale_cnt == image_scale_cnt:
                    keep_video_si = keep_image_si
                else:
                    keep_video_si = random.randint(image_scale_cnt, video_scale_cnt-1)
                noise_apply_strength = [noise_prob if i == keep_image_si or i == keep_video_si else 0 for i, noise_prob in enumerate(noise_apply_strength)]
            for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                tokens_remain = tokens_remain - np.array(scale_schedule[si]).prod()
                if tokens_remain < 0 and (not args.allow_less_one_elem_in_seq or examples > 1):
                    valid_scales = si
                    break
                    
                rel_si_in_one_clip = si % len(image_scale_schedule)
                if si < len(image_scale_schedule): # image
                    repeat_times = image_scale_repetition[rel_si_in_one_clip]
                else:
                    repeat_times = video_scale_repetition[rel_si_in_one_clip]
                select_repeat_idx = np.random.randint(0, repeat_times)
                frame_ss, frame_ee = scale_pack_info[si]['frame_ss'], scale_pack_info[si]['frame_ee']
                target = codes_out[:,:,frame_ss:frame_ee]
                for repeat_idx in range(repeat_times):
                    if (not infer_mode) and (repeat_idx==select_repeat_idx):
                        visual_rope_cache_list.append(get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule, si, real_si, device, args, scale_pack_info, first_full_spatial_size_scale_index))

                    if next_var_input.shape[-3:] != target.shape[-3:]:
                        next_var_input = F.interpolate(next_var_input, size=target.shape[-3:], mode=vae.quantizer.z_interplote_up).contiguous()
                    cum_var_input = next_var_input
                    this_scale_var_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down).contiguous()
                    if repeat_idx > 0 and args.inner_scale_boost:
                        residual = residual - quantized
                    else:
                        residual = target - cum_var_input
                    if args.use_two_stage_lfq:
                        if ph * pw >= vae.quantizer.detail_scale_min_tokens:
                            is_semantic_scale = False
                            C1 = vae.quantizer.detail_scale_dim
                            lfq = vae.quantizer.lfq_detail
                        else:
                            is_semantic_scale = True
                            C1 = vae.quantizer.semantic_scale_dim
                            lfq = vae.quantizer.lfq_semantic
                        residual = interpolate(residual, size=(C1, *vae_scale_schedule[si]), mode=vae.quantizer.z_interplote_down, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                    else:
                        residual = F.interpolate(residual, size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down).contiguous()
                        try:
                            lfq = vae.quantizer.lfq_detail
                        except:
                            lfq = vae.quantizer.lfq
                    quantized, _, bit_indices, loss = lfq(residual) # quantized shape: [B, d, t, h, w], bit_indices shape: [B,t,h,w,d]

                    if args.reduce_accumulate_error_method == 'bsc':
                        if si < min(len(vae_scale_schedule)-1, self_correction.noise_apply_layers):
                            pred_bit_indices, quantized = self_correction.apply_noise_requant(bit_indices, quantized, args, device, si, lfq, noise_apply_strength)
                        else:
                            pred_bit_indices = bit_indices
                    else:
                        raise NotImplementedError(args.reduce_accumulate_error_method)

                    if infer_mode or (repeat_idx==select_repeat_idx):
                        pred_all_bit_indices.append(pred_bit_indices)
                        var_input_list.append(this_scale_var_input)
                        gt_all_bit_indices.append(bit_indices)
                        other_info_by_scale.append({'largest_scale': scale_schedule[-1], 'real_si': si})
                    if args.use_two_stage_lfq:
                        quantized_scaled = interpolate(quantized, size=target.shape[-4:], mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
                    else:
                        quantized_scaled = F.interpolate(quantized, size=target.shape[-3:], mode=vae.quantizer.z_interplote_up).contiguous()
                    next_var_input = cum_var_input + quantized_scaled
                    real_si += 1
                
                if si < len(vae_scale_schedule)-1: # since first scale is [sos], here we only need len(vae_scale_schedule)-1 cum_var_input and x_BLC_wo_prefix
                    if vae_scale_schedule[si][-2:] == vae_scale_schedule[-1][-2:]:
                        if args.noise_input:
                            next_var_input = torch.randn((B, v_d, *vae_scale_schedule[si+1]), device=device, dtype=raw_features.dtype)
                        else:
                            next_var_input = torch.zeros((B, v_d, *vae_scale_schedule[si+1]), device=device, dtype=raw_features.dtype)
                        if infer_mode: noise_list.append(next_var_input)

            sequece_packing_scales.append(scale_schedule[:valid_scales])
            flatten_packing_scales.extend(scale_schedule[:valid_scales])
            if infer_mode:
                return noise_list, x_recon_raw, pred_all_bit_indices, None, None, scale_pack_info
    
    # train partial scales to enable training 480p without sp
    if args.allow_less_one_elem_in_seq and len(sequece_packing_scales) == 1 and np.array(sequece_packing_scales[0]).prod(-1).sum() > args.train_max_token_len:
        scale_schedule = sequece_packing_scales[0]

        if args.train_with_var_seq_len:
            if len(scale_schedule) == scales_in_one_clip * 3: # train 10s video
                outcomes = [
                        lambda: list(range(scales_in_one_clip)),
                        lambda: list(range(scales_in_one_clip + 8)),
                        lambda: list(range(scales_in_one_clip + 11)),
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+11],
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+12],
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+13],
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + list(range(2*scales_in_one_clip, 2*scales_in_one_clip + 11)),
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + list(range(2*scales_in_one_clip, 2*scales_in_one_clip + 11)),
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + [2*scales_in_one_clip + 11],
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + [2*scales_in_one_clip + 12],
                        lambda: [scales_in_one_clip-1] + [2*scales_in_one_clip-1] + [2*scales_in_one_clip + 13],
                    ]
            else:
                if args.drop_720p_last_scale:
                    outcomes = [
                        lambda: list(range(scales_in_one_clip)),
                        lambda: list(range(scales_in_one_clip + 8)),
                        lambda: list(range(scales_in_one_clip + 11)),
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+11],
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+12],
                        lambda: list(range(scales_in_one_clip + 8)) + [scales_in_one_clip+13],
                    ]
                else:
                    outcomes = [
                        lambda: list(range(scales_in_one_clip)),
                        lambda: list(range(scales_in_one_clip + 8)),
                        lambda: list(range(scales_in_one_clip + 11)),
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+11],
                        lambda: list(range(scales_in_one_clip + 11)) + [scales_in_one_clip+12],
                        lambda: [scales_in_one_clip-1] + [scales_in_one_clip+13],
                        lambda: [scales_in_one_clip-1] + [scales_in_one_clip+14],
                    ]
            
            probabilities = np.array(json.loads(args.video_var_len_prob), dtype=np.float32)[:len(outcomes)]
            probabilities /= probabilities.sum()

            # Choose one of the outcome functions based on the probabilities and execute it
            select_si_list = np.random.choice(outcomes, p=probabilities)()

        else:
            select_si_list = [scales_in_one_clip-1] # context first fsuper_scale_lengthsrame must be selected
            if args.train_192pshort:
                # select_si_list.append(2*scales_in_one_clip-4)
                if args.train_192pshort > 1:
                    select_si_list = list(range(0, scales_in_one_clip+args.train_192pshort))
                else:
                    select_si_list = list(range(0, scales_in_one_clip+11))
            else:
                select_si_list = list(range(0, scales_in_one_clip)) # all first frame must be selected
                select_si_list.append(scales_in_one_clip + np.random.choice([11, 12, 13], p=[0.7, 0.2, 0.1]))

            other_si_list = list(range(scales_in_one_clip-1)) + list(range(scales_in_one_clip, 2*scales_in_one_clip))
            other_si_list = list(set(other_si_list) - set(select_si_list))
            np.random.shuffle(other_si_list)
            train_token_len = np.array(scale_schedule)[select_si_list].prod(-1).sum() + text_lens[0]
            for si in other_si_list:
                token_len = np.array(scale_schedule[si]).prod(-1).sum()
                if train_token_len + token_len <= args.train_max_token_len:
                    train_token_len += token_len
                    select_si_list.append(si)
                
        select_si_list.sort()
        new_si_2_real_si, real_si_2_new_si = {}, {}
        for new_si, real_si in enumerate(select_si_list):
            new_si_2_real_si[new_si] = real_si
            real_si_2_new_si[real_si] = new_si
        
        sequece_packing_scales = [[scale_schedule[si] for si in select_si_list]]
        flatten_packing_scales = [flatten_packing_scales[si] for si in select_si_list]
        gt_all_bit_indices = [gt_all_bit_indices[si] for si in select_si_list]
        pred_all_bit_indices = [pred_all_bit_indices[si] for si in select_si_list]
        var_input_list = [var_input_list[si] for si in select_si_list]
        visual_rope_cache_list = [visual_rope_cache_list[si] for si in select_si_list]
        other_info_by_scale = [other_info_by_scale[si] for si in select_si_list]

        # remap scale_pack_info
        new_scale_pack_info = {}
        for new_query_sid in new_si_2_real_si:
            real_query_sid = new_si_2_real_si[new_query_sid]
            new_scale_pack_info[new_query_sid] = {'ref_sids': []}
            for real_ref_sid in scale_pack_info_list[0][real_query_sid]['ref_sids']:
                new_ref_sid = real_si_2_new_si[real_ref_sid]
                new_scale_pack_info[new_query_sid]['ref_sids'].append(new_ref_sid)
        scale_pack_info_list = [new_scale_pack_info]
        
    scale_lengths = [ pt * ph * pw for pt,ph,pw in flatten_packing_scales]
    scale_lengths = scale_lengths + text_lens
    valid_scales = len(flatten_packing_scales) + len(text_lens)

    cur_seq_len = np.sum(scale_lengths)
    if args.train_with_var_seq_len:
        pad_seq_len = int(np.ceil(cur_seq_len/args.pad_to_multiplier))*args.pad_to_multiplier - cur_seq_len
    else:
        pad_seq_len = args.train_max_token_len - cur_seq_len
    assert pad_seq_len >= 0, f'pad_seq_len: {pad_seq_len} < 0, {scale_lengths=}'
    if pad_seq_len:
        scale_lengths = scale_lengths + [pad_seq_len]
    max_sid_nums = 2000
    querysid_refsid = torch.zeros((max_sid_nums, max_sid_nums), device=args.device, dtype=torch.bool) # Attention! this shape should be the same for different iterations !!!
    for i in range(valid_scales):
        querysid_refsid[i][i] = True
    base = 0
    for ind, scale_schedule in enumerate(sequece_packing_scales):
        scale_pack_info = scale_pack_info_list[ind]
        for local_querysid in range(len(scale_schedule)):
            global_querysid = local_querysid + base
            global_text_sid = len(flatten_packing_scales) + ind
            querysid_refsid[global_querysid][global_text_sid] = True
            for local_refsid in (scale_pack_info[local_querysid]['ref_sids']):
                global_refsid = base + local_refsid
                querysid_refsid[global_querysid][global_refsid] = True
        base += len(scale_schedule)
        
    gt_ms_idx_Bl = []
    for item in gt_all_bit_indices:
        if args.apply_spatial_patchify:
            # item shape: (B,t,H,W,d)
            item = item.permute(0,1,4,2,3) # (B,t,d,H,W)
            # (B,t,d,H,W) -> (B,t,4d,H/2,W/2)
            item = torch.nn.functional.pixel_unshuffle(item, 2)
            _, tt, dd, hh, ww = item.shape
            # (B,t,4d,H/2,W/2) -> (B,t,H/2,W/2,4d) -> (B,t*H/2*w/2,4d)
            item = item.permute(0,1,3,4,2).reshape(B, tt*hh*ww, dd)
        else:
            _, tt, hh, ww, dd = item.shape
            item = item.reshape(B, tt*hh*ww, dd)
        gt_ms_idx_Bl.append(item.type(torch.long))
    gt_BLC = gt_ms_idx_Bl # torch.cat(gt_ms_idx_Bl, 1).contiguous().type(torch.long)
    for i in range(len(var_input_list)):
        if args.apply_spatial_patchify:
            # (B,d,t,H,W) -> (B,t,d,H,W) -> (B,t,4d,H/2,W/2) -> (B,t,H/2,W/2,4d)
            var_input_list[i] = torch.nn.functional.pixel_unshuffle(var_input_list[i].permute(0,2,1,3,4), 2).permute(0,1,3,4,2)
            var_input_list[i] = var_input_list[i].reshape(B, -1, 4*vae.codebook_dim)
        else:
            # (B,d,t,H,W) -> (B,t,H,W,d)
            var_input_list[i] = var_input_list[i].permute(0,2,3,4,1)
            var_input_list[i] = var_input_list[i].reshape(B, -1, vae.codebook_dim)
    x_BLC = torch.cat(var_input_list, 1)
    visual_rope_cache = torch.cat(visual_rope_cache_list, dim=4)
    x_BLC_mask = None
    return x_BLC, x_BLC_mask, gt_BLC, pred_all_bit_indices, visual_rope_cache, sequece_packing_scales, scale_lengths, querysid_refsid, other_info_by_scale


def video_decode(
    vae,
    all_indices,
    scale_schedule,
    label_type,
    args=None,
    noise_list=None,
    trunc_scales=-1,
    **kwargs,
):
    image_scale_repetition = json.loads(args.image_scale_repetition)
    video_scale_repetition = json.loads(args.video_scale_repetition)
    assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
    real_si = 0
    noise_ptr = 0
    summed_codes = [noise_list[noise_ptr]]
    noise_ptr += 1
    v_d = summed_codes[0].shape[1]
    for si, (pt, ph, pw) in enumerate(scale_schedule):
        if trunc_scales > 0 and si >= trunc_scales:
            break
        if si < len(image_scale_repetition): # image
            repeat_times = image_scale_repetition[si%len(image_scale_repetition)]
        else:
            repeat_times = video_scale_repetition[si%len(image_scale_repetition)]
        for repeat_idx in range(repeat_times):
            tgt_shape = (pt, scale_schedule[-1][-2], scale_schedule[-1][-1])
            if args.use_two_stage_lfq:
                if ph * pw >= vae.quantizer.detail_scale_min_tokens:
                    is_semantic_scale = False
                    lfq = vae.quantizer.lfq_detail
                else:
                    is_semantic_scale = True
                    lfq = vae.quantizer.lfq_semantic
                codes = lfq.indices_to_codes(all_indices[real_si], label_type)
                codes = interpolate(codes, size=(v_d, *tgt_shape), mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer, is_semantic_scale=is_semantic_scale).contiguous()
            else:
                codes = vae.quantizer.lfq_detail.indices_to_codes(all_indices[real_si], label_type)
                codes = F.interpolate(codes, size=tgt_shape, mode=vae.quantizer.z_interplote_up).contiguous()
            
            summed_codes[-1] = F.interpolate(summed_codes[-1], size=tgt_shape, mode=vae.quantizer.z_interplote_up).contiguous()
            summed_codes[-1] += codes
            real_si += 1
        if si < len(scale_schedule) - 1:
            if scale_schedule[si][-3:] == tgt_shape:
                summed_codes.append(noise_list[noise_ptr])
                noise_ptr += 1
    if trunc_scales < 0:
        assert real_si == len(all_indices), f'all_repeated_scales={real_si} != len(all_indices)={len(all_indices)}'
    summed_codes = torch.cat(summed_codes, dim=-3)
    x_recon = vae.decode(summed_codes, slice=True)
    x_recon = torch.clamp(x_recon, min=-1, max=1)
    return x_recon

def get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule, sid, real_sid, device=None, args=None, scale_pack_info=None, first_full_spatial_size_scale_index=None):
    # freqs_scales: (2, max_scales, ceil(dim_div_2 / 4))
    # freqs_frames: (2, max_frames, ceil(dim_div_2 / 4))
    rope2d_freqs_grid['freqs_scales'] = rope2d_freqs_grid['freqs_scales'].to(device)
    rope2d_freqs_grid['freqs_frames'] = rope2d_freqs_grid['freqs_frames'].to(device)
    rope2d_freqs_grid['freqs_height'] = rope2d_freqs_grid['freqs_height'].to(device)
    rope2d_freqs_grid['freqs_width'] = rope2d_freqs_grid['freqs_width'].to(device)
    upt, uph, upw = scale_schedule[-1]
    pt, ph, pw = scale_schedule[sid]
    dim_div_2_div_4 = rope2d_freqs_grid['freqs_scales'].shape[2]
    dim_div_2 = dim_div_2_div_4 * 4
    f_scales = rope2d_freqs_grid['freqs_scales'][:, real_sid].reshape(2, 1, dim_div_2_div_4)
    frame_ss, frame_ee = scale_pack_info[sid]['frame_ss'], scale_pack_info[sid]['frame_ee']
    f_frames = rope2d_freqs_grid['freqs_frames'][:, frame_ss:frame_ee]
    f_height = rope2d_freqs_grid['freqs_height'][:, (torch.arange(ph) * (uph / ph)).round().int()]
    f_width = rope2d_freqs_grid['freqs_width'][:, (torch.arange(pw) * (upw / pw)).round().int()]
    rope_embeds = torch.cat([
        f_scales[   :,     :,  None,   None,   None,   :].expand(-1, -1, pt, ph, pw, -1),
        f_frames[   :,  None,      :,  None,   None,   :].expand(-1,  1, -1, ph, pw, -1),
        f_height[   :,  None,  None,      :,   None,   :].expand(-1,  1, pt, -1, pw, -1),
        f_width[    :,  None,  None,   None,      :,   :].expand(-1,  1, pt, ph, -1, -1),
    ], dim=-1)  # (2, 1, pt, ph, pw, dim_div_2)
    rope_embeds = rope_embeds.reshape(2, 1, 1, 1, 1*pt*ph*pw, dim_div_2)  # (2, 1, 1, 1, 1*pt*ph*pw, dim_div_2)
    return rope_embeds
