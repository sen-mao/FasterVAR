# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import os
import json

import numpy as np
import torch
import torch.nn.functional as F

semantic_scale_ind = 7
detail_frame_inds = [18,19]

def flatten_two_level_list(two_level_list):
    flatten_list = []
    for item in two_level_list:
        flatten_list.extend(item)
    return flatten_list

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
            frame_ss, frame_ee = 0, scale_schedule[scales_per_clip*1-1][0]
        else:
            frame_ss = scale_schedule[scales_per_clip*1-1][0] + (clipid-1) * compress_frames_inner_clip
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
                'left_ref': [clipid-1],
                'right_ref': [-1],
            }
        # append inner scale ind to clip ind, (frame pack)
        if args.context_from_largest_no > 0:
            meta[si]['left_ref'] = [(meta[si]['left_ref'][i], max(0, scales_per_clip - args.context_from_largest_no - args.context_interval*i)) for i in range(len(meta[si]['left_ref']))]
            meta[si]['right_ref'] = [(meta[si]['right_ref'][i], max(0, scales_per_clip - args.context_from_largest_no - args.context_interval*i)) for i in range(len(meta[si]['right_ref']))]
    for si in meta:
        meta[si]['left_ref_sids'], meta[si]['right_ref_sids'] = [], []
        for clipid, innsid in (meta[si]['left_ref']):
            if clipid != -1:
                meta[si]['left_ref_sids'].append(clipid_innsid2sid[(clipid, innsid)])
        for fid, innsid in (meta[si]['right_ref']):
            if fid != -1:
                meta[si]['right_ref_sids'].append(clipid_innsid2sid[(clipid, innsid)])
        meta[si]['ref_sids'] = meta[si]['left_ref_sids'] + meta[si]['right_ref_sids']
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
    text_lens=[],
    caption_nums=None,
    rank=0,
    vis_verbose=False,
    np_generator=None,
    skip_last=0,
    train_max_token_len=0,
    first_frame_features=[],
    **kwargs,
):
    if vae_features is None:
        raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW, scale_schedule=None, slice=True)
        raw_features_list = [raw_features]
        x_recon_raw = vae.decode(raw_features[0], slice=True)
        x_recon_raw = torch.clamp(x_recon_raw, min=-1, max=1)
        print(f'raw_features.shape: {raw_features[0].shape}')
    else:
        raw_features_list = vae_features

    if np_generator is not None:
        random_obj = np_generator
    else:
        random_obj = np.random.default_rng()
    
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
    select_repeat_idx_list = []
    examples = len(raw_features_list)
    assert len(image_scale_repetition) == len(video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
    assert examples == 1, f'currently only support examples==1, buf found {examples=}'
    with torch.amp.autocast('cuda', enabled = False):
        for example_ind, complete_raw_features in enumerate(raw_features_list):
            complete_raw_features = complete_raw_features[0]
            if first_frame_features[example_ind] is None:
                first_frame_feature_ = complete_raw_features[:,:,0:1] # [B,d,1,h,w]
            else:
                first_frame_feature_ = first_frame_features[example_ind][0] # [B,d,1,h,w]
            # assert complete_raw_features.shape[-3] > 21
            # 前21帧，构成一个 I1V1 的 clip
            # 后面的 t-21 帧，构成一个 V2 的 clip，它的 condition 是 V1 resize 后的结果和 V1 的最后一帧
            new_raw_features_list = [complete_raw_features[:,:,:21], complete_raw_features[:,:,21:]]
            t, h, w = new_raw_features_list[0].shape[-3:]
            h_div_w = h / w
            mapped_h_div_w_template = h_div_w_template_list[np.argmin(np.abs(h_div_w-h_div_w_template_list))]
            min_t = min(dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'].keys())
            image_scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'][min_t]
            scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][args.pn]['pt2scale_schedule'][t]
            
            for ind, raw_features in enumerate(new_raw_features_list):
                if raw_features.numel() == 0:
                    break
                mode = 'first_iv_clip'
                global_si_base = 0
                if ind == 1:
                    scale_schedule = scale_schedule[scales_in_one_clip:]
                    scale_schedule = [(raw_features.shape[-3], ph, pw) for pt, ph, pw in scale_schedule]
                    mode = 'second_v_clip'
                    global_si_base = sum(image_scale_repetition) + sum(video_scale_repetition)

                if args.apply_spatial_patchify:
                    vae_scale_schedule = [(pt, ph*2, pw*2) for pt, ph, pw in scale_schedule]
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
                valid_scales = len(vae_scale_schedule) - skip_last
                assert len(image_scale_repetition) == len(image_scale_schedule), f'{len(image_scale_repetition)} != {len(image_scale_schedule)}'
                real_si = 0
                noise_apply_strength = self_correction.noise_apply_strength
                for si in range(valid_scales):
                    pt, ph, pw = vae_scale_schedule[si]
                    rel_si_in_one_clip = si % len(image_scale_schedule)
                    if si < len(image_scale_schedule): # image
                        repeat_times = image_scale_repetition[rel_si_in_one_clip]
                    else:
                        repeat_times = video_scale_repetition[rel_si_in_one_clip]
                    select_repeat_idx = random_obj.integers(0, repeat_times)
                    select_repeat_idx_list.append(select_repeat_idx)
                    frame_ss, frame_ee = scale_pack_info[si]['frame_ss'], scale_pack_info[si]['frame_ee']
                    target = codes_out[:,:,frame_ss:frame_ee]
                    for repeat_idx in range(repeat_times):
                        if (not infer_mode) and (repeat_idx==select_repeat_idx):
                            visual_rope_cache_list.append(get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule[-1], scale_schedule[si], list(range(frame_ss, frame_ee)), real_si, device))
                        if next_var_input.shape[-3:] != target.shape[-3:]:
                            next_var_input = F.interpolate(next_var_input, size=target.shape[-3:], mode=vae.quantizer.z_interplote_up).contiguous()
                        cum_var_input = next_var_input
                        this_scale_var_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si], mode=vae.quantizer.z_interplote_down).contiguous()
                        residual = target - cum_var_input
                        if args.use_two_stage_lfq:
                            if rel_si_in_one_clip >= args.semantic_scales:
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
                                pred_bit_indices, quantized = self_correction.apply_noise_requant(bit_indices, quantized, args, device, si, lfq, noise_apply_strength, num_lvl=2, np_generator=random_obj)
                            else:
                                pred_bit_indices = bit_indices
                        else:
                            raise NotImplementedError(args.reduce_accumulate_error_method)

                        if infer_mode or (repeat_idx==select_repeat_idx):
                            pred_all_bit_indices.append(pred_bit_indices)
                            var_input_list.append(this_scale_var_input)
                            gt_all_bit_indices.append(bit_indices)
                            other_info_by_scale.append({'largest_scale': scale_schedule[-1], 'real_si': si, 'mode': mode, 'global_si': real_si+global_si_base})
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
                if ind == 0:
                    former_clip_features = raw_features[:,:,-20:]

            
            if infer_mode:
                return noise_list, x_recon_raw, pred_all_bit_indices, None, None, scale_pack_info
        
    if vis_verbose:
        print(f'Rank={rank}, {sequece_packing_scales=} {select_repeat_idx_list=}', force=True)
    
    if args.train_second_clip_only:
        drop_scales = len(sequece_packing_scales[0])
        sequece_packing_scales = sequece_packing_scales[1:]
        scale_pack_info_list = scale_pack_info_list[1:]
        gt_all_bit_indices = gt_all_bit_indices[drop_scales:]
        pred_all_bit_indices = pred_all_bit_indices[drop_scales:]
        other_info_by_scale = other_info_by_scale[drop_scales:]
        var_input_list = var_input_list[drop_scales:]
        visual_rope_cache_list = visual_rope_cache_list[drop_scales:]

    flatten_packing_scales = flatten_two_level_list(sequece_packing_scales)

    def add_noise(features, noise_choices=[0.00, 0.15, 0.30]):
        feature_std = features.std()
        rand_noise_strength = np.random.choice(noise_choices)
        return features + rand_noise_strength * feature_std * torch.randn_like(features)

    # add conditions
    semantic_condition = F.interpolate(former_clip_features, size=(20, *scale_schedule[semantic_scale_ind][-2:]), mode=vae.quantizer.z_interplote_down)
    semantic_condition = add_noise(semantic_condition)
    assert former_clip_features.shape[2] == 20
    detail_condition = torch.cat([first_frame_feature_, add_noise(former_clip_features[:,:,detail_frame_inds])], dim=2)
    var_input_list.extend([semantic_condition, detail_condition])
    
    visual_rope_cache_list.append(get_visual_rope_embeds(rope2d_freqs_grid, detail_condition.shape[-3:], semantic_condition.shape[-3:], list(range(1, 21)), 800, device))
    visual_rope_cache_list.append(get_visual_rope_embeds(rope2d_freqs_grid, detail_condition.shape[-3:], detail_condition.shape[-3:], [0]+[item+1 for item in detail_frame_inds], 801, device))

    # set scale_lengths and querysid_refsid
    scale_lengths = [ pt * ph * pw for pt,ph,pw in flatten_packing_scales]
    scale_lengths = scale_lengths + [torch.tensor(semantic_condition.shape[-3:]).prod().item(), torch.tensor(detail_condition.shape[-3:]).prod().item()]
    scale_lengths = scale_lengths + text_lens

    valid_scales = len(scale_lengths)
    pad_seq_len = train_max_token_len - np.sum(scale_lengths)
    assert pad_seq_len >= 0, f'pad_seq_len: {pad_seq_len} < 0, {scale_lengths=}'
    if pad_seq_len:
        scale_lengths = scale_lengths + [pad_seq_len]
    max_sid_nums = 2000
    querysid_refsid = torch.zeros((max_sid_nums, max_sid_nums), device=args.device, dtype=torch.bool) # Attention! this shape should be the same for different iterations !!!
    for i in range(valid_scales):
        querysid_refsid[i][i] = True
    base = 0
    for ind, scale_schedule in enumerate(sequece_packing_scales):
        real_example_ind = ind // 2 # for each example, there are two scale_schedule
        scale_pack_info = scale_pack_info_list[ind]
        for local_querysid in range(len(scale_schedule)):
            global_querysid = base + local_querysid
            if other_info_by_scale[base+local_querysid]['mode'] == 'first_iv_clip':
                global_text_sid = len(flatten_packing_scales) + 2 + sum(caption_nums[:real_example_ind]) + 0
                querysid_refsid[global_querysid][global_text_sid] = True
            elif other_info_by_scale[base+local_querysid]['mode'] == 'second_v_clip':
                global_text_sid = len(flatten_packing_scales) + 2 + sum(caption_nums[:real_example_ind]) + 1
                querysid_refsid[global_querysid][global_text_sid] = True
                querysid_refsid[global_querysid][len(flatten_packing_scales)+0] = True # i can see semantic condition
                querysid_refsid[global_querysid][len(flatten_packing_scales)+1] = True # i can see detail condition
            else:
                raise ValueError(f'Unknown mode: {other_info_by_scale[base+local_querysid]["mode"]}')
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
    return x_BLC, x_BLC_mask, gt_BLC, pred_all_bit_indices, visual_rope_cache, sequece_packing_scales, scale_lengths, querysid_refsid, other_info_by_scale, pad_seq_len

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
    summed_codes = []
    scales_in_one_clip = args.first_full_spatial_size_scale_index+1
    clips = len(noise_list) - 1
    for clip_id in range(clips):
        if clip_id == 1:
            scale_schedule = scale_schedule[(args.first_full_spatial_size_scale_index+1):]
            t = all_indices[-1].shape[1] # [B,t,h,w,d]
            scale_schedule = [(t, ph, pw) for pt, ph, pw in scale_schedule]
        summed_codes.append(noise_list[noise_ptr])
        noise_ptr += 1
        v_d = summed_codes[0].shape[1]
        for si, (pt, ph, pw) in enumerate(scale_schedule):
            if si < len(image_scale_repetition): # image
                repeat_times = image_scale_repetition[si%len(image_scale_repetition)]
            else:
                repeat_times = video_scale_repetition[si%len(image_scale_repetition)]
            for repeat_idx in range(repeat_times):
                tgt_shape = (pt, scale_schedule[-1][-2], scale_schedule[-1][-1])
                if args.use_two_stage_lfq:
                    if (si % scales_in_one_clip) >= args.semantic_scales:
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
            
            if si < len(scale_schedule)-1 and scale_schedule[si][-2:] == tgt_shape[-2:]:
                summed_codes.append(noise_list[noise_ptr])
                noise_ptr += 1
            
    summed_codes = torch.cat(summed_codes, dim=-3)
    x_recon = vae.decode(summed_codes, slice=True)
    x_recon = torch.clamp(x_recon, min=-1, max=1)
    return x_recon

def get_visual_rope_embeds(rope2d_freqs_grid, largest_scale, current_scale, t_list, real_sid, device=None):
    # freqs_scales: (2, max_scales, ceil(dim_div_2 / 4))
    # freqs_frames: (2, max_frames, ceil(dim_div_2 / 4))
    rope2d_freqs_grid['freqs_scales'] = rope2d_freqs_grid['freqs_scales'].to(device)
    rope2d_freqs_grid['freqs_frames'] = rope2d_freqs_grid['freqs_frames'].to(device)
    rope2d_freqs_grid['freqs_height'] = rope2d_freqs_grid['freqs_height'].to(device)
    rope2d_freqs_grid['freqs_width'] = rope2d_freqs_grid['freqs_width'].to(device)
    _, uph, upw = largest_scale
    pt, ph, pw = current_scale
    dim_div_2_div_4 = rope2d_freqs_grid['freqs_scales'].shape[2]
    dim_div_2 = dim_div_2_div_4 * 4
    f_scales = rope2d_freqs_grid['freqs_scales'][:, real_sid].reshape(2, 1, dim_div_2_div_4)
    f_frames = rope2d_freqs_grid['freqs_frames'][:, t_list]
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
