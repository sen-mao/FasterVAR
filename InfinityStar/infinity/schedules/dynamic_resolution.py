# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import json
import math
import copy

import tqdm
import numpy as np

vae_stride = 16
ratios = [1.000, 1.250, 1.333, 1.500, 1.750, 2.000, 2.500, 3.000]

def get_ratio2hws_video_v2():
    ratio2hws_video_common_v2 = {}
    for h_div_w in [1, 100/116, 3/4, 2/3, 9/16, 1/2, 2/5, 1/3]:
        scale_schedule = []
        # 40*40 is 480p, 60*60 is 720p
        # for scale in list(range(1,1+16)) + [20, 24, 30, 40]:
        for scale in [1,2,3,4,5,6,7,8,10,12,16] + [24, 32, 40, 60]:
            area = scale * scale
            pw_float = math.sqrt(area / h_div_w)
            ph_float = pw_float * h_div_w
            ph, pw = int(np.round(ph_float)), int(np.round(pw_float))
            scale_schedule.append((ph, pw))
        ratio2hws_video_common_v2[h_div_w] = scale_schedule
    total_pixels2scales = {
        '0.06M': 11,
        '0.25M': 13,
        '0.40M': 14,
        '0.90M': 15,
    }
    return ratio2hws_video_common_v2, total_pixels2scales

def append_dummy_t(ratio2hws):
    for key in ratio2hws:
        for i in range(len(ratio2hws[key])):
            h, w = ratio2hws[key][i]
            ratio2hws[key][i] = (1, h, w)
    return ratio2hws

def get_first_full_spatial_size_scale_index(vae_scale_schedule):
    for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
        if vae_scale_schedule[si][-2:] == vae_scale_schedule[-1][-2:]:
            return si

def get_full_spatial_size_scale_indices(vae_scale_schedule):
    full_spatial_size_scale_indices = []
    for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
        if vae_scale_schedule[si][-2:] == vae_scale_schedule[-1][-2:]:
            full_spatial_size_scale_indices.append(si)
    return full_spatial_size_scale_indices

def repeat_schedule(scale_schedule, repeat_scales_num, times):
    new_scale_schedule = []
    for i in range(repeat_scales_num):
        new_scale_schedule.extend([scale_schedule[i] for _ in range(times)])
    new_scale_schedule.extend(scale_schedule[repeat_scales_num:])
    return new_scale_schedule

def get_ratio2hws_pixels2scales(dynamic_scale_schedule, video_frames):
    compressed_frames = video_frames // 4 + 1
    if dynamic_scale_schedule == '13_hand_craft':
        ratio2hws = {
            1.000: [(1,1),(2,2),(4,4),(6,6),(8,8),(12,12),(16,16),(20,20),(24,24),(32,32),(40,40),(48,48),(64,64)],
            1.250: [(1,1),(2,2),(3,3),(5,4),(10,8),(15,12),(20,16),(25,20),(30,24),(35,28),(45,36),(55,44),(70,56)],
            1.333: [(1,1),(2,2),(4,3),(8,6),(12,9),(16,12),(20,15),(24,18),(28,21),(36,27),(48,36),(60,45),(72,54)],
            1.500: [(1,1),(2,2),(3,2),(6,4),(9,6),(15,10),(21,14),(27,18),(33,22),(39,26),(48,32),(63,42),(78,52)],
            1.750: [(1,1),(2,2),(3,3),(7,4),(11,6),(14,8),(21,12),(28,16),(35,20),(42,24),(56,32),(70,40),(84,48)],
            2.000: [(1,1),(2,2),(4,2),(6,3),(10,5),(16,8),(22,11),(30,15),(38,19),(46,23),(60,30),(74,37),(90,45)],
            2.500: [(1,1),(2,2),(5,2),(10,4),(15,6),(20,8),(25,10),(30,12),(40,16),(50,20),(65,26),(80,32),(100,40)],
            3.000: [(1,1),(2,2),(6,2),(9,3),(15,5),(21,7),(27,9),(36,12),(45,15),(54,18),(72,24),(90,30),(111,37)],
        }
        ratio2hws = append_dummy_t(ratio2hws)
        total_pixels2scales = {
            '0.06M': 7,
            '0.25M': 10,
            '1M': 13,
        }
        predefined_t = [1 for _ in range(len(ratio2hws[1.000]))]
        dynamic_resolution_h_w = get_full_ratio2hws(ratio2hws, video_frames, total_pixels2scales, predefined_t)
        for ratio in dynamic_resolution_h_w:
            for pn in dynamic_resolution_h_w[ratio]:
                base_scale_schedule = dynamic_resolution_h_w[ratio][pn]['scales']
                ts = np.round(np.linspace(1,compressed_frames,7))
                dynamic_resolution_h_w[ratio][pn]['image_scales'] = base_scale_schedule
                if dynamic_scale_schedule == 'infinity_loop_full_time':
                    dynamic_resolution_h_w[ratio][pn]['video_scales'] = [(compressed_frames, pn[1], pn[2]) for pn in base_scale_schedule]
                else:
                    dynamic_resolution_h_w[ratio][pn]['video_scales'] = [(int(t), pn[1], pn[2]) for (t, pn) in zip(ts, base_scale_schedule)]
                del dynamic_resolution_h_w[ratio][pn]['scales']
    elif dynamic_scale_schedule in ['infinity_elegant_clip20frames_v2', 'infinity_star_interact']:
        ratio2hws, total_pixels2scales = get_ratio2hws_video_v2()
        ratio2hws = append_dummy_t(ratio2hws)
        dynamic_resolution_h_w = get_full_ratio2hws(ratio2hws, video_frames, total_pixels2scales, predefined_t=None)
        compressed_frames_in_one_clip = 20
        compressed_frames_per_sec = 16 // 4
        duration_resolution = 1
        for ratio in dynamic_resolution_h_w:
            for pn in dynamic_resolution_h_w[ratio]:
                base_scale_schedule = dynamic_resolution_h_w[ratio][pn]['scales']
                image_scale_schedule = base_scale_schedule
                spatial_time_schedule = []
                spatial_time_schedule.extend(image_scale_schedule)
                assert (compressed_frames - 1) % compressed_frames_in_one_clip == 0
                clips = (compressed_frames - 1) // compressed_frames_in_one_clip
                scales_in_one_clip = len(base_scale_schedule)
                for _ in range(clips):
                    spatial_time_schedule.extend([(compressed_frames_in_one_clip, h, w) for _, h, w in base_scale_schedule])
                dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'] = {1: image_scale_schedule}
                pt_interval = duration_resolution*compressed_frames_per_sec
                for pt in range(1+compressed_frames_per_sec,compressed_frames+1, pt_interval): # duration_resolution is 1s
                    tmp_clips = 1 + int(np.ceil((pt-1) / compressed_frames_in_one_clip))
                    dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'][pt] = spatial_time_schedule[:scales_in_one_clip*tmp_clips]
                    pt_last_clip = (pt - 1) % compressed_frames_in_one_clip
                    if pt_last_clip > 0:
                        for i in range(scales_in_one_clip):
                            tmp_t, tmp_h, tmp_w = dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'][pt][-i-1]
                            dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'][pt][-i-1] = (pt_last_clip, tmp_h, tmp_w)
                dynamic_resolution_h_w[ratio][pn]['image_scales'] = scales_in_one_clip
                dynamic_resolution_h_w[ratio][pn]['scales_in_one_clip'] = scales_in_one_clip
                dynamic_resolution_h_w[ratio][pn]['max_video_scales'] = len(dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'][compressed_frames])
                del dynamic_resolution_h_w[ratio][pn]['scales']
    elif dynamic_scale_schedule == 'infinity_star_extract_features':
        ratio2hws, total_pixels2scales = get_ratio2hws_video_v2()
        ratio2hws = append_dummy_t(ratio2hws)
        dynamic_resolution_h_w = get_full_ratio2hws(ratio2hws, video_frames, total_pixels2scales, predefined_t=None)
        for ratio in dynamic_resolution_h_w:
            for pn in dynamic_resolution_h_w[ratio]:
                base_scale_schedule = dynamic_resolution_h_w[ratio][pn]['scales']
                image_scale_schedule = base_scale_schedule
                spatial_time_schedule = []
                spatial_time_schedule.extend(image_scale_schedule)
                clips = compressed_frames - 1
                dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'] = {}
                for pt in range(1,compressed_frames+1, 1): # duration_resolution is 1s
                    dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'][pt] = [(pt, h, w) for _, h, w in base_scale_schedule]
                dynamic_resolution_h_w[ratio][pn]['image_scales'] = len(base_scale_schedule)
                dynamic_resolution_h_w[ratio][pn]['scales_in_one_clip'] = len(base_scale_schedule)
                dynamic_resolution_h_w[ratio][pn]['max_video_scales'] = len(base_scale_schedule)
                del dynamic_resolution_h_w[ratio][pn]['scales']
    else:
        raise ValueError(f'dynamic_scale_schedule={dynamic_scale_schedule} not implemented')
    return dynamic_resolution_h_w


def get_full_ratio2hws(ratio2hws, video_frames, total_pixels2scales, predefined_t=None):
    compressed_frames = video_frames//4+1
    if predefined_t and predefined_t != 'auto':
        refined_predefined_t = [min(t, compressed_frames) for t in predefined_t]
    full_ratio2hws = {}
    for ratio, hws in ratio2hws.items():
        real_ratio = hws[-1][1] / hws[-1][2]
        full_ratio2hws[int(real_ratio*1000)/1000] = hws
        if ratio != 1.000:
            full_ratio2hws[int(1/real_ratio*1000)/1000] = [(item[0], item[2], item[1]) for item in hws]

    dynamic_resolution_h_w = {}
    for ratio in full_ratio2hws:
        dynamic_resolution_h_w[ratio] = {}
        for total_pixels, scales_num in total_pixels2scales.items():
            pixel = (full_ratio2hws[ratio][scales_num-1][1] * vae_stride, full_ratio2hws[ratio][scales_num-1][2] * vae_stride)
            scales = full_ratio2hws[ratio][:scales_num]
            if predefined_t and predefined_t != 'auto':
                scales = [ (t, h, w) for t, (_, h, w) in zip(refined_predefined_t, scales) ]
            elif predefined_t == 'auto':
                refined_predefined_t = np.linspace(1, compressed_frames, scales_num).astype(int)
                scales = [ (t, h, w) for t, (_, h, w) in zip(refined_predefined_t, scales) ]
            dynamic_resolution_h_w[ratio][total_pixels] = {
                'pixel': pixel,
                'scales': scales
            }
    return dynamic_resolution_h_w

def get_dynamic_resolution_meta(dynamic_scale_schedule, video_frames=1000):
    dynamic_resolution_h_w = get_ratio2hws_pixels2scales(dynamic_scale_schedule, video_frames)
    h_div_w_templates = []
    for h_div_w in dynamic_resolution_h_w.keys():
        h_div_w_templates.append(h_div_w)
    h_div_w_templates = np.array(h_div_w_templates)
    return dynamic_resolution_h_w, h_div_w_templates

def get_h_div_w_template2indices(h_div_w_list, h_div_w_templates):
    indices = list(range(len(h_div_w_list)))
    h_div_w_template2indices = {}
    pbar = tqdm.tqdm(total=len(indices), desc='get_h_div_w_template2indices...')
    for h_div_w, index in zip(h_div_w_list, indices):
        pbar.update(1)
        nearest_h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        if nearest_h_div_w_template_ not in h_div_w_template2indices:
            h_div_w_template2indices[nearest_h_div_w_template_] = []
        h_div_w_template2indices[nearest_h_div_w_template_].append(index)
    for h_div_w_template_, sub_indices in h_div_w_template2indices.items():
        h_div_w_template2indices[h_div_w_template_] = np.array(sub_indices)
    return h_div_w_template2indices

def get_activated_h_div_w_templates(h_div_w_list, h_div_w_templates):
    if h_div_w_list is None:
        activated_h_div_w_templates = h_div_w_templates
    else:
        activated_h_div_w_templates = []
        h_div_w_templates = np.array(h_div_w_templates)
        for h_div_w in h_div_w_list:
            index = np.argmin(np.abs(h_div_w - h_div_w_templates))
            activated_h_div_w_templates.append(h_div_w_templates[index])
    activated_h_div_w_templates = sorted(list(set(activated_h_div_w_templates)))
    return activated_h_div_w_templates

if __name__ == '__main__':
    video_frames = 81
    dynamic_resolution_h_w = get_ratio2hws_pixels2scales('infinity_elegant_clip20frames_v2', video_frames)
    for h_div_w in dynamic_resolution_h_w:
        if h_div_w >= 1:
            for pn in ['0.25M']:
                print(h_div_w, pn, np.array(dynamic_resolution_h_w[h_div_w][pn]['pt2scale_schedule'][1]).prod(-1).sum())
    
    import pdb; pdb.set_trace()
