# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

def get_encode_decode_func(dynamic_scale_schedule):
    if dynamic_scale_schedule == 'infinity_star_interact':
        from infinity.schedules.infinity_star_interact import video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info
    elif 'infinity_elegant' in dynamic_scale_schedule:
        from infinity.schedules.infinity_elegant import video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info
    else:
        raise NotImplementedError(f'dynamic_scale_schedule not implemented: {dynamic_scale_schedule}')
    return video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info
