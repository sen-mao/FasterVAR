# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
import sys
import json
import os
import os.path as osp
from tqdm import tqdm
import sys
import time
import numpy as np
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, gen_one_example, save_video, transform
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.arg_util import Args


def _init_prompt_rewriter():
    from tools.prompt_rewriter import OpenAIGPTModel
    """Initialize the OpenAI GPT model."""
    # Initialize the OpenAI GPT model
    model_name = 'gpt-4o-2024-08-06'
    ak = os.environ.get("OPEN_API_KEY", "")
    if len(ak) == 0:
        raise ValueError("Please provide your OpenAI API key in the OPEN_API_KEY environment variable.")
    model = OpenAIGPTModel(model_name, ak, if_global=True)
    system_prompt = (
        "You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description to make the video more realistic and beautiful. 0. Preserve ALL information, including style words and technical terms. 1. If the subject is related to person, you need to provide a detailed description focusing on basic visual characteristics of the person, such as appearance, clothing, expression, posture, etc. You need to make the person as beautiful and handsome as possible. When the subject is only one person or object, do not use they to describe him/her/it to avoid confusion with multiple subjects. 2. If the input does not include style, lighting, atmosphere, you can make reasonable associations. 3. We only generate a four-second video based on your descriptions. So do not generate descriptions that are too long, too complex or contain too many activities. 4. You can add some descriptions of camera movements with regards to the scenes and allow the scenes to have very natural and coherent movements. 6. If the input is in Chinese, translate the entire description to English. 7. Output ALL must be in English. 8. Here are some expanded descriptions that can serve as examples: 1. The video begins with a distant aerial view of a winding river cutting through a rocky landscape, with the sun casting a soft glow over the scene. As the camera moves closer, the river's flow becomes more visible, and the surrounding terrain appears more defined. The camera continues to approach, revealing a steep cliff with a person sitting on its edge. The person is positioned near the top of the cliff, overlooking the river below. The camera finally reaches a close-up view, showing the person sitting calmly on the cliff, with the river and landscape fully visible in the background. 2. In a laboratory setting, a machine with a metallic structure and a green platform is seen. A small, clear plastic bottle is positioned on the green platform. The machine has a control panel with red and green lights on the right side. A nozzle is positioned above the bottle, and it begins to dispense liquid into the bottle. The liquid is dispensed in small droplets, and the nozzle moves slightly between each droplet. The background includes other laboratory equipment and a mesh-like structure. 3. The video shows a panoramic view of a cityscape with a prominent building featuring a green dome and ornate architecture in the center. Surrounding the main building are several other structures, including a white building with balconies on the left and a taller building with multiple windows on the right. In the background, there are hills with scattered buildings and greenery. The camera remains stationary, capturing the scene from a fixed position, with no noticeable changes in the environment or the buildings throughout the frames. 4. In a dimly lit room with red and blue lighting, a person holds up a smartphone to record a video of a band performing. The band members are seated, with one holding a guitar and another playing a double bass. The smartphone screen shows the band members being recorded, with the camera capturing their movements and expressions. The background includes a lamp and some furniture, adding to the cozy atmosphere of the scene. 5. In a grassy area with scattered trees, a large tree stands prominently in the center. A lion is perched on a thick branch of this tree, looking out into the distance. The sky is overcast, adding a somber tone to the scene. 6. A man in a green sweater holding a paper turns around and speaks to a group of people seated in a theater. He then points at a man in a yellow sweater sitting in the front row. The man in the yellow sweater looks at the paper in his hand and begins to speak. The man in the green sweater lowers his head and then looks up at the man in the yellow sweater again. 7. An elderly man, wearing a beige sweater over a yellow shirt, is sitting in front of a laptop. He holds a pair of glasses in his right hand and appears to be deep in thought, resting his head on his hand. He then raises the glasses and rubs his eyes with his fingers, showing signs of fatigue. After rubbing his eyes, he places the glasses on his sweater and looks down at the laptop screen. 8. A woman and a child are sitting at a table, each holding a pencil and coloring on a piece of paper. The woman is coloring a green leafy plant, while the child is coloring a red and blue object. The table has several colored pencils, a container filled with more pencils, and a few small colorful blocks. The woman is wearing a striped shirt, and the child is focused on their drawing. 9. A person wearing teal running shoes and colorful socks is running on a wet, sandy surface. The camera captures the movement of their legs and feet as they lift off the ground and land back, creating a clear shadow on the wet sand. The shadow elongates and shifts with each step, indicating the person's motion. The background remains consistent with the wet, textured sand, and the focus is solely on the runner's feet and their shadow. 10. A man is running along the shoreline of a beach, with the ocean waves gently crashing onto the shore. The sun is setting in the background, casting a warm glow over the scene. The man is wearing a light-colored jacket and shorts, and his hair is blowing in the wind as he runs. The water splashes around his legs as he moves forward, and his reflection is visible on the wet sand. The waves create a dynamic and lively atmosphere as they roll in and out."
    )
    gpt_model = OpenAIGPTModel(model_name, ak, if_global=True)
    return gpt_model, system_prompt

class InferencePipe:
    def __init__(self, args):
        # load text encoder
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        self.vae = load_visual_tokenizer(args)
        self.vae = self.vae.float().to('cuda')
        # load infinity
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        
        self._models = [self.text_tokenizer, self.text_encoder, self.vae, self.infinity, self.self_correction]

        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)

        if args.enable_rewriter:
            self.gpt_model, self.system_prompt = _init_prompt_rewriter()   


def perform_inference(pipe, data, args):
    
    prompt = data["prompt"]
    seed = data["seed"]
    mapped_duration=5
    num_frames=81

    # If an image_path is provided, perform image-to-video generation.
    image_path = data.get("image_path", None)

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)    
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    gt_leak, gt_ls_Bl = -1, None

    if image_path is not None:
        ref_image = [cv2.imread(image_path)[:,:,::-1]]
        ref_img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in ref_image]
        ref_img_T3HW = torch.stack(ref_img_T3HW, 0) # [t,3,h,w]
        ref_img_bcthw = ref_img_T3HW.permute(1,0,2,3).unsqueeze(0) # [c,t,h,w] -> [b,c,t,h,w]
        _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(pipe.vae, ref_img_bcthw.cuda(), vae_features=None, self_correction=pipe.self_correction, args=args, infer_mode=True, dynamic_resolution_h_w=dynamic_resolution_h_w)
        gt_leak=len(scale_schedule)//2

    generated_image_list = []
    negative_prompt=''
    prompt = f'{prompt}, Close-up on big objects, emphasize scale and detail'
    negative_prompt = ""
    if args.append_duration2caption:
        prompt = f'<<<t={mapped_duration}s>>>' + prompt
    
    start_time = time.time()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        generated_image, _ = gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt,
            negative_prompt=negative_prompt,
            g_seed=seed,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            cfg_list=args.cfg, 
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],
            vae_type=args.vae_type,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
        )
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)
        print(generated_image.shape)
        generated_image_list.append(generated_image)
            
    generated_image = torch.cat(generated_image_list, 2)
    end_time = time.time()
    elapsed_time = end_time - start_time    
    
    return {
            "output": generated_image.cpu().numpy(),
            "elapsed_time": elapsed_time,
        }


if __name__ == '__main__':
    # For optimal performance, enabling the prompt rewriter is recommended.
    # To utilize the GPT model, ensure the following environment variables are set:
    # export OPEN_API_KEY="YOUR_API_KEY"
    # export GLOBAL_AZURE_ENDPOINT="YOUR_ENDPOINT"
    enable_rewriter=0
    checkpoints_dir = './'
    

    # infer args
    args = Args()
    args.pn='0.90M'
    args.fps=16
    args.video_frames=81
    args.model_path=os.path.join(checkpoints_dir, 'infinitystar_8b_720p_weights')
    args.checkpoint_type='torch_shard' # omnistore
    args.vae_path=os.path.join(checkpoints_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt=os.path.join(checkpoints_dir, 'text_encoder/flan-t5-xl-official/')
    args.model_type='infinity_qwen8b'
    args.text_channels=2048
    args.dynamic_scale_schedule='infinity_elegant_clip20frames_v2'
    args.bf16=1
    args.use_apg=1
    args.use_cfg=0
    args.cfg=34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold=0.05
    args.image_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    args.video_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]'
    args.append_duration2caption=1
    args.use_two_stage_lfq=1
    args.detail_scale_min_tokens=750
    args.semantic_scales=12
    args.max_repeat_times=10000
    args.enable_rewriter=enable_rewriter

    # load models
    pipe = InferencePipe(args)

    
    prompt = "A handsome smiling gardener inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic"
    image_path = 'assets/reference_image.webp'  # Remove this for Text-to-Video (T2V) generation
    data = {
        'seed': 41,
        'image_path': image_path,
        'prompt': prompt,
    }
    if args.enable_rewriter:
        # Step 1: Rewrite the prompt using GPT
        # rewritten_prompt = prompt
        rewritten_prompt = pipe.gpt_model(
            prompt = ("Rewrite the following video descriptions, add more details of the subject and the camera movement to enhance the quality of the video. Do not use the word 'they' to refer to a single person or object. Concatenate all sentences together, not present them in paragraphs. Please rewrite with concise and clear language: "
                    + prompt),
            system_prompt=pipe.system_prompt,
        )
        print(f"Rewritten prompt: {rewritten_prompt}")
        prompt = rewritten_prompt
    data['prompt'] = prompt
                
    output_dict = perform_inference(pipe, data, args)
    save_dir = 'output'
    gen_video_path = osp.join(os.path.join(save_dir, 'gen_videos'), f'demo.mp4')
    save_video(output_dict['output'], fps=args.fps, save_filepath=gen_video_path)
            
    print(f"Video genernation done: {gen_video_path=}")
