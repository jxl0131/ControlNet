import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import math
import gradio as gr
import numpy as np
import torch

from PIL import Image
from briarmbg import BriaRMBG

rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
device = torch.device('cuda')
rmbg = rmbg.to(device=device, dtype=torch.float32)

from share import *
import config

import cv2
import einops
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

model = create_model('./models/relight_v21.yaml').cpu()
model.load_state_dict(load_state_dict('./lightning_logs/version_30/checkpoints/epoch=11-step=83999.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():

        detected_map = resize_image(input_image, image_resolution)
        H, W, C = detected_map.shape
        print(C)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        # 如果是guess mode，需要将un_cond中的控制图设置为None，这样才能让模型从控制图中猜测语义
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}#guess_mode表示没有提示词，这时候必须是cond有控制图且un_cond没有控制图
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        # 如果是guess mode，需要将control_scales设置为递减的，这样才能避免控制图中的语义太强
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results



@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha

@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise):
    print(1111)
    input_fg, matting = run_rmbg(input_fg)
    print(22222)
    # 尺寸调整
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    mask = resize_and_center_crop(matting[...,0], image_width, image_height) # clip(0, 1) float32
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    
    mask = np.expand_dims(mask, axis=2)
    input =( fg * mask +  bg * (1-mask)).astype(np.uint8)
    input = np.concatenate([input, (mask*255).astype(np.uint8)], axis=2)# 给test_image加上mask通道
    Image.fromarray(input).save('input.png')
    
    print(3333)
    strength= 1.0
    eta = 0.0
    results_1 = process(input, prompt, a_prompt, n_prompt, num_samples, image_width,  steps, guess_mode, strength, cfg, seed, eta)
    guess_mode = False #无提示词，有控制图，模型从控制图中猜测语义
    
    results_2 = process(input, prompt, a_prompt, n_prompt, num_samples, image_width, steps, guess_mode, strength, cfg, seed, eta)
    extra_images = [fg, bg,results_1[1],results_2[1]]

    return extra_images



block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## IC-Light (Relighting with Foreground and Background Condition)")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(source='upload', type="numpy", label="Foreground")
                input_bg = gr.Image(source='upload', type="numpy", label="Background")
            prompt = gr.Textbox(label="Prompt")
            
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)
                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=512, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=7.0, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=0.9, value=0.5, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='lowres, bad anatomy, bad hands, cropped, worst quality')
        with gr.Column():
            # result_gallery = gr.Gallery(  label='Outputs')
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')


    ips = [input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise]
    relight_button.click(fn=process_relight, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
