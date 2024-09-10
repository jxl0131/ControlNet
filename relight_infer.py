import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from share import *
import config
# safetensors               0.2.7
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
# from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

model = create_model('./models/relight_v21.yaml').cpu()
model.load_state_dict(load_state_dict('./lightning_logs/version_27/checkpoints/epoch=106-step=93624.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution,detect_resolution,  ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        # input_image = HWC3(input_image)
        # detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
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

if __name__ == '__main__':

    from PIL import Image

    test_image = Image.open('source_rgba.png')
    test_image = np.array(test_image)
    guess_mode =True #无提示词，有控制图，模型从控制图中猜测语义
    strength = 2.0 #控制图引导的强度
    scale = 9.0 #无提示词引导的强度
    results = process(test_image, 'portrait with harmonious and natural lighting', 'best quality, extremely detailed', 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', 1, 512, 512, 20, guess_mode, strength, scale, -1, 0.0)
    Image.fromarray(results[1]).save('epoch=106-step=93624-test_result_cfgw.png')

    guess_mode =False #有提示词，有控制图，模型从控制图中猜测语义
    results = process(test_image, 'portrait with harmonious and natural lighting', 'best quality, extremely detailed', 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', 1, 512, 512, 20, guess_mode, strength, scale, -1, 0.0)
    Image.fromarray(results[1]).save('epoch=106-step=93624-test_result.png')

