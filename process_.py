import json,os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

dataset_dir = "/data/jixinlong/jixinlong/datasets/relight/"

image_width = 768
image_height = 960
ct =0 
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
# 读取/data/jixinlong/jixinlong/datasets/easyportrait/train中所有图片，用resize_and_center_crop处理后放入/data/jixinlong/jixinlong/datasets/relight/ori

for filename in os.listdir('/data/jixinlong/jixinlong/datasets/easyportrait/train'):
    image = cv2.imread('/data/jixinlong/jixinlong/datasets/easyportrait/train/' + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_and_center_crop(image, image_width, image_height)
    cv2.imwrite(dataset_dir + 'ori/' + filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    ct +=1
    print(ct)
