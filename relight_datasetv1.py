import json,os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

dataset_dir = "/data/jixinlong/jixinlong/datasets/relight2/"


class MyDataset(Dataset):
    def __init__(self):
        nms = os.listdir(dataset_dir+'relight')#[:20]
        self.data = nms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nm = self.data[idx]

        # In the training process, we randomly replace 50% text prompts ct with empty strings. This approach increases ControlNet’s ability to directly recognize semantics in the input conditioning images (e.g., edges, poses, depth, etc.) as a replacement for the prompt.
        # 随机替换50%的文本提示ct为空字符串。这种方法增加了ControlNet直接识别输入条件图像中的语义（例如边缘、姿势、深度等）作为提示的能力。
        ramdom = np.random.rand()
        if ramdom > 0.5:
            prompt = ''
        else:
            prompt = 'portrait with harmonious and natural lighting'

        # mask = cv2.imread(dataset_dir+'mask/'+nm, cv2.IMREAD_GRAYSCALE)
        mask_path = dataset_dir.replace('relight2','relight')+'mask/'+nm[:-6]+'.jpg'
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=2)/255.0

        ori_path = mask_path.replace('mask','ori')
        source = cv2.imread(dataset_dir+'relight/'+nm) * mask +  cv2.imread(ori_path) * (1-mask)
        source = source.astype(np.uint8)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        
        # 给source加上mask通道       
        source = np.concatenate([source, (mask*255).astype(np.uint8)], axis=2)

        target = cv2.imread(ori_path)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # 短边缩放到512，长边等比例缩放，然后裁剪中心512*512
        h, w = source.shape[:2]
        if h > w:
            new_w = 512
            new_h = int(h * 512 / w)
        else:
            new_h = 512
            new_w = int(w * 512 / h)
        source = cv2.resize(source, (new_w, new_h))
        target = cv2.resize(target, (new_w, new_h))
        h, w = source.shape[:2]
        source = source[(h-512)//2:(h+512)//2, (w-512)//2:(w+512)//2]
        target = target[(h-512)//2:(h+512)//2, (w-512)//2:(w+512)//2]
        # Image.fromarray(source[...,:3]).save('source1.png')
        # Image.fromarray(source).save('source_rgba1.png')
        # Image.fromarray(target).save('target1.png')

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

