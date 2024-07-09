import json,os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

dataset_dir = "/data/jixinlong/jixinlong/datasets/relight/"


class MyDataset(Dataset):
    def __init__(self):
        # self.data = []
        # with open('/data/jixinlong/jixinlong/datasets/fill50k/prompt.json', 'rt') as f:
        #     for line in f:
        #         self.data.append(json.loads(line))
        nms = os.listdir(dataset_dir+'ori')#[:20]
        self.data = nms
        # for nm in nms:

            
        #     self.data.append({'source': source, 'target': target})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nm = self.data[idx]

        prompt = 'portrait with harmonious and natural lighting'

        fg = cv2.imread(dataset_dir+'fg/'+nm)
        mask = np.zeros(fg.shape[:2], dtype=np.uint8)
        mask [(fg != fg[0,0,:]).all(axis=2)] = 1
        mask = np.expand_dims(mask, axis=2)

        source = cv2.imread(dataset_dir+'relight/'+nm) * mask +  cv2.imread(dataset_dir+'ori/'+nm) * (1-mask)
        # 给source加上mask
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # Image.fromarray(source).save('source.png')
        
        source = np.concatenate([source, mask*255], axis=2)
        # Image.fromarray(source).save('source_rgba.png')
        
        target = cv2.imread(dataset_dir+'ori/'+nm)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # source和target尺寸调整为一半
        new_size = (512, 512)
        source = cv2.resize(source, new_size)
        target = cv2.resize(target, new_size)
        # Image.fromarray(target).save('target.png')

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

