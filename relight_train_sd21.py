import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from relight_datasetv1 import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = '/data/jixinlong/jixinlong/ControlNet/lightning_logs/version_27/checkpoints/epoch=122-step=107624.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
accumulate_grad_batches=4

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/relight_v21.yaml').cpu()
state_dict = load_state_dict(resume_path, location='cpu')
# 修改权重字典中的卷积层权重形状，并将新增加的权重初始化为0
# import torch
# old_conv_weight = state_dict['control_model.input_hint_block.0.weight']
# old_conv_weight = torch.cat([old_conv_weight, torch.zeros((16,1,3,3))], dim=1)
# state_dict['control_model.input_hint_block.0.weight'] = old_conv_weight
# state_dict['control_model.input_hint_block.0.weight'] = torch.zeros((16,4,3,3))
# control_model.input_hint_block.0.weight
model.load_state_dict(state_dict, strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32,callbacks=[logger],accumulate_grad_batches=accumulate_grad_batches)
# , strategy="ddp_sharded"

# Train!
trainer.fit(model, dataloader)
