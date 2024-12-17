"""standard libraries"""
import os
import sys
import glob
import torch
import pandas as pd
import numpy as np
import pdb
import random
import matplotlib.pyplot as plt

"""third party imports"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
 
"""local specific imports"""
from data import LegoDataset
from train.utils import reconstruction

model_path = '/home/panagiotis/Downloads/fine_checkpoint.pth'
yaml_path = '/home/panagiotis/workstation/nerf_workspace/combined_2nd_version/conf/nerf_config.yaml'
params = yaml.full_load(open(yaml_path, 'r')).get('dict')
pretrained_models = torch.load(model_path)['models']
data_conf = yaml.full_load(open(params['data']['conf'], 'r')).get(params['data']['desgn'])
device = 'cpu'
dataset = LegoDataset(
    data_conf['path'], 
    ToTensor(), 
    data_conf['ray_conf'], 
    data_conf['hier_samp_conf'], 
    device
)
dataloader = DataLoader(
    dataset,
    batch_size=params['batch_size']['test'],
    shuffle=True,
    drop_last=True
)
media = reconstruction(
    pretrained_models,
    dataloader,
    params['inv_tr_conf'],
    params['net_conf']['model_info'],
    device
)
fig, ax = plt.subplots(1, 3)
ax[0,0].imshow(media['target'])
ax[0,0].axis('off')
ax[0,1].imshow(media['coarse_im'])
ax[0,1].axis('off')
ax[0,2].imshow(media['fine_im'])
ax[0,2].axis('off')
ax[0,0].set_title(r'TARGET')
ax[0,1].set_title(r'COARSE')
ax[0,2].set_title(r'FINE')
plt.show()






