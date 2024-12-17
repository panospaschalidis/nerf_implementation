"""standard libraries"""
import argparse
import yaml
import os
import sys
import glob
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import pdb
import wandb

"""third party imports"""
from torch.utils.data import DataLoader
from simple_3dviz import Scene
from simple_3dviz import Mesh, Spherecloud
from simple_3dviz.utils import save_frame

"""local specific imports"""
from data import data_splitter
from train.trainer import (NeRFTrainer, PixelNeRFTrainer, 
                            UniSURFTrainer, ImageSURFTrainer)

path = '/home/panagiotis/workstation/nerf_workspace/combined_version/conf/nerf_config.yaml'
params = yaml.full_load(open(path, 'r')).get('nerf_dict')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#dataset = LegoDataset(
#    params['data_conf'],
#    ToTensor(),
#    params['ray_conf'],
#    params['hier_samp_conf'],
#    device
#)

train_dataset, valid_dataset, test_dataset = data_splitter(
    os.path.join(os.getcwd(),params['data']['conf']),
    params['data']['desgn'],
    device
)

x = np.linspace(-0.5, 0.5, num =5)
centers = np.array(np.meshgrid(x, x, x)).reshape(3, -1).T
def scatter3D(centers):
    # centers shouyld be in shape number_of_centersx3
    scene = Scene(background=(0.0, 0.0, 0.0, 1.0), size=(512, 512))
    colors = np.array([1, 1, 0, 1])
    sizes = np.ones(centers.shape[0])*0.02
    s = Spherecloud(centers, colors, sizes)
    scene.add(s)
    scene.camera_position = (2, 2, -7)
    scene.render()
    save_frame("scene_2.png", scene.frame)
world_cords = train_dataset[0]['samples'][0,:,:]
camera_cords = (world_cords @  train_dataset[0]['w2cpose'][:3,:3].T) + train_dataset[0]['w2cpose'][:3,3]
# from camera space to raster space
raster_cords = -camera_cords[...,:2]/camera_cords[...,2:]
#raster_cords[...,0] = raster_cords[...,0] * self.focal[:,None,None]
#raster_cords[...,1] = raster_cords[...,1] *(- self.focal[:,None,None])
#raster_cords = raster_cords + self.prcpal_p[:,None,None,:]


