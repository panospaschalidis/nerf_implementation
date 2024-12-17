"""standard libraries"""
import os
import sys
import glob
import torch
import pandas as pd
import numpy as np
import imageio.v2
import pdb
import random 
"""third party imports"""
from torch.utils.data import Dataset

"""local specific imports"""

path = os.path.join(os.getcwd(), 'tiny_nerf_data.npz')
all_ = np.load(path)
seq_of_views = np.arange(len(all_['images']))
random.shuffle(seq_of_views)
random_seq = seq_of_views
seq = {
'train': random_seq[0:int(round(len(random_seq)*0.75))],
'val': random_seq[int(round(len(random_seq)*0.75)):-10],
'test': random_seq[-10:]
}

os.mkdir(os.path.join(os.getcwd(), 'lego_dataset'))
paths = {
'train': os.path.join(os.getcwd(), 'lego_dataset', 'train'),
'val': os.path.join(os.getcwd(), 'lego_dataset', 'val'),
'test': os.path.join(os.getcwd(), 'lego_dataset', 'test')
}

for key in seq:
    os.mkdir(paths[key])
    images = all_['images'][seq[key]]
    poses = all_['poses'][seq[key]]
    focal = all_['focal']
    np.savez_compressed(
        os.path.join(paths[key],os.path.basename(paths[key])),
        images=images,
        poses=poses,
        focal=focal
    )
