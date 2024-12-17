"""data module contains Dataset and Dataloader classes"""

"""standard libraries"""
import yaml
import os
import pdb 
import torch

"""third party imports"""
from torch.utils.data import random_split
from torchvision.transforms import ToTensor

"""local specific imports"""
from .datasets import LegoDataset, SRNDataset, NMRDataset, DTUDataset
from .utils import ray_tracing, strf_sampling


def data_splitter(conf, desgn):
    # conf corresponds to dataset_conf.yaml path where as 
    # desgn to dataset designator(Lego, SRN, MVR, DTU etc)
    data_conf = yaml.full_load(open(conf, 'r')).get(desgn)
    if desgn == 'Lego':
        dset = LegoDataset
    elif desgn == 'SRN':
        dset = SRNDataset
    elif desgn == 'NMR':
        dset = NMRDataset
    elif desgn == 'DTU':
        dset = DTUDataset
    else:
     raise NotImplementedError
    
    train_set = dset(
        data_conf['path'] + 'train', 
        data_conf['input_conf'],
        'train'
    )
    val_set = dset(
        data_conf['path'] + 'val',
        data_conf['input_conf'],
        'val'
    )
    test_set = dset(
        data_conf['path'] + 'test',
        data_conf['input_conf'], 
        'test'
    )
    return train_set, val_set, test_set

