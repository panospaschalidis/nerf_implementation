"""standard libraries"""
import argparse
import yaml
import os
import sys
import glob
import torch
import numpy as np
import torch.nn as nn
import pdb
import wandb

"""third party imports"""
from torch.utils.data import DataLoader

"""local specific imports"""
from data import data_splitter
from train.trainer import Trainer
                           
parser = argparse.ArgumentParser(
    description = "Radiance Fields"
)
parser.add_argument(
    "--conf",
    default=None,
    help="path of configuration file"
)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="checkpoint path for training resume or model evaluation"
)
print(f"In case you run pixelnerf replace ti_fine with ti_refine at \
line 227 in data/utils.py -> def inverse_transform_sampling")

args = parser.parse_args()
if not args.checkpoint:
    wandb.init(project='nerf')
    os.environ["WANDB_RUN_ID"] = wandb.run.name
else:
    project_name = os.path.splitext(os.path.splitext(os.listdir(args.checkpoint)[0])[0])[0]
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = project_name
    wandb.init(project='nerf')

params = yaml.full_load(open(args.conf, 'r')).get('dict')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset, valid_dataset, test_dataset = data_splitter(
    os.path.join(os.getcwd(),params['data']['conf']),
    params['data']['desgn']
)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=params['batch_size']['train'], 
    shuffle=True,
    num_workers=0,
    drop_last=True
)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=params['batch_size']['valid'], 
    shuffle=True,
    num_workers=0,
    drop_last=True
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=params['batch_size']['test'], 
    shuffle=True,
    num_workers=4,
    drop_last=True
)
if params['net_conf']['encoder']['use']:
    params['net_conf']['encoder']['scale_size'] = \
    train_dataset[0]['prcpal_p']*2
loss_fn = nn.MSELoss()
dataloaders={
    'train': train_dataloader,
    'valid': valid_dataloader,
    'test': test_dataloader
}
trainer = Trainer(
    dataloaders,
    params['net_conf'],
    loss_fn,
    os.path.join(args.checkpoint,os.listdir(args.checkpoint)[0]),
    device
)
trainer.train_implementation(params['net_conf']['inv_tr_conf'])

