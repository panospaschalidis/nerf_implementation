"""standard libraries"""
import torch
import pdb
import os
import argparse
import yaml 
import matplotlib.pyplot as plt
import numpy as np
"""third party imports"""
from torch.utils.data import DataLoader

"""local specific imports"""
from train.utils import reconstruction
from data import data_splitter


parser = argparse.ArgumentParser(
    description = "Radiance Fields reconstruction process"
)
parser.add_argument(
    "--conf",
    default=None,
    help="configuration file path"
)
parser.add_argument(
    "--model_path",
    default=None,
    help="path to given checkpoint"
)
parser.add_argument(
    "--render_path",
    default=None,
    help="path to render directory"
)
args = parser.parse_args()
params = yaml.full_load(open(args.conf, 'r')).get('dict')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dataset, valid_dataset, test_dataset = data_splitter(
    os.path.join(os.getcwd(),params['data']['conf']),
    params['data']['desgn'],
    device
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=params['batch_size']['test'], 
    shuffle=True,
    drop_last=True
)

dict_ = {}
for j in range(len(trained_models)):
    models =  torch.load(os.path.join(args.model_path, map_location=('cpu'))['models']
    models['coarse'].device = device
    models['coarse'].encoder.device = device
    images = [] 
    target = []  
    for i in range(50):
        if i!=torch.tensor(test_dataset.input_views):
            test_dataloader.dataset.target_view = i
            print(f"{i}")
            media = reconstruction(
                models,
                test_dataloader,
                params['inv_tr_conf'],
                device
            )
            images.append(media['coarse_im'][None,...])
            if j==0:
                target.append(media['target'][None,...])
    if j==0:
        dict_['target'] = torch.cat(target)

    dict_[trained_models[j]] = torch.cat(images)
    
for val in range(len(dict_['target'])):
    fig, ax = plt.subplots(3,3)
    print(f"{val}")
    for num, key  in enumerate(dict_):
        ax[int(np.ceil((num+1)/3))-1,int(num%3)].imshow(dict_[key][val,...])
        ax[int(np.ceil((num+1)/3))-1,int(num%3)].axis("off")
        ax[int(np.ceil((num+1)/3))-1,int(num%3)].set_title(key)
    fig_label= os.path.join(args.render_path,str(val)+'.png')
    fig.savefig(fig_label)

