"""standard libraries """
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
import wandb

"""third party libraries"""

"""load specific imports"""
from model import NeRF, PixelNeRF
from .optim_loops import train_loop_nerf, valid_loop_nerf
from .utils import reconstruction, save_models, load_checkpoint
from .metrics import Metrics


class Trainer():

    def __init__(self, dataloaders, net_conf, l_fn, checkpoint, device):
        self._dataloaders = dataloaders
        #if net_conf['encoder']['use']:
        #    net_conf['encoder']['scale_size'] = \
        #        next(iter(dataloaders['train']))['prcpal_p'][0]*2
        self.net_conf = net_conf
        self._loss_fn = l_fn
        self._device = device
        self.supported_models = {
            'nerf': NeRF,
            'pixelnerf': PixelNeRF
        }
        self.net_type = net_conf['model_info']['type']
        self.net = self.supported_models[net_conf['model_info']['type']]
        self.checkpoint = checkpoint

    def train_preparation(self):
        # models
        # print(model_ti)
        dtype = torch.float32
        if not self.checkpoint:
            self.model = self.net(self.net_conf).to(dtype).to(self._device)
            l_r, l_r_final, self.epochs, self.bbox_limit = self.net_conf['learn_conf'].values()
            # optimizers
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=l_r,
                eps=1e-08
            )
            self.resume_epoch = None
        else:
            l_r, l_r_final, self.epochs, self.bbox_limit = self.net_conf['learn_conf'].values()
            self.model, self.optimizer, self.resume_epoch = load_checkpoint(
                self.checkpoint,
                self._device
                ).values()

        self.has_decay = False
        if self.net_conf['learn_conf']['l_r_dec'] != None:
            # exponential learning rate decay
            gamma = np.exp((np.log(l_r_final)-np.log(l_r))/(self.epochs-1))
            self.scheduler =torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma,
                last_epoch=-1,
                verbose=True
            )
            self.has_decay = True

    def train_implementation(self, inv_tr_conf):
        self.train_preparation()
        trn_metrics = Metrics(self._dataloaders['train'].batch_size)
        val_metrics = Metrics(self._dataloaders['valid'].batch_size)
        start = 0 if not self.resume_epoch else self.resume_epoch+1
        bbox = True
        for epoch in range(start, self.epochs):
            if self.net_type=='pixelnerf':
                if epoch==self.bbox_limit:
                    bbox =  False
                    self._dataloaders['train'].dataset.bbox = False
                    self._dataloaders['valid'].dataset.bbox = False

            losses = train_loop_nerf(
                self._dataloaders['train'],
                self.model,
                self._loss_fn,
                self.optimizer,
                inv_tr_conf,
                self.net_conf['model_info'],
                self._device
            )
            # epoch loss and psnr estimation
            trn_metrics.loss = losses
            trn_metrics.psnr = trn_metrics.loss
            log = {'trn_optim_loss': trn_metrics.loss['optim']}
            log['train_optim_psnr'] = trn_metrics.psnr['optim']

            # scheduler step if learning decay is performed 
            if self.has_decay:
                self.scheduler.step()

            # validation loop
            losses = valid_loop_nerf(
                self._dataloaders['valid'],
                self.model,
                self._loss_fn,
                inv_tr_conf,
                self.net_conf['model_info'],
                self._device
                )
            val_metrics.loss = losses
            val_metrics.psnr = val_metrics.loss
            log['val_optim_loss'] = val_metrics.loss['optim']
            log['val_optim_psnr'] =val_metrics.psnr['optim']
            print(f"Epoch {epoch+1}\n-------------------------------")
            if  inv_tr_conf['Fine']:
                print(f"""MSE Train Loss-Coarse:{trn_metrics.loss['coarse']}""")
                print(f"""MSE Train Loss-Fine:{trn_metrics.loss['fine']}""")
                print(f"""MSE Train Loss-Opti:{trn_metrics.loss['optim']}""")
                print(f"""MSE Val Loss-Coarse:{val_metrics.loss['coarse']}""")
                print(f"""MSE Val Loss-Fine:{val_metrics.loss['fine']}""")
                print(f"""MSE Val Loss-Opti:{val_metrics.loss['optim']}""")
            else:
                print(f"""MSE Train Loss-Opti:{trn_metrics.loss['optim']}""")
                print(f"""MSE Val Loss-Opti:{val_metrics.loss['optim']}""")
           # media = reconstruction(
           #     self.model, 
           #     self._dataloaders['test'], 
           #     inv_tr_conf, 
           #     self._device
           # )
           #log['target'] = wandb.Image(np.asarray(media['target'].to('cpu')))
     #       log['coarse_im'] = wandb.Image(np.asarray(media['coarse_im'].to('cpu')))
     #       if inv_tr_conf['Fine']:
     #         log['fine_im'] = wandb.Image(np.asarray(media['fine_im'].to('cpu')))            
            wandb.log(log, step=epoch)
            if epoch>0 and (np.ceil(epoch/100)!=np.ceil((epoch+1)/100)):
                save_models(self.model, self.optimizer, epoch, bbox, wandb.run.name)

class PixelNeRFTrainer(Trainer):
    def __init__():
        pass

class UniSURFTrainer(Trainer):
    def __init__():
        pass

class ImageSURFTrainer(Trainer):
    def __init__():
        pass

