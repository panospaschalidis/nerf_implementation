"""standard libraries"""
import numpy as np
import torch
import pdb

"""third party imports"""


"""local specific imports"""


class Metrics():

    def __init__(self, dtl_batch_size):
        # computation of mean loss depending on losses of every batch 
        # being processed during one epoch
        # dataloader batch size
        self.dtl_b_size = dtl_batch_size
        self._loss = {}
        self._psnr = {}
    @property    
    def loss(self):
         return self._loss
    
    @loss.setter
    def loss(self, losses):
        for key in losses:
            self._loss[key] = torch.sum(losses[key])/(len(losses[key])*self.dtl_b_size)
    
    @property 
    def psnr(self):
        return self._psnr
    
    @psnr.setter
    def psnr(self, loss):
        # loss_psnr_estiamtion returns mean loss and psnr after every epoch
        # during each epoch len(losses) optimizations are performed
        # that correspond to the batches each dataloader divides the datail
        for key in loss:
            self._psnr[key] = -10*torch.log(loss[key].to(torch.float32))/np.log(10)

    def ssim(self):
        pass

    def lpips(self):
        pass



