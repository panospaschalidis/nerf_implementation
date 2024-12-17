"""standard libraries"""
import torch
import numpy as np
import pdb
import time 

"""third party imports"""
from torch import nn
from torch.nn.functional import avg_pool1d

"""local specific imports"""
from .encoder import Encoder
from .customresnet import LinearResnetBlock
from .create_nerf import Nerf_creator
from .create_pixelnerf import Pixelnerf_creator

class NeRF(nn.Module):

    def __init__(self, net_conf):
        super().__init__()
        # self._inpt, self._PE_conf, self._width, self.out = model_arch.values()
        self._arch = net_conf['arch']
        self.encoder_conf = net_conf['encoder']
        self.fine_flag = net_conf['inv_tr_conf']['Fine']
        inpt=[]
        inpt.append(self._arch['inpt']*2*self._arch['PE_conf']['points'] + \
                                                        self._arch['inpt'])
        if self.encoder_conf['use']:
            inpt[0] += self.encoder_conf['channels']
            self.encoder = Encoder(self.encoder_conf)

        inpt.append(self._arch['inpt']*2*self._arch['PE_conf']['dirs'] + \
                                                        self._arch['inpt'])
        # defining net
        self.coarse = Nerf_creator(
            {
            'inpt': inpt,
            'latent': self._arch['latent_dim'],
            'out': self._arch['outpt'],
            'num_in_views': self.encoder_conf['num_in_views']
            }
        )
        if self.fine_flag:
            self.fine = Nerf_creator(
                {
                'inpt': inpt,
                'latent': self._arch['latent_dim'],
                'out': self._arch['outpt'],
                'num_in_views': self.encoder_conf['num_in_views']
                }
            )

    def encode(self, batch ):
        # create feature volume
        self.encoder(batch['input_views'].permute(0,3,1,2))
        self.w2cpose = batch['w2cpose']
        self.focal = batch['focal']
        self.prcpal_p = batch['prcpal_p']
         
    def forward(self, points, dirs, coarse=True):
        # inpt = torch.cat(3D_points,PE(3D_points))
        # apply positional encoding to both points and directions
        PEd_points = NeRF.PE(points, self._arch['PE_conf']['points'])
        encoder_vectors = None
        if self.encoder_conf['use']:
            # since all points belong to viewing frustum of specific camera, 
            # samples along each ray will have the same projection on image 
            # plane. Thus, we compute only the projection of the first sample
            # on each ray creating a camera_cords and then raster_cords of 
            # shape b_sxr_bx1x3 and b_sxr_bx1x2
            # # COMMENT
            # # for all points
            # # camera_cords = points @ self.w2cpose[:,:,:3].permute(0,2,1).unsqueeze(1)
            # # camera_cords = camera_cords + self.w2cpose[...,3][:,None,None,:]
            # # and all the factors camera_cords[_,_,:,0]/camera_cords[_,_,:,1] should be the same
            # from world space to camera space
            camera_cords = points[:,:,0,:] @ \
                                self.w2cpose[:,:,:3].permute(0,2,1) 
            camera_cords = camera_cords + self.w2cpose[:,None,:,3]
            # from camera space to raster space
            raster_cords = -camera_cords[...,:2]/camera_cords[...,2:] 
            raster_cords[...,0] = raster_cords[...,0] * self.focal[:,None]
            raster_cords[...,1] = raster_cords[...,1] *(- self.focal[:,None])
            raster_cords = raster_cords + self.prcpal_p[:,None,:]
            self.encoder.vector_extractor(raster_cords.unsqueeze(2))
            encoder_vectors = self.encoder.vectors.repeat_interleave(
                    PEd_points.shape[-2],
                    dim=2
            )
        PEd_dirs = NeRF.PE(dirs, self._arch['PE_conf']['dirs'])
        if coarse:
            return self.coarse(PEd_points, PEd_dirs, encoder_vectors)
        else:
            return self.fine(PEd_points, PEd_dirs, encoder_vectors)

    @staticmethod
    def PE(inpt, rep):
        # inpt corresponds to input where as rep  
        # corresponds to repetiitions of harmonic encodings
        device = inpt.device
        dtype = inpt.dtype
        L = torch.arange(rep).to(device)
        vector = (2**L)*np.pi
        tr_inpt = inpt[...,None]*vector[None,...] # b_sxr_bxNcx3xrep
        sin_tr_inpt = torch.sin(tr_inpt)
        cos_tr_inpt = torch.cos(tr_inpt)
        out = torch.cat(
            (torch.sin(tr_inpt), 
            torch.cos(tr_inpt)), 
            axis=3
        ) #...x6xrep    
        final_out = out.permute(0,1,2,4,3).reshape(
                                        *out.shape[:3], 
                                        out.shape[3]*out.shape[4]
                                        ) #b_sxr_bxNcx(6*rep)
        final_out = torch.cat(
            (inpt,
            final_out),
            axis=3
        )
        return final_out.to(dtype)


class PixelNeRF(nn.Module):
    
    def __init__(self, net_conf):
        super().__init__()
        self.view_space = net_conf['arch']['view_space']
        self.model_type = net_conf['model_info']['type']
        self.encoder_conf = net_conf['encoder']
        self.fine_flag = net_conf['inv_tr_conf']['Fine']
        self.arch_params = {}
        self.encoder = Encoder(self.encoder_conf)
        self._arch = net_conf['arch'].get(self.model_type)
        model_dict = {'nerf': Nerf_creator, 'pixelnerf': Pixelnerf_creator}
        self.arch_params = {}
        
        if self.model_type=='nerf':
            # define architecture parameters
            inpt = []
            # inpt[0] corresponds to points
            inpt.append(self._arch['inpt'])
            if self._arch['PE_conf']['points']:
                inpt[0] = inpt[0]*2*self._arch['PE_conf']['points'] + inpt[0]
            # since we use pixelnerf it is obvious that we use feature vectors
            inpt[0] += self.encoder_conf['channels']
            # inpt[1] corresponds to dirs
            inpt.append(self._arch['inpt'])
            if self._arch['PE_conf']['dirs']:
                inpt[1] = inpt[1]*2*self._arch['PE_conf']['dirs'] + inpt[1]
            self.arch_params['inpt'] = inpt
            self.arch_params['latent'] =  self._arch['latent_dim']
            self.arch_params['out'] = self._arch['outpt']
            self.arch_params['num_in_views'] =  self.encoder_conf['num_in_views']
        else:
            inpt =[]
            inpt.append(self._arch['inpt'])
            if self._arch['PE_conf']['points']:
                inpt[0] += inpt[0]*2*self._arch['PE_conf']['points']
            # add ray direction length
            # inpt += self._arch['inpt']
            inpt.append(self._arch['inpt'])
            if self._arch['PE_conf']['dirs']:
                inpt[1] = inpt[1] + self._arch['inpt']*2*self._arch['PE_conf']['dirs']
            self.arch_params['inpt'] = inpt
            self.arch_params['latent_dim'] =  self._arch['latent_dim']
            self.arch_params['out'] = self._arch['outpt']
            self.arch_params['res_blocks'] = self._arch['res_blocks']
            self.arch_params['mean_layer'] = self._arch['average']
            self.arch_params['num_in_views'] =  self.encoder_conf['num_in_views']
            
        self.coarse = model_dict.get(self.model_type)(self.arch_params)
        if self.fine_flag:
            self.fine = model_dict.get(self.model_type)(self.arch_params)
    
    def encode(self, batch ):
        # create feature volume
        self.encoder(batch['input_views'].flatten(0,1))
        self.w2cpose = batch['w2cpose']
        self.focal = batch['focal']
        self.prcpal_p = batch['prcpal_p']

    def forward(self, points, dirs, coarse=True):
        camera_cords = points.unsqueeze(1) @ self.w2cpose[...,:3].permute(0,1,3,2).unsqueeze(2)
        camera_cords = camera_cords + self.w2cpose[...,3][:,:,None,None]
        # since all points belong to viewing frustum of specific camera, 
        # samples along each ray will have the same projection on image 
        # plane. Thus, we compute only the projection of the first sample
        # on each ray creating a camera_cords and then raster_cords of 
        # shape b_sxr_bx1x3 and b_sxr_bx1x2

        # # COMMENT
        # # for all points
        # # and all the factors camera_cords[_,_,_,:,0]/camera_cords[_,_,_,:,1] 
        # # should not be the same, since points lie on rays in and out of 
        # # input_views viewing frustum, and they do not intersect 
        # # with input_views origin
        
        # from world space to camera space
        raster_cords = -camera_cords[...,:2]/camera_cords[...,2:] 
        raster_cords[...,0] = raster_cords[...,0] * self.focal[:, None, None, None]
        raster_cords[...,1] = raster_cords[...,1] *(- self.focal[:, None, None, None])
        raster_cords = raster_cords + self.prcpal_p[:, None, None, None, :]
        self.encoder.vector_extractor(raster_cords.flatten(0,1))
        if self.view_space: 
            """working in view space coordinate system"""
            camera_dirs = dirs[:,:,0,:].unsqueeze(1) @ \
                               self.w2cpose[...,:3].permute(0,1,3,2) 
            camera_dirs = camera_dirs + self.w2cpose[...,3].unsqueeze(-2)
            camera_dirs = camera_dirs.unsqueeze(3).repeat_interleave(dirs.shape[2], dim=3)
            points = PixelNeRF.PE(camera_cords.flatten(0,1), self._arch['PE_conf']['points'])
            # dirs = camera_dirs.flatten(0,1)
            if self._arch['PE_conf']['dirs']:
                dirs =  PixelNeRF.PE(camera_dirs.squeeze(1), self._arch['PE_conf']['dirs'])
            # by using flatten(0,1) we reshape our tensor from
            # batch_sizexnviewsxnum_of_raysxnum_ofsmaplesx3 -> 
            #-> (batch_sizexnviews)xnum_of_raysxnum_ofsmaplesx3 
        else:
            """ working in world space coordinate system"""
            points = PixelNeRF.PE(
                points.unsqueeze(1).repeat_interleave(
                    self.encoder_conf['num_in_views'], 
                    dim=1
                    ).flatten(0,1),
                self._arch['PE_conf']['points']
            )
           # dirs= dirs.unsqueeze(1).repeat_interleave(
           #     self.encoder_conf['num_in_views'], 
           #     dim=1).flatten(0,1)
            if self._arch['PE_conf']['dirs']:
                dirs =  PixelNeRF.PE(dirs, self._arch['PE_conf']['dirs'])
        if coarse:
            return self.coarse(points, dirs, self.encoder.vectors)            
        else:
            return self.fine(points, dirs, self.encoder.vectors)            

    @staticmethod
    def PE(inpt, rep):
        # inpt corresponds to input where as rep  
        # corresponds to repetiitions of harmonic encodings
        device = inpt.device
        dtype = inpt.dtype
        L = torch.arange(rep).to(device)
        vector = (2**L)*np.pi
        tr_inpt = inpt[...,None]*vector[None,...] # b_sxr_bxNcx3xrep
        sin_tr_inpt = torch.sin(tr_inpt)
        cos_tr_inpt = torch.cos(tr_inpt)
        out = torch.cat(
            (torch.sin(tr_inpt), 
            torch.cos(tr_inpt)), 
            axis=3
        ) #...x6xrep
        final_out = out.permute(0,1,2,4,3).reshape(
                                        *out.shape[:3], 
                                        out.shape[3]*out.shape[4]
                                        ) #b_sxr_bxNcx(6*rep)
        final_out = torch.cat(
            (inpt,
            final_out),
            axis=3
        )
        return final_out.to(dtype)


class UniSURF(nn.Module):
    def __init__():
        pass
    def forward():
        pass


class ImageSURF(nn.Module):
    def __init__():
        pass
    def forward():
        pass





