"""standard libraries"""
import torch
import numpy as np
import pdb
import time 

"""third party imports"""
from torch import nn
from torch.nn.functional import avg_pool1d

"""local specific imports"""

class Nerf_creator(nn.Module):

    def __init__(self, arch_params):
        super().__init__()
        self.num_in_views = arch_params['num_in_views']
        self.inpt = arch_params['inpt']
        self.latent = arch_params['latent']
        self.out =  arch_params['out']
        self.density_1 = nn.Sequential(
            nn.Linear(self.inpt[0], self.latent[0]),
            nn.ReLU(),
            nn.Linear(self.latent[0], self.latent[0]),
            nn.ReLU(),
            nn.Linear(self.latent[0], self.latent[0]),
            nn.ReLU(),
            nn.Linear(self.latent[0], self.latent[0]),
            nn.ReLU()
        )
        self.density_2 = nn.Sequential(
            nn.Linear(self.inpt[0]+self.latent[0], self.latent[0]),
            nn.ReLU(),
            nn.Linear(self.latent[0], self.latent[0]),
            nn.ReLU(),             
            nn.Linear(self.latent[0], self.latent[0]),
            nn.ReLU(),
            nn.Linear(self.latent[0], self.latent[0]+1)
        )
        self.color = nn.Sequential(
                nn.Linear(self.latent[0]+self.inpt[1], self.latent[1]),
                nn.ReLU(),
                nn.Linear(self.latent[1], self.out),
                nn.Sigmoid()
        )

    def forward(self, points, dirs, encoder_vectors= None):
        if encoder_vectors!=None:
            points =  torch.cat(
                (points, encoder_vectors),
                dim=-1
            )
        # forward pass - 1st density layer
        out = self.density_1(points)
        # concatenation of fourth layer with PEd_points in order 
        # to form fifth layer's activation (fourth and fifth as
        # refered in NeRF)
        new_inpt = torch.cat((points,out), axis=-1)
        # forward pass - 2nd density layer
        new_out = self.density_2(new_inpt)
        # extraction of opacities after applying relu to [...,0]
        # indexed elements of new_out

        if self.num_in_views>1:
             density = torch.relu(
                 torch.mean(
                    new_out.reshape(
                        int(new_out.shape[0]/self.num_in_views),
                        self.num_in_views,
                        *new_out.shape[1:]
                    ),
                    dim=1
                 )[...,0]
            )
        else:
            density = torch.relu(new_out[...,0])

        # second concatenation betwen remaining new_out and PEd_dirs
        # and traversal of color layer
        col_inpt = torch.cat((new_out[...,1:], dirs), axis=-1)
        if self.num_in_views>1:
            col_inpt = torch.mean(
                col_inpt.reshape(
                    int(col_inpt.shape[0]/self.num_in_views),
                    self.num_in_views,
                    *col_inpt.shape[1:]
                ),
                dim=1
             )
        color = self.color(col_inpt)
        return density, color

