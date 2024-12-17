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

class Pixelnerf_creator(nn.Module):
    
    def __init__(self, arch_params):
        super().__init__()
        # input consists of points concatenated with encoding 
        # of points plus viewing directions
        self.inpt = arch_params['inpt']
        self.outpt = arch_params['out']
        self.latent_dim = arch_params['latent_dim']
        self.resnet_blocks = arch_params['res_blocks']
        self.mean_layer = arch_params['mean_layer']
        self.num_in_views =  arch_params['num_in_views']
        # pdb.set_trace()
        self.input_linear = nn.Linear(self.inpt[0], self.latent_dim)
        nn.init.constant_(self.input_linear.bias, 0)
        nn.init.kaiming_normal_(self.input_linear.weight, a=0, mode='fan_in')
        self.resnet_0 = nn.ModuleList(
            [LinearResnetBlock(
                self.latent_dim,
                self.latent_dim)
                for i in range(self.resnet_blocks-2)
            ]
        )
        self.linear_1 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.latent_dim,self.latent_dim+1)
        )
        nn.init.constant_(self.linear_1[1].bias, 0)
        nn.init.kaiming_normal_(self.linear_1[1].weight, a=0, mode='fan_in')
        self.linear_2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.latent_dim+1,self.latent_dim)
        )
        nn.init.constant_(self.linear_2[1].bias, 0)
        nn.init.zeros_(self.linear_2[1].weight)
        self.linear_3 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.latent_dim+self.inpt[1],self.latent_dim+self.inpt[1]),
                nn.ReLU(),
                nn.Linear(self.latent_dim+self.inpt[1],self.latent_dim+self.inpt[1])
        )
        nn.init.constant_(self.linear_3[1].bias, 0)
        nn.init.kaiming_normal_(self.linear_3[1].weight, a=0, mode='fan_in')
        nn.init.constant_(self.linear_3[3].bias, 0)
        nn.init.zeros_(self.linear_3[3].weight)
        self.residual_linear = nn.ModuleList(
             [nn.Linear(
                 self.latent_dim, 
                 self.latent_dim)
                 for i in range(self.mean_layer)
             ]
        )
        for block in self.residual_linear:
            nn.init.constant_(block.bias, 0)
            nn.init.kaiming_normal_(block.weight, a=0, mode='fan_in')
        self.output_linear = nn.Sequential(
          nn.Linear(self.latent_dim+self.inpt[1], self.outpt),
          nn.Sigmoid()
        )
        nn.init.constant_(self.output_linear[0].bias, 0)
        nn.init.kaiming_normal_(self.output_linear[0].weight, a=0, mode='fan_in')
        # we further add a sigmoid module in order to limit color values 
        # in range 0, 1
        # self.sigmoid = nn.Sigmoid()
    

    def forward(self, points, dirs, encoder_vectors):
        # inpt = torch.cat(
        #     (points, dirs),
        #     dim=-1
        # )
        inpt = points
        out = self.input_linear(inpt)
       
        #print('input_linear')
        for ind in range(len(self.resnet_0)):
            residual = self.residual_linear[ind](encoder_vectors)
            out = out + residual
            out = self.resnet_0[ind](out)
       
        #('self.resnet_0')
        if self.num_in_views>1:
            out = torch.mean(
                out.reshape(
                    int(out.shape[0]/self.num_in_views),
                    self.num_in_views,
                    *out.shape[1:]
                ),
                dim=1
            )
        out_1 = self.linear_1(out)
        density = torch.relu(out_1[...,0])
        
        out_1 = self.linear_2(out_1) + out
        
        final_in =  torch.cat(
            (out_1, dirs),
            dim=-1
        )
        # pdb.set_trace()
        out_2 = self.linear_3(final_in) + final_in
        rgb = self.output_linear(out_2)
       
        #('self.output_linear')
        # density = rgbsigma[...,-1]
        # color = self.sigmoid(rgbsigma[...,:-1])
        return density, rgb           

