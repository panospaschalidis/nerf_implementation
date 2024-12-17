"""standard libraries"""
import torch
import pdb

"""third party imports"""


"""local specific imports"""
from data.utils import strf_sampling, inverse_transform_sampling

class DataCalibrator():

    def __init__(self, data, model_info, device):
        self.data = data
        self.device = device
        self.model_info = model_info

    def nerf(self):
        # define the axes in order to perform batch indexing
        # following dimensions correspond to samples dimensions
        # b_sxr_bxNcx3
        dim0, dim1, dim2, dim3 = self.data['samples'].shape
        ax0 = torch.arange(dim0)[:,None,None,None]
        # for ax1 batch['random_indices] will be used so as 
        # to perform ray_batching
        ax1 = self.data['ray_indices'][...,None,None]
        ax2 = torch.arange(dim2)[None,None,:,None]
        ax3 = torch.arange(dim3)[None,None,None,:]
        # samples shape ->  b_sxr_bxNcx3
        samples = self.data['samples'][ax0,ax1,ax2,ax3]
        # target shape -> b_sx1x(H*W)x3 where H, W are image plane dimensions
        self.data['target'] = self.data['target'].reshape((dim0,dim1,dim3))
        self.data['target'] = self.data['target'].unsqueeze(1)
        target = self.data['target'][ax0,-1,ax1[:,None,:,-1],ax3]
        # rays_dir shape -> b_sxr_bxNcx3
        rays_dir = self.data['rays_dir'][ax0, ax1, ax2, ax3]
        # ti shape -> b_sxr_bx(Nc+1)
        # since ti's second axis is of Nc+1 length  
        # we redefine ax2
        ax2 =  torch.arange(dim2+1)[None,None,:]
        ti =self.data['ti'][ax0[...,-1],ax1[...,-1],ax2]
        return {
                'samples': samples.to(self.device),
                'target': target.to(self.device),
                'rays_dir': rays_dir.to(self.device),
                'rays_origin': self.data['rays_origin'].to(self.device),
                'ti': ti.to(self.device)
                }

    def pixelnerf(self, bbox_flag):
        batch_size, nviews, H, W, _ = self.data['rgb'].shape
        num_in_views = self.model_info['input_conf']['num_in_views']
        num_of_rays = self.model_info['input_conf']['ray_batch']
        ind = [torch.randperm(nviews)[:num_in_views] for i in range(batch_size)]
        ind = torch.stack(ind)
        dim0 = torch.arange(batch_size)[...,None]
        input_views = self.data['rgb'][dim0, ind, ...].permute(0,1,4,2,3) # batch_sizexnum_in_viewsx3HxW
        # since input views will be encoded through encoder defien in model module
        # we reshape input_views into the desired shape of
        # (batch_sizexnum_in_views)x3HxW 
        # reshape performs row major ordering 
        input_views = input_views.flatten(0,1)
        input_c2w_poses = self.data['poses'][dim0, ind, ...]
        w2crotmatrix = input_c2w_poses[...,:3,:3].permute(0,1,3,2)
        w2ctranslation = (-w2crotmatrix @ input_c2w_poses[...,:3,3].unsqueeze(3))#[...,None]
        w2cpose = torch.cat((w2crotmatrix, w2ctranslation), dim=-1)

        # rays = torch.empty((batch_size, num_of_rays, 3))
        box_ind = []
        rows = []
        cols = []
        for instance in range(batch_size):
            box_ind.append(torch.randint(nviews, (num_of_rays,))[None,...]) # 1xnum_of_rays
            if bbox_flag:
                rows.append(self.data['b_box'][instance, box_ind[instance], 0] + (torch.rand(num_of_rays)*( 
                        self.data['b_box'][instance, box_ind[instance], 1] - \
                        1 - self.data['b_box'][instance, box_ind[instance], 0])).int())
                cols.append(self.data['b_box'][instance, box_ind[instance], 2]+(torch.rand(num_of_rays)*( 
                        self.data['b_box'][instance, box_ind[instance], 3] - \
                        1 - self.data['b_box'][instance, box_ind[instance], 2])).int())
            else:
                rows.append(torch.randint(H*W,()))
                cols.append(torch.randint(H*W,()))
            
        # ray_ind.append(rows*W +cols) # rows, cols of shape 1xnum_of_rays
        rays_dir = self.data['rays_dir'][
                        torch.arange(batch_size)[:,None], 
                        torch.cat(rows)*W + torch.cat(cols), 
                        torch.cat(box_ind), 
                        :
                        ]
        rays_origin = self.data['rays_origin'][torch.arange(batch_size)[:,None], torch.cat(box_ind), :]
        samples, ti = strf_sampling(
            self.model_info['hier_samp_conf']['Nc'],
            rays_dir,
            rays_origin.unsqueeze(2),
            self.model_info['hier_samp_conf']['t_near'],
            self.model_info['hier_samp_conf']['t_far']
        )
        rays_dir = rays_dir.unsqueeze(2).repeat_interleave(
            self.model_info['hier_samp_conf']['Nc'],
            dim=2
        )
        target = self.data['rgb'][
                                    torch.arange(batch_size)[:,None],
                                    torch.cat(box_ind),
                                    torch.cat(rows),
                                    torch.cat(cols),
                                    :
                                    ].unsqueeze(1)

        
        return {
                'input_views': input_views.to(self.device),
                'w2cpose': w2cpose,
                'focal': self.data['focal'],
                'prcpal_p': self.data['prcpal_p'],
                'samples': samples.float().to(self.device),
                'target': target.to(self.device),
                'rays_dir': rays_dir.to(self.device),
                'rays_origin': rays_origin.to(self.device),
                'ti': ti.float().to(self.device)
                }

