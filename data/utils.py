""" utils.py contains methods used in data module

inside both Dataset and Dataloader classes"""
"""standard libraries"""
import os
import numpy as np
import random
import torch
import pdb

"""third party imports"""


def csvreturn(list_of_files):
    # the assumption of just one csv file
    # inside list_of_files has been made
    csv = [
        x for x in list_of_files 
        if os.path.splitext(x) == 'csv'
        ]
    return csv[0]


def ray_tracing(image_shape, focal, pose):
    # implementation of ray tracing assuming
    # 1. pose corresponds to c2w matrix in row major
    # 2. H equals W
    # 3. focal corresponds to effective focal length
    # Since  |u|   |fx*xc+ox*zc|
    #        |v| = |fy*yc+oy*zc|
    #        |1|   |zc|
    # if principal point is on center of image xnd zc equals 1
    # xc = u/fx & yc = v/fy
    H, W = image_shape[:2]
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    y =-(y-H/2) / focal
    x =(x-W/2) / focal
    z = -torch.ones(y.shape)
    # coordiantes of image plane in camera space are
    # expressed in homogeneous coordinates
    cords = torch.cat(
        (x.flatten()[None,:],
        y.flatten()[None,:],
        z.flatten()[None,:]),
        axis=0
        ).permute(1,0) #ray_batchX3
    # ray direction is equal with cords - origin_in_
    # _camera_space. Since product of pose with cords 
    # is equal with the summation of pose[:3,:3]*cords 
    # plus pose[3,:] and camera origin in world system
    # is pose[3,:], as it is (0,0,0) in camera space
    # rays directions in world space are given as follows
    #if pose.ndim>2:
    cords = cords/torch.linalg.norm(cords, dim=-1).unsqueeze(-1)
    cords = cords.unsqueeze(1).repeat_interleave(pose.shape[0],dim=1)
    rays_dir = torch.sum(cords[:,:,None,:]*pose[:,:3,:3],axis=-1)
    rays_origin = pose[:,:3,3]
   # else:
    #    rays_dir = torch.sum(cords[:,None,:]*pose[:3,:3],axis=-1)
     #   rays_origin = torch.tensor(pose[:3,3])
    # rays_origin = np.broadcast_to(pose[:3,3],rays_dir.shape)
    # in case you would like to transfrom rays_origin vector
    # to a matrix of rays_dir.shape uncommnet line below rays_dir
    # definition
    return rays_dir, rays_origin    

def strf_sampling(Nc, rays_dir, rays_origin, t_near, t_far):
    # Sampling on each ray using equation ray -> o+t*d
    # In order to implement equation (2) from NeRF
    # we have to create the bounds of the bins from where
    # sampling is performed. Since we need 64 samples 
    # therefore 64 bins we create their bounds simultaneously
    device = rays_dir.device
    shape = rays_dir.shape[:-1]
    i = torch.arange(1,Nc+1)
    tn = t_near + ((i-1)/Nc)*(t_far-t_near)
    tf = t_near + (i/Nc)*(t_far-t_near)
    #pdb.set_trace()
    ti = torch.tensor(
        np.random.uniform(
            tn,
            tf,
            (*rays_dir.shape[:rays_dir.ndim-1],Nc)
        ) # num_of_raysxnviewsxNc
    )   
    ti = ti.to(device)
    # rays_dir has shape num_of_raysx3 and ti num_of_raysxNc
    # rays_dir[i,:] corresponds to a direction 
    # in order to apply ray tracing equation we need 
    # to multiply each one of rays_dir[i,:] with the
    # corresponding Nc values from ti[i,:]
    # hence we use einsum add rays_origin, creating 
    # a matrix of shape ray_batchxNcx3 containing all the points
    # sampled on each pixel's ray
    samples = ti.unsqueeze(-1)*rays_dir.unsqueeze(1) + rays_origin.unsqueeze(1)
    ti = torch.cat(
        (t_near*torch.ones(*ti.shape[:-1],1, device=device),
        ti),
        axis=-1
    ) # num_of_raysxnviewsx(Nc+1)
    dtype = rays_dir.dtype
    return samples.reshape((*shape,*samples.shape[1:])).to(dtype), \
            ti.reshape((*shape,*ti.shape[1:])).to(dtype)

#def strf_sampling(Nc, rays_dir, rays_origin, t_near, t_far):
#    # Sampling on each ray using equation ray -> o+t*d
#    # In order to implement equation (2) from NeRF
#    # we have to create the bounds of the bins from where
#    # sampling is performed. Since we need 64 samples 
#    # therefore 64 bins we create their bounds simultaneously
#    i = torch.arange(1,Nc+1)
#    tn = t_near + ((i-1)/Nc)*(t_far-t_near)
#    tf = t_near + (i/Nc)*(t_far-t_near)
#    
#    ti = torch.tensor(
#        np.random.uniform(
#            tn,
#            tf,
#            (*rays_dir.shape[:rays_dir.ndim-1],Nc)
#        ) # num_of_raysxnviewsxNc
#    )
#    # rays_dir has shape num_of_raysx3 and ti num_of_raysxNc
#    # rays_dir[i,:] corresponds to a direction 
#    # in order to apply ray tracing equation we need 
#    # to multiply each one of rays_dir[i,:] with the
#    # corresponding Nc values from ti[i,:]
#    # hence we use einsum add rays_origin, creating 
#    # a matrix of shape ray_batchxNcx3 containing all the points
#    # sampled on each pixel's ray
#    ti_broad = torch.broadcast_to(
#        ti,
#        (rays_dir.shape[-1],*ti.shape)
#    ).permute(1,2,0,3) # num_of_raysxnviewsxNc -> 3xnum_of_raysxnviewsxNc ->
#    # -> num_of_raysxnviewsx3xNc
#    # squeeze(1) affects ray_origin only in nerf case where
#    # rays_dir are of size raysxNcx3 and rays_origin of size 1x3
#    # in pixelnerf case there isn't any dimension of size 1 in rays_origin
#    samples = (rays_dir[...,None]*ti_broad).permute(0,1,3,2) + \
#                                            rays_origin
#    # num_of_raysxnviewsx3x1 * num_of_raysxnviewsx3xNc + 1x3 -> 
#    # -> num_of_raysxnviewsxNcx3 + 1x3  
#    # we have to compute as well di that correspond to 
#    # differences between adjacent ti, after adding t_near
#    # as first element of every row in ti 
#    ti = torch.cat(
#        (t_near*torch.ones(*ti.shape[:-1],1),
#        ti),
#        axis=-1
#    ) # num_of_raysxnviewsx(Nc+1)
#    dtype = rays_dir.dtype
#    return samples.squeeze(1).to(dtype), ti.squeeze(1).to(dtype)
#

def inverse_transform_sampling(weights, ti, inv_tr_conf, rays_dir, rays_origin):
    # fine_sampling function performs inverse transform sampling
    # and refine sampling as well depending on inv_tr_conf values
    device = weights.device
    weights = weights + 1e-5 # in order to avoid Nan
    sum_of_weights = torch.broadcast_to(
        torch.sum(weights,axis=-1),
        (weights.shape[-1],*weights.shape[:2])
        ).permute(1,2,0)
    weights_hat = weights*(sum_of_weights**(-1))
    cdf = torch.cumsum(weights_hat,-1)
    # cdf for t<t0 is equal with zero, thus we have to 
    # include zero as first element of every cdf
    cdf = torch.cat(
        (cdf.new_zeros((*weights_hat.shape[:2],1)),
        cdf),
        axis=-1
    )
    u = torch.tensor(
        np.random.uniform(0, 1, (*cdf.shape[:2], inv_tr_conf['Nf']))
    ).float().to(device)
    # we need to locate in which of the cdf bins
    # each u lies
    # first we compute the upper limit since each bin 
    # of cdf is closed on the left side, so any faults
    # caused from the case of getting u equal with one 
    # of the cdf values is prevented
    # cdf.shape -> b_sxr_bx(Nc+1) u.shape->b_sxr_bxNf
    u_br = torch.broadcast_to(
        u,
        (cdf.shape[-1],*u.shape)
        ).permute(1,2,0,3) #b_sxr_bx(Nc+1)xNf
    # before subtracting u from cdf we have to reshape cdf to b_sxr_bx(Nc+1)x1
    diff = cdf[...,None]-u_br #b_sxr_bx(Nc+1)xNf
    #locate negative indices and replace them with zero
    ind_of_int = torch.where(diff>0, diff, diff.max()+1).argmin(axis=2) #b_sxr_bxNf
    batch_size, rays_batch, Nc = cdf.shape
    w_h = cdf[
        torch.arange(batch_size)[:,None,None],
        torch.arange(rays_batch)[None,:,None],
        ind_of_int
    ]
    w_l = cdf[
        torch.arange(batch_size)[:,None,None],
        torch.arange(rays_batch)[None,:,None],
        ind_of_int-1
    ]
    t_h = ti[
        torch.arange(batch_size)[:,None,None],
        torch.arange(rays_batch)[None,:,None],
        ind_of_int
    ]
    t_l = ti[
        torch.arange(batch_size)[:,None,None],
        torch.arange(rays_batch)[None,:,None],
        ind_of_int-1
    ]
    ti_fine = t_l + ( (u-w_l)*( (w_h-w_l)**(-1) ) ) *(t_h-t_l)
    # ReFine
    if inv_tr_conf['Refine']:
        # estimation of ray termination depth
        exp_term_depth = torch.sum(ti[...,1:]*weights, dim=-1)
        random = torch.randn((*exp_term_depth.shape, inv_tr_conf['Nrf']), device=device)
        ti_refine = random*inv_tr_conf['std']+exp_term_depth[...,None]
        ti_refine = torch.max(
          torch.min(ti_refine,1.8*torch.ones_like(ti_refine[...,-1].unsqueeze(-1),device=device)),
          0.8*torch.ones_like(ti_refine[...,-1].unsqueeze(-1))
          )
        #ti_fine = torch.cat(
        #            (ti_fine,
        #            ti_refine),
        #            dim=-1
        #)
    ti_final = torch.sort(torch.cat((ti_fine,ti),axis=-1),axis=-1)[0]
    ti_fine = ti_final[:,:,1:] # from b_sxr_bx(Nf+Nc+1) to b_sxr_bx(Nf+Nc) 
    # estimation of new samples based on rays_dir, rays_origin and ti_final
    rays_dir = torch.broadcast_to(
        rays_dir,
        (ti_fine.shape[-1],*rays_dir.shape)
        ).permute(1,2,0,3)
    ti_fine = torch.broadcast_to(
        ti_fine,
        (rays_dir.shape[-1],*ti_fine.shape)
        ).permute(1,2,3,0)
    dtype = rays_dir.dtype
    # pdb.set_trace()
    samples_new = rays_dir*(ti_fine.to(dtype)) + rays_origin
    return rays_dir, samples_new, ti_final.to(dtype)


def random_indices(mat_shape, ray_batch):
    # dependeing on value of ray_batch 
    # random_indices return a map of random
    # indices selected from the meshgrid of 
    # mat_shape shaped matrix
    H, W = mat_shape[:2]
    # in order to sample randomly without repetition
    # we use random.sample
    # range(H*W) corresponds to (0,H*W), thus random.sample
    # gives values from 0 to H*W-1
    ray_indices = random.sample(range(H*W), ray_batch)
    return torch.tensor(ray_indices)


