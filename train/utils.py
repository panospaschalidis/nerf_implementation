""" utils.py contains methods used in train module """

"""standard libraries"""
import torch
import numpy as np
import pdb
import os

"""third party imports"""
from data.utils import inverse_transform_sampling


def colorestimation(weights,colors):
    weights = torch.broadcast_to(
        weights,
        (colors.shape[-1],*weights.shape)
        ).permute(1,2,3,0)
    est_colors = torch.sum(weights * colors, axis=2)
    #est_colors = est_colors + 1 - torch.sum(weights[...,-1],dim=-1).unsqueeze(-1)
    return est_colors      


def weights_estimation(density, ti):
        # estimation of opacity
        di = ti[...,1:]-ti[...,:-1] # b_sxr_bxNc
        # density.shape is not equal with density.shape
        # di shape is b_sxr_bx(Nc-1) since Nc samples 
        # lead to Nc-1 bins-distances(di).
        # Last point on ray is considered as fully opaque 
        # since optical energy transmission is terminated 
        # at that point. Thus regardless of density value
        # at that point we consider density at that point 
        # as infinite, hence we always set opacity[...,Nc]
        # equal to 1(we could achieve same result just by 
        # adding an extra element in di of infinite length
        # just as NeRF authors do(1e10)
        
        opacity = 1-torch.exp(-density[...,:-1]*di)
        opacity = torch.cat(
            (opacity,
            opacity.new_ones((*opacity.shape[:2],1))
            ),
            axis=-1
        )
        # estimation of transmittance 
        other_terms = torch.cumsum(-density[...,:-1]*di,-1)
        zero_terms = other_terms.new_zeros(other_terms.shape[:-1])
        transmittance= torch.exp(
            torch.cat((zero_terms[...,None], other_terms), axis=-1)
        )
        weights = opacity * transmittance
        return weights, opacity


def reconstruction(models, dataloader, inv_tr_conf, device):
    # reconstruction functon applies model to one random input
    # from test_dataloader, since batch_size['test']=1
    # models parsed to reconstruction function have been 
    # already set to validation mode since last time processed 
    # was in loop_NeRF function, with argument is_train set to False
    with torch.no_grad():
        data = next(iter(dataloader))
        for key, value in data.items():
            data[key] = value.to(device)
        media={}
        media['target'] = data['target'][-1,...]
        H, W = data['target'].shape[1:3]
        if models['coarse'].encoder_conf['use']:
            models['coarse'].encode(data)
            if inv_tr_conf['Fine']:
                models['fine'].encode(data)
            total_rays = torch.prod(data['prcpal_p'])*4
            # coarse model forward pass
            density0, colors0 = models['coarse'](
                data['samples'][:,:int(total_rays/4),...], 
                data['rays_dir'][:,:int(total_rays/4),...]
            )
            density1, colors1 = models['coarse'](
                data['samples'][:,int(total_rays/4):int(total_rays/2),...], 
                data['rays_dir'][:,int(total_rays/4):int(total_rays/2),...]
            )
            density2, colors2 = models['coarse'](
                data['samples'][:,int(total_rays/2):int(total_rays*(3/4)),...], 
                data['rays_dir'][:,int(total_rays/2):int(total_rays*(3/4)),...]
            )
            density3, colors3 = models['coarse'](
                data['samples'][:,int(total_rays*(3/4)):,...], 
                data['rays_dir'][:,int(total_rays*(3/4)):,...]
            )
            density = torch.cat((density0, density1, density2, density3), dim=1)
            colors = torch.cat((colors0, colors1, colors2, colors3), dim=1)
            # density->b_sxr_bxNc color->b_sxr_bxNcx3
            # volumetric rendering is being performed by 
            # functions weights_estiamtion and colorestimation
            weights,_ = weights_estimation(density, data['ti'][...,1:])
            media['depth_coarse'] = torch.sum(
                (weights*data['ti'][...,1:]).squeeze(0),
                dim=1).reshape(H,W)
            media['coarse_im'] = colorestimation(weights, colors)[-1,:,:].reshape(H, W, 3)
            if inv_tr_conf['Fine']:
                data['rays_dir'], data['samples'], data['ti'] = inverse_transform_sampling(
                    weights.detach(), 
                    data['ti'], 
                    inv_tr_conf, 
                    data['rays_dir'][:,:,-1,:], # drop axis corresponding to number of samples on ray 
                    data['rays_origin'].unsqueeze(2))
                # torch..cuda.empty_cache()
                # fine model forward pass
                density0, colors0 = models['fine'](
                    data['samples'][:,:int(total_rays/2),...], 
                    data['rays_dir'][:,:int(total_rays/2),...]
                )
                density1, colors1 = models['fine'](
                data['samples'][:,int(total_rays/2):,...], 
                data['rays_dir'][:,int(total_rays/2):,...]
                )
                density = torch.cat((density0, density1), dim=1)
                colors = torch.cat((colors0, colors1), dim=1)
                weights, _ = weights_estimation(density, data['ti'][...,1:])
                media['depth_fine'] = torch.sum(
                    (weights*data['ti'][...,1:]).squeeze(0),
                    dim=1).reshape(H,W)
                media['fine_im'] = colorestimation(weights, colors)[-1,:,:].reshape(H, W, 3)
        else:
            density, colors = models['coarse'](
                data['samples'], 
                data['rays_dir']
            )
            # density->b_sxr_bxNc color->b_sxr_bxNcx3
            # volumetric rendering is being performed by 
            # functions weights_estiamtion and colorestimation
            weights,_ = weights_estimation(density, data['ti'][...,1:])
            media['depth_coarse'] = torch.sum(
                (weights*data['ti'][...,1:]).squeeze(0),
                dim=1).reshape(H,W)
            media['coarse_im'] = colorestimation(weights, colors)[-1,:,:].reshape(H, W, 3)
            if inv_tr_conf['Fine']:
                data['rays_dir'], data['samples'], data['ti'] = inverse_transform_sampling(
                    weights.detach(), 
                    data['ti'], 
                    inv_tr_conf, 
                    data['rays_dir'][:,:,-1,:], # drop axis corresponding to number of samples on ray 
                    data['rays_origin'].unsqueeze(2))
                # torch..cuda.empty_cache()
                # fine model forward pass
                density, colors = models['fine'](
                    data['samples'], 
                    data['rays_dir']
                )
                weights,_ = weights_estimation(density, data['ti'][...,1:])
                media['depth_fine'] = torch.sum(
                    (weights*data['ti'][...,1:]).squeeze(0),
                    dim=1).reshape(H,W)
                media['fine_im'] = colorestimation(weights, colors)[-1,:,:].reshape(H, W, 3)

    return media

def save_models(model, optimizer, epoch, bbox, project_name ):
    if bbox:
        path = os.path.join(
            'checkpoints/bbox',
            project_name + '.' + str(epoch) + '.pth'
            )
    else:
        path = os.path.join(
            'checkpoints/nobbox',
            project_name + '.' + str(epoch) + '.pth'
            )

    torch.save({
        'model': model,
        'optimizer': optimizer,
        'epoch': epoch
        }, path
    )
        
def load_checkpoint(path, device):
    return torch.load(path, map_location=torch.device(device))
