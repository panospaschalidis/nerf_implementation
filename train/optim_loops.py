"""standard libraries"""
import torch
import pdb

"""third party imports"""


"""local specific imports"""
from data.utils import strf_sampling, inverse_transform_sampling
from .processing import DataCalibrator
from .utils import colorestimation, weights_estimation


def train_loop_nerf(
        dataloader, 
        model, 
        loss_fn, 
        optimizer, 
        inv_tr_conf,
        model_info,
        device):

    coarse_losses = []
    fine_losses = []
    losses = []
    #if model_info['type']=='nerf':
    #    model.train(True)
    # in orderto use just one loop for both validation and training 
    # we enable or disable gradient calculation for model parameters 
    # and consequently for modle output
    #if not is_train:
    #    models['coarse'].requires_grad_(is_train)
    #    models['fine'].requires_grad_(is_train)
    ## torch.cuda.empty_cache()
    for val, data in enumerate(dataloader):
        #print(f"""{val} from {len(dataloader)}""")
        for key, value in data.items():
            data[key] = value.to(device)
        if model.encoder_conf['use']:
            model.encode(data)
        # coarse model forward pass
        density, colors = model(data['samples'], data['rays_dir'], coarse=True) 
        # density->b_sxr_bxNc color->b_sxr_bxNcx3
        # volumetric rendering is being performed by 
        # functions weights_estimation and colorestimation
        weights, _ = weights_estimation(density, data['ti'][...,1:])
        est_colors_coarse = colorestimation(weights, colors)
        loss = loss_fn(est_colors_coarse,data['target'])
        coarse_losses.append(loss*len(data['samples']))
        # coarse model backward pass
        if inv_tr_conf['Fine']:
            data['rays_dir'], data['samples'], data['ti'] = inverse_transform_sampling(
                weights.detach(), 
                data['ti'], 
                inv_tr_conf, 
                data['rays_dir'][:,:,-1,:], # drop axis corresponding to number of samples on ray 
                data['rays_origin'].unsqueeze(2))
            # torch..cuda.empty_cache()
            # fine model forward pass
            density, colors = model(data['samples'], data['rays_dir'], coarse=False)
            weights, _ = weights_estimation(density, data['ti'][...,1:])
            est_colors_fine = colorestimation(weights, colors)
            loss_fine = loss_fn(est_colors_fine, data['target'])
            loss += loss_fine 
            fine_losses.append(loss_fine*len(data['samples']))
            losses.append(loss*len(data['samples']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()   
    if inv_tr_conf['Fine']:    
        return {
            'coarse': torch.tensor(coarse_losses), 
            'fine': torch.tensor(fine_losses),
            'optim': torch.tensor(losses)
            }
    else:
        return {'optim': torch.tensor(coarse_losses)}

def valid_loop_nerf(dataloader, 
                    model, 
                    loss_fn, 
                    inv_tr_conf, 
                    model_info, 
                    device):

    coarse_losses=[]
    fine_losses=[]
    losses = []
    #if model_info['type']=='nerf':
    #    model.train(False)
    # in orderto use just one loop for both validation and training 
    # we enable or disable gradient calculation for model parameters 
    # and consequently for modle output
    #if not is_train:
    #    models['coarse'].requires_grad_(is_train)
    #    models['fine'].requires_grad_(is_train)
    ## torch.cuda.empty_cache()
    with torch.no_grad():
        for val, data in enumerate(dataloader):
            #print(f"""{val} from {len(dataloader)}""")
            for key, value in data.items():
                data[key] = value.to(device)
            if model.encoder_conf['use']:
                model.encode(data)
            # coarse model forward pass
            density, colors = model(data['samples'], data['rays_dir'], coarse=True) 
            # density->b_sxr_bxNc color->b_sxr_bxNcx3
            # volumetric rendering is being performed by 
            # functions weights_estiamtion and colorestimation
            weights, _ = weights_estimation(density, data['ti'][...,1:])
            est_colors_coarse = colorestimation(weights, colors)
            loss = loss_fn(est_colors_coarse,data['target'])
            coarse_losses.append(loss*len(data['samples']))
            if inv_tr_conf['Fine']:
                data['rays_dir'], data['samples'], data['ti'] = inverse_transform_sampling(
                    weights.detach(), 
                    data['ti'], 
                    inv_tr_conf, 
                    data['rays_dir'][:,:,-1,:], # drop axis corresponding to number of samples on ray 
                    data['rays_origin'].unsqueeze(2))
                # torch..cuda.empty_cache()
                # fine model forward pass
                density, colors = model(data['samples'], data['rays_dir'], coarse=False)
                weights, _ = weights_estimation(density, data['ti'][...,1:])
                est_colors_fine = colorestimation(weights, colors)
                loss_fine = loss_fn(est_colors_fine, data['target'])
                loss += loss_fine
                fine_losses.append(loss_fine*len(data['samples']))
                losses.append(loss*len(data['samples']))
    if inv_tr_conf['Fine']:    
        return {
            'coarse': torch.tensor(coarse_losses), 
            'fine': torch.tensor(fine_losses),
            'optim': torch.tensor(losses)
            }
    else:
        return {'optim': torch.tensor(coarse_losses)}

def valid_loop_pixelneRF():
    pass

def train_loop_UniSURF():
    pass

def valid_loop_UniSURF():
    pass

def train_loop_ImageSURF():
    pass

def valid_loop_ImageSURF():
    pass
