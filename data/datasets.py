"""standard libraries"""
import os
import sys
import glob
import torch
import numpy as np
import pdb
import random

"""third party imports"""
from torch.utils.data import Dataset
from torchvision import transforms
 
"""local specific imports"""
from data.utils import csvreturn, ray_tracing, strf_sampling, random_indices


class LegoDataset(Dataset):

    def __init__(self, path, input_conf, dat_type):
        path = path+'.npz'
        self.images = np.load(path)['images'].astype(np.float32)
        self.poses = np.load(path)['poses'].astype(np.float32)
        self.focal = np.load(path)['focal'].astype(np.float32)
        self.transform = transforms.ToTensor()
        self.ray_batch = input_conf['ray_conf']['ray_batch'] # , for single valued dictionaries
        self.t_near, self.t_far, self.samples_on_ray = \
            input_conf['hier_samp_conf'].values()
        self.dat_type = dat_type
        self.encoder_use = input_conf['encoder']
    
    def __len__(self):
        return(len(self.images))

    def __getitem__(self,idx):
        # pixel intensities are already in float38, hence the usual 
        # transform from uint8 to float32 is not required
        image_shape = self.images[idx].shape
        item = {}
        item['rays_dir'], item['rays_origin'] = ray_tracing(
                                        image_shape,
                                        self.focal,
                                        torch.tensor(self.poses[idx][None,...]),
        )
        if self.dat_type!='test':
            ray_indices = random_indices(image_shape, self.ray_batch)
            item['rays_dir'] = item['rays_dir'][ray_indices,...]
            item['target'] = self.transform(self.images[idx]).permute(1,2,0)
            item['target'] = item['target'].flatten(0,1) #HxWx3->(HxW)x3
            item['target'] = item['target'][ray_indices,...]
        else:
            item['target'] = self.transform(self.images[idx]).permute(1,2,0)

        item['samples'], item['ti'] = strf_sampling(
                                self.samples_on_ray,
                                item['rays_dir'].squeeze(1),
                                item['rays_origin'],
                                self.t_near,
                                self.t_far
                                )
                
        item['rays_dir'] = item['rays_dir'].repeat_interleave(
            self.samples_on_ray,
            dim=1) # Ncxray_batchx3
        if  self.encoder_use:
            item['input_views'] = self.transform(self.images[idx]).permute(1,2,0) 
            H, W = image_shape[:2]
            item['focal'] = torch.tensor(self.focal)
            w2crotmatrix = np.transpose(self.poses[idx][:3,:3], (1,0))
            w2ctranslation = (-w2crotmatrix @ self.poses[idx][:3,3])[...,None]
            item['w2cpose'] = torch.tensor(
                np.concatenate((w2crotmatrix, w2ctranslation), axis=1)
            )
            item['prcpal_p'] = torch.tensor([W/2, H/2])
        #for key in item:
        #    if not isinstance(item[key], int):
        #        print(f"{key}: {item[key].shape}")
        return item


class SRNDataset(Dataset):

    def __init__(self, path, conf, dat_type):
        #super().__init__()
        self.instance_paths = glob.glob(os.path.join(path,'*'))
        # self.instance_paths = random.sample(self.instance_paths, int((len(self.instance_paths)/60)))
        # SRN authors create perinent dataset under specific conventions
        # reganding the camera coordinate systems. Specifically they consider
        # the yaxis pointing downwards and the zaxis away from world coordinate
        # system. In order to adjust the extracted poses to the standard camera
        # coordinate systems we define self._coord_trans that is going to 
        # transform the given poses
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.transform = transforms.Compose(
            [
                transforms.Normalize((0.5, 0.5, 0.5, 0), (0.5, 0.5, 0.5, 1))
            ]
        )
        self.num_in_views = conf['num_in_views']
        self.input_views = conf['input_views']
        self.target_view = conf['target_view']
        self.num_of_rays = conf['ray_batch']
        self.t_near, self.t_far, self.samples_on_ray = conf['hier_samp_conf'].values()
        self.bbox = conf['bbox']
        # type of dataset train, val, test
        self.dat_type = dat_type
    
    def __len__(self):
        return len(self.instance_paths)

    def __getitem__(self,idx):
        instance_path = self.instance_paths[idx]
        data = np.load(instance_path)
        # images need casting to float, since are of uint8 type 
        # due to the fact that ToTensor tranform can not be applied
        # to batch of images, we use division with 255 sice this is 
        # what in general ToTensor does 
        images = torch.tensor(data['rgb'][...,:3]).div(255).to(torch.float32)   
        images = (images-0.5)/0.5
        #images = 0.5*images + 0.5
        nviews, H, W, _ = images.shape
        poses = torch.tensor(data['pose'], dtype=torch.float32) @ \
                self._coord_trans
        focal = torch.tensor(data['focal'], dtype=torch.float32)
        item = {}
        item['poses'] = poses
        item['focal'] = focal
        item['prcpal_p'] = torch.tensor([H/2, W/2])
        # random selection of input viewsi
        if self.input_views==None:
            indices = torch.randperm(nviews)
            ind = indices[:self.num_in_views]
            rest_of_views = torch.sort(indices[self.num_in_views:])
        else:
            indices = torch.arange(nviews).tolist()
            ind = torch.tensor(self.input_views)
            rest_of_views = torch.tensor([
                indices.pop(index) 
                for index in ind
                ])

        input_views = images[ind,...].permute(0,3,1,2) # num_in_viewsx3HxWi
        item['input_views'] = input_views
        input_c2w_poses = poses[ind, ...]
        # w2c matrices computation of input views
        w2crotmatrix = input_c2w_poses[...,:3,:3].permute(0,2,1)
        w2ctranslation = (-w2crotmatrix @ input_c2w_poses[...,:3,3].unsqueeze(2))#[...,None]
        item['w2cpose'] = torch.cat((w2crotmatrix, w2ctranslation), dim=-1)
       
        if self.dat_type != 'test':
            # bounding box estimation #
            # each instance consists of a number of images from 
            # various views. Each image contains an object in a
            # white (255) background, thus subtraction of image
            # with 255 
            im_index, rows, _ , _ = torch.where(images!=1)
            indices = im_index.unique(return_counts=True)[1]
            r_ind_max = indices.cumsum(dim=0)-1
            r_ind_min = torch.cat(
                (torch.tensor(0).unsqueeze(0),
                indices.cumsum(dim=0)[:-1])
            )
            # im_index corresponds to each image index inside pertinent instance
            # rows correpond to rows of elements that satisfy torch.where condition
            # since values are being returned in a row major order, rows vector 
            # across the different splits defined by indices contain sorted values,
            # hence all we to do is to extract first and last value from each split
            # in order to obtain min and max row value for each view bounding box 

            # in order to perform same operation for columns of bounding box we
            # traspose matrix images, in order to exploit the row major order 
            # previously mentioned for the columns
            im_index, cols, _ , _ = torch.where(images.permute(0,2,1,3)!=1)
            indices = im_index.unique(return_counts=True)[1]
            c_ind_max = indices.cumsum(dim=0)-1
            c_ind_min =torch.cat(
                (torch.tensor(0).unsqueeze(0),
                indices.cumsum(dim=0)[:-1])
            )
            bound_boxes = torch.cat( 
                (rows[r_ind_min][:,None],
                rows[r_ind_max][:,None],
                cols[c_ind_min][:,None],
                cols[c_ind_max][:,None]),
                dim=1
            )
            rays_dir, rays_origin = ray_tracing(
                                                [H, W],
                                                focal,
                                                poses
            )
            

            if self.bbox and self.dat_type=='train':
                box_ind = torch.randint(nviews, (self.num_of_rays,))
                rows = (bound_boxes[box_ind, 0] + torch.rand(self.num_of_rays)*(bound_boxes[box_ind, 1] + 1 - bound_boxes[box_ind, 0])).long()
                cols = (bound_boxes[box_ind, 2] + torch.rand(self.num_of_rays)*(bound_boxes[box_ind, 3] + 1 - bound_boxes[box_ind, 2])).long()
            else:
                # in case of sampling both inside and outside of the bounding boxes
                box_ind = torch.randint(nviews, (self.num_of_rays,))
                rows = 0 + (torch.rand(self.num_of_rays)*((W-1) + 1 - 0)).long()
                cols = 0 + (torch.rand(self.num_of_rays)*((H-1) + 1 - 0)).long()
                #prob = torch.randint(0, nviews* H * W, (self.num_of_rays,))
                # number of view
                #box_ind = np.ceil(prob/(H*W)).long()-1
                # number of row inside view
                #rows = np.ceil((prob-(box_ind*H*W))/W).long()-1
                #cols = prob - box_ind*H*W - rows*W -1







            rays_dir = rays_dir[
                            rows*W+cols, 
                            box_ind, 
                            :
                            ]

            item['rays_origin'] = rays_origin[box_ind, :]

            samples, ti = strf_sampling(
                self.samples_on_ray,
                rays_dir,
                rays_origin[box_ind,:],
                self.t_near,
                self.t_far
            )
            item['samples'] = samples
            item['ti'] = ti

            item['rays_dir'] = rays_dir.unsqueeze(1).repeat_interleave(
                self.samples_on_ray,
                dim=1
            )

            item['target'] = 0.5*images[ 
                            box_ind,
                            rows,
                            cols,
                            :3
                            ] + 0.5
        else:
            # test dataset case
            # select one view as reconstruction target view
            # index = rest_of_views[ torch.randint(len(rest_of_views), ()) ]
            if self.target_view is None:
                index = torch.randint(len(rest_of_views),())
            else:
                index = self.target_view 
            item['target'] = images[index,...,:3]
            rays_dir, rays_origin = ray_tracing(
                                                [H, W],
                                                focal,
                                                poses[index][None,...]
            )
            item['rays_origin'] = rays_origin
            samples, ti = strf_sampling(
                self.samples_on_ray,
                rays_dir,
                rays_origin[:, None, None, :],
                self.t_near,
                self.t_far
            )
            item['samples'] = samples
            item['ti'] = ti
            item['rays_dir'] = rays_dir.repeat_interleave(
                self.samples_on_ray,
                dim=1
            )
            item['target'] = images[rest_of_views,...,:3]
            
        return item


class NMRDataset(Dataset):
    def __init__(self,path,transform):
        pass
    def __len__(self):
        pass
    def __getitem__(self,idx):
        pass

class DTUDataset(Dataset):
    def __init__(self,path,transform):
        pass
    def __len__(self):
        pass
    def __getitem__(self,idx):
        pass

