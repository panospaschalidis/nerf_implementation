"""standard libraries"""
import argparse
import os
import sys
import glob
import torch
import pandas as pd
import numpy as np
import pdb
import random
import matplotlib.pyplot as plt
import yaml


"""third party imports"""
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser(
    description = "Radiance Fields mesh extraction process"
)
parser.add_argument(
    "--model_path",
    default="checkpoints/exalted-bush-58.4900.pth",
    help="path to given checkpoint"
)

args = parser.parse_args()
## repalce model_path with your stored model path
model = torch.load(args.model_path, map_location=('cpu'))['model']
device = 'cpu'
N = 64
t = torch.linspace(-1.2, 1.2, N+1)

query_pts = torch.stack(torch.meshgrid(t, t, t, indexing='xy'), -1).float()
print(query_pts.shape)
sh = query_pts.shape
flat = query_pts.reshape([-1,3])
dirs = torch.zeros_like(flat)
chunk = 1024*64
density= [] 
with torch.no_grad():
    for i in range(0, flat.shape[0], chunk):
        density.append(model(flat[None, None,i:i+chunk,...], dirs[None, None,i:i+chunk,...])[0])

density = torch.cat(density, -1).squeeze()
density = density.reshape(*sh[:3],-1)
density = torch.maximum(density[...,-1],torch.tensor(0))
# plt.hist(torch.maximum(torch.tensor(0), density.ravel()), log=True)
import mcubes
threshold = 50. 
print('fraction occupied', torch.mean((density > threshold).to(torch.float)))
vertices, triangles = mcubes.marching_cubes(np.asarray(density), threshold)
print('done', vertices.shape, triangles.shape)


import trimesh

mesh = trimesh.Trimesh(vertices / N - .5, triangles)
mesh_out_file = os.path.join('extracted_meshes', 
                            os.path.splitext(
                                os.path.basename(
                                    args.model_path))[0]+"_"+str(threshold)+".ply")
trimesh.exchange.export.export_mesh(mesh, mesh_out_file)
mesh.show()

