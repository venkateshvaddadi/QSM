#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:56:39 2021

@author: venkatesh
"""


import numpy as np

import torch

#%%
def dipole_kernel(matrix_size, voxel_size, B0_dir=[0,0,1]):
    [Y,X,Z] = np.meshgrid(np.linspace(-np.int(matrix_size[1]/2),np.int(matrix_size[1]/2)-1, matrix_size[1]),
                       np.linspace(-np.int(matrix_size[0]/2),np.int(matrix_size[0]/2)-1, matrix_size[0]),
                       np.linspace(-np.int(matrix_size[2]/2),np.int(matrix_size[2]/2)-1, matrix_size[2]))
    X = X/(matrix_size[0])*voxel_size[0]
    Y = Y/(matrix_size[1])*voxel_size[1]
    Z = Z/(matrix_size[2])*voxel_size[2]
    D = 1/3 - np.divide(np.square(X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2]), np.square(X)+np.square(Y)+np.square(Z) + np.finfo(float).eps )
    D = np.where(np.isnan(D),0,D)

    D = np.roll(D,np.int(np.floor(matrix_size[0]/2)),axis=0)
    D = np.roll(D,np.int(np.floor(matrix_size[1]/2)),axis=1)
    D = np.roll(D,np.int(np.floor(matrix_size[2]/2)),axis=2)
    D = np.float32(D)
    D = torch.tensor(D).unsqueeze(dim=0)
    
    return D

#%%





#%%
def dipole_kernel_TKD(matrix_size, voxel_size, B0_dir=[0,0,1]):
    [Y,X,Z] = np.meshgrid(np.linspace(-np.int(matrix_size[1]/2),np.int(matrix_size[1]/2)-1, matrix_size[1]),
                       np.linspace(-np.int(matrix_size[0]/2),np.int(matrix_size[0]/2)-1, matrix_size[0]),
                       np.linspace(-np.int(matrix_size[2]/2),np.int(matrix_size[2]/2)-1, matrix_size[2]))
    X = X/(matrix_size[0])*voxel_size[0]
    Y = Y/(matrix_size[1])*voxel_size[1]
    Z = Z/(matrix_size[2])*voxel_size[2]
    D = 1/3 - np.divide(np.square(X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2]), np.square(X)+np.square(Y)+np.square(Z) + np.finfo(float).eps )
    D = np.where(np.isnan(D),0,D)
    
    D = np.roll(D,np.int(np.floor(matrix_size[0]/2)),axis=0)
    D = np.roll(D,np.int(np.floor(matrix_size[1]/2)),axis=1)
    D = np.roll(D,np.int(np.floor(matrix_size[2]/2)),axis=2)
    D = np.float32(D)
    # D = torch.tensor(D).unsqueeze(dim=0)
    
    D[D>0.2]=0.2
    D[D<-0.2]=-0.2
    D = torch.tensor(D).unsqueeze(dim=0)
    return D
#%%

matrix_size = [64, 64, 64]
voxel_size = [1,  1,  1]
B0_dir=[0,0,1]
D=dipole_kernel(matrix_size,voxel_size,B0_dir)
