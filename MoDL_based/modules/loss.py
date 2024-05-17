#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:50:15 2020

@author: cds

This snippet has been taken from https://github.com/jaywonchung/CAD-QSMNet/blob/master/loss.py

"""


import math
import torch
import torch.nn.functional as F

import numpy as np
import torch



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


def sobel_kernel():
    s = [
        [
            [1,   2,   1],
            [2,   4,   2],
            [1,   2,   1]
        ],
        [
            [0,   0,   0],
            [0,   0,   0],
            [0,   0,   0]
        ],
        [
            [-1, -2, -1],
            [-2, -4, -2],
            [-1, -2, -1]
        ]
    ]
    s = torch.FloatTensor(s)
    sx = s
    sy = s.permute(1, 2, 0)
    sz = s.permute(2, 0, 1)
    ss = torch.stack([sx, sy, sz]).unsqueeze(1)

    return ss




def total_loss(chi, y, b, d, m, b_mean, b_std, y_mean, y_std, sobel):    
    
    # chi = predicetd susc
    # y   = cosmos susc
    # b   = phs
    # d   = dipole kernel
    # m   = mask
    # y_mean = label mean
    # y_std  = label std
    # b_mean = input_mean
    # b_std  = input_std
    
    def _l1error(x1, x2):
        return torch.mean(torch.abs(x1 - x2))


    def _chi_to_b(chi, b, d, m, b_mean, b_std, y_mean, y_std):
        
        # chi = predicetd susc
        # b   = phs
        # d   = dipole kernel
        # m   = mask
        # y_mean = label mean
        # y_std  = label std
        # b_mean = input_mean
        # b_std  = input_std
        
        
        # Restore from normalization
        
        chi = chi * y_std + y_mean
        b   = b   * b_std + b_mean
    
        # Multiply dipole kernel in Fourier domain
        chi_fourier = torch.rfft(chi, signal_ndim=3, onesided=False)
        #print(d.size())
        #print(chi_fourier.size())
        b_hat_fourier_real = (chi_fourier[:,:,:,:,:,0] * d).unsqueeze(dim=5)
        b_hat_fourier_imag = (chi_fourier[:,:,:,:,:,1] * d).unsqueeze(dim=5) 
        
        b_hat_fourier =torch.cat((b_hat_fourier_real,b_hat_fourier_imag), dim=5)
        b_hat =       torch.irfft(b_hat_fourier, signal_ndim=3, onesided=False)
    
        # Multiply masks
        b = b * m
        b_hat = b_hat * m    
        return b, b_hat
    
    def loss_l1(chi, y):
        return _l1error(chi, y)


    def loss_model(b, b_hat):    
        return _l1error(b, b_hat)
    
    def loss_gradient(b, b_hat, chi, y, sobel):
        #difference1 = F.conv3d(b - b_hat,   sobel, padding=1)
        #difference2 = F.conv3d(y - chi  ,   sobel, padding=1)
        difference   = F.conv3d(y - chi  ,   sobel, padding=1)
        return torch.mean(torch.abs(difference))
    
    w1 = 0.5
    w2 = 1
    w3 = 0.1
     
    b, b_hat = _chi_to_b(chi, b, d, m, b_mean, b_std, y_mean, y_std)
    
    loss_model = w1 * loss_model(b, b_hat)
    
    loss_l1    = w2 * loss_l1(chi, y)
    
    loss_grad  = w3 * loss_gradient(b, b_hat, chi, y, sobel)
    
    loss = loss_l1 + loss_model + loss_grad
    
    #loss = loss_model 
    #loss = loss_l1 
    #loss = loss_grad    
    return loss
    



