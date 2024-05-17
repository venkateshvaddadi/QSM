#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 19:45:30 2021

@author: venkatesh
"""


#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd

#%%
import os
print(os.getcwd())




#%%
# experiments_folder="savedModels/QSMnet/"
# experiment_name="Jul_04_08_53_pm_model_B_2_N_16800/"

experiments_folder="savedModels/MODL_QSM_MODELS_dw_UNet_loss_l1/experiments_on_given_data/dw_WideResNet/"
experiment_name="Dec_02_06_12_pm_model_K_1_B_2_N_2000_data_source_1/"


path=experiments_folder+experiment_name+'/loss.mat'
loss = scipy.io.loadmat(path)
print(loss.keys())

Training_loss=loss['train_loss']
Validation_loss=loss['valid_loss']

Training_loss=np.array(Training_loss).reshape(-1,)
Validation_loss=np.array(Validation_loss).reshape(-1,)


plt.plot((Training_loss),label='Training')
plt.plot((Validation_loss),label='Validation')

plt.xlabel('epochs')
plt.ylabel('Train Loss ') 
plt.legend(loc="upper right") 


#%%
print('validation sorted indices:',np.argsort(Validation_loss))


#%%


path=experiments_folder+experiment_name+'/model_details.csv'
df = pd.read_csv(path)
#print(df)
val_sorted_indices=df['valid_loss'].to_numpy().argsort()
print('valiation_loss_sorted_indices:')
print(val_sorted_indices)
