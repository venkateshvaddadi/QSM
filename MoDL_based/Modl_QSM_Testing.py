#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:24:27 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:44:32 2021

@author: venkatesh
"""




#%%
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from torch import nn



import torch
import numpy as np
import torch.optim as optim
import time
import tqdm
import scipy.io
import matplotlib.pyplot as plt
import os
from datetime import datetime

#%%

import sys
sys.path.insert(0,'modules/' )

from modules.QSMnetDataset_updated import mydataloader
from modules.utils import *


from modules.loss_updated import *
#from modules.model_for_dw import Dw
from modules.config import Config
from modules.model_for_dw_unet import QSMnet
from modules.model_for_dw_unet import QSMnet
from modules.model_for_dw_deepqsm import DeepQSM
from modules.model_for_dw_simple_cnn import Dw
from modules.WideResnet import WideResNet

#%%

matrix_size = [176,176, 160]
voxel_size = [1,  1,  1]

#%%
import os
print(os.getcwd())


#%%
#loading the model\
is_data_normalized=True
K_unrolling=3
epoch=27
device_id=0
#%%
experiments_folder="savedModels/MODL_QSM_MODELS_dw_UNet_loss_l1/experiments_on_given_data/dw_WideResNet/"
experiment_name="Dec_02_06_12_pm_model_K_1_B_2_N_2000_data_source_1/"
model_name="Modl_QSM_model_"+str(epoch)+"_.pth"
model_path=experiments_folder+"/"+experiment_name+"/"+model_name
print('model_path:',model_path)
#%%
model='WideResNet'

if(model=='deepqsm'):
    dw = DeepQSM().cuda(device_id)
elif(model=='QSMnet'):
    dw=QSMnet().cuda(device_id)
elif(model=='WideResNet'):
    dw=WideResNet().cuda(device_id)
elif(model=='simple_cnn'):
    dw=Dw().cuda(device_id)

dw.load_state_dict(torch.load(model_path))
dw.eval()

#%%
print('dw_lambda_val',dw.lambda_val.item())
print("Evaluation happening")
#%%
   

last_string=model_path.split("/")[-1]
directory=model_path.replace(last_string,"")
print('directory:',directory)
#%%
#%%

data_source='given_data'
Training_patient_no=4
data_source_no=1

if(data_source=='generated_data'):

    raw_data_path='../../QSM_data/data_for_experiments/generated_data/raw_data_noisy_sigma_0.1/'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1/'
    #data_path='../QSM_data/data_for_experiments/generated_data/single_patient_patches/patient_'
    patients_list =[7,32,9,10]

elif(data_source=='generated_noisy_data'):
    raw_data_path='../../QSM_data/data_for_experiments/generated_data/raw_data_noisy_sigma_0.01/patient_'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1_sigma_0.01/'
    patients_list =[7,32,9,10]


elif(data_source=='generated_undersampled_data'):
    raw_data_path='../../QSM_data/data_for_experiments/generated_data/sampling_data/sampled_0.2/patient_'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1/'
    patients_list =[7,32,9,10]





elif(data_source=='given_data'):

    raw_data_path='../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
    if(data_source_no==1):
        patients_list =[7,8,9,10,11,12]
        data_path='../../QSM_data/data_for_experiments/given_data/data_source_1/'
        csv_path='../../QSM_data/data_for_experiments/given_data/data_source_1//'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'


    elif(data_source_no==2):
        patients_list =[10,11,12,1,2,3]
        data_path='../../QSM_data/data_for_experiments/given_data/data_source_2/'
        csv_path='../../QSM_data/data_for_experiments/given_data/data_source_2//'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==3):
        patients_list =[1,2,3,4,5,6]
        data_path='../../QSM_data/data_for_experiments/given_data/data_source_3/'
        csv_path='../../QSM_data/data_for_experiments/given_data/data_source_2//'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==4):
        patients_list =[4,5,6,7,8,9]
        data_path='../../QSM_data/data_for_experiments/given_data/data_source_4/'
        csv_path='../../QSM_data/data_for_experiments/given_data/data_source_2//'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'




elif(data_source=='given_single_patient_data'):
    raw_data_path='../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
    if(Training_patient_no==1):
        data_path='../../QSM_data/data_for_experiments/given_data/single_patient/patient_1/'
        patients_list =[2,3,4,5,7,8,9,10,11,12]
    elif(Training_patient_no==2):
        data_path='../../QSM_data/data_for_experiments/given_data/single_patient/patient_2/'
        patients_list =[1,3,4,5,7,8,9,10,11,12]

    elif(Training_patient_no==3):
        data_path='../../QSM_data/data_for_experiments/given_data/single_patient/patient_3/'
        patients_list =[1,2,4,5,7,8,9,10,11,12]
    elif(Training_patient_no==4):
        data_path='../../QSM_data/data_for_experiments/given_data/single_patient/patient_4/'
        patients_list =[1,2,3,5,7,8,9,10,11,12]

import os
print(os.listdir(data_path))
print(os.listdir(raw_data_path))



import os
print(os.listdir(data_path))




import os
print(os.listdir(data_path))
#%%
#%%

# testdata = mydataloader(data_path+'/csv_files/test.csv', data_path+'/Testing_Data',training=False)
# testloader = DataLoader(testdata, batch_size = 1, shuffle=False, num_workers=1)
# print('len(testloader)',len(testloader))
# print("-"*100)
#%%
if(is_data_normalized):
    stats = scipy.io.loadmat(csv_path+'/csv_files/tr-stats.mat')
    sus_mean= stats['out_mean'][0][0]
    sus_std = stats['out_std' ][0][0]
else:
    sus_mean=0
    sus_std=1
    print("Data was not normalized...")
    print(sus_mean,sus_std)
    
#%%


config  = Config()

# we need to write the code for the training here........

if config.gpu==True:
    dw = dw.cuda(device_id)





#%%
dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk = torch.unsqueeze(dk, dim=0)
print(dk.shape)

dk=dk.float().cuda(device_id)
Dk_square=torch.multiply(dk, dk)
Dk_square=Dk_square.cuda(device_id)



#%%
def b_gpu(y,lambda_val, z_k):
    
    #print('\t \t  calling b_gpu:')
    #print('y.shape:',y.shape)
    #print('z_k.shape:',z_k.shape)

    #print(y)
    #print(lambda_val)
    #print(z_k)

    output1 = torch.fft.fftn(y)
    output2 = dk * output1
    output3 = torch.fft.ifftn(output2)
    output3 = torch.real(output3)
    
    #print('output3.get_device:', output3.get_device())
    #print('lambda_val.get_device:', lambda_val.get_device())
    #print('z_k.get_device:',z_k.get_device())
    
    output4 = output3+lambda_val*z_k

    return output4

# x sshould be in gpu....
    
def A_gpu(x,lambda_val):
    #print('\t \t calling A')
    #print('---------------------')
    output1 = Dk_square*torch.fft.fftn(x)
    output2 = torch.fft.ifftn(output1)
    output2 = torch.real(output2)

    output3 = output2+lambda_val * x
    
    return output3

def CG_GPU(local_field_gpu, z_k_gpu):

    #print('CG GPU Calling............')
    #print('--------------------------')
    
    x_0 = torch.zeros(size=(1, 1, 176, 176, 160)).cuda(device_id)

    r_0 = b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)-A_gpu(x_0,dw.lambda_val)
    
    #print('r_0.shape',r_0.shape)
    p_0 = r_0

    #print('r_0.shape', r_0.shape)
    #print('P_0 shape', p_0.shape)

    r_old = r_0
    p_old = p_0
    x_old = x_0

    r_stat = []
    
    r_stat.append( torch.sum(r_old * r_old).item())

    for i in range(25):

        # alpha calculation
        r_old_T_r_old = torch.sum(r_old * r_old)
        #print('\t r_old_T_r_old',r_old_T_r_old,r_old_T_r_old.shape)

        if(r_old_T_r_old.item()<1e-10):
            #print('r_stat',r_stat,'iteration:',len(r_stat))
            return x_old
        
        
        if(r_old_T_r_old>r_stat[-1] and r_stat[-1] < 1e-06):
            #print("Convergence issue:",r_old_T_r_old,r_stat[-1])
            return x_old

        r_stat.append( torch.sum(r_old * r_old).item())

        #print(r_stat)
        
        Ap_old = A_gpu(p_old,dw.lambda_val)
        #print('\t Ap_old.shape',Ap_old.shape)
        
        p_old_T_A_p_old = torch.sum(p_old * Ap_old)
        #print('\t p_old_T_A_p_old',p_old_T_A_p_old)
        
        alpha = r_old_T_r_old/p_old_T_A_p_old
        #print('\t alpha',alpha)

        # updating the x
        x_new = x_old+alpha*p_old
        #print('\t x_new',x_new.shape)
        

        # updating the remainder
        r_new = r_old-alpha*Ap_old
        #print('\t r_new.shape',r_new.shape)
        
        # beta calculation
        r_new_T_r_new = torch.sum(r_new * r_new)

        
        beta = r_new_T_r_new/r_old_T_r_old

        # new direction p calculation
        p_new = r_new+beta*p_old

        # preparing for the new iteration...

        r_old = r_new
        p_old = p_new
        x_old = x_new
        

   # print(x_new.shape)

    return x_new


#%%

outdir = directory+'predictions_'+str(epoch)+"/"
print(outdir)
import os
try:
    os.makedirs(outdir)
except:
    print("aalready tested..")

#%%
with torch.no_grad():

    for i in patients_list:
        print("Patinte:"+str(i)+"\n")
        for j in range(1,6):
            print("orientation:",j)
            phs=scipy.io.loadmat(raw_data_path+"patient_"+str(i)+'/phs'+str(j)+'.mat')['phs']
            sus=scipy.io.loadmat(raw_data_path+"patient_"+str(i)+'/cos'+str(j)+'.mat')['cos']
            msk=scipy.io.loadmat(raw_data_path+"patient_"+str(i)+'/msk'+str(j)+'.mat')['msk']
            
            
            phs=torch.unsqueeze(torch.unsqueeze(torch.tensor(phs),0),0)
            sus=torch.unsqueeze(torch.unsqueeze(torch.tensor(sus),0),0)
            msk=torch.unsqueeze(torch.unsqueeze(torch.tensor(msk),0),0)
    
            phs=phs.cuda(device_id).float()
            sus=sus.cuda(device_id).float()
            msk=msk.cuda(device_id)

            phs_F = torch.fft.fftn(phs,dim=[2,3,4])
            phs_F = dk * phs_F
                        
            x_0 = torch.real(torch.fft.ifftn(phs_F,dim=[2,3,4]))
                                    
            print(sus_mean,sus_std)
                        
            x_k=x_0
            for k in range(K_unrolling):
              x_k=(x_k-sus_mean)/sus_std
              z_k=dw(x_k)
              z_k=z_k*sus_std+sus_mean
              x_k=CG_GPU(phs, z_k)
            
            x_k_cpu=(x_k.detach().cpu().numpy())*(msk.detach().cpu().numpy() )
            
            
            mdic  = {"modl" : x_k_cpu}
            filename  = outdir + 'modl-net-'+ str(i)+'-'+str(j)+'.mat'
            scipy.io.savemat(filename, mdic)

        #print(phs.shape)
    
    
#%%

# Taking back to CPU
x_k_cpu=(x_k.detach().cpu().numpy())*(msk.detach().cpu().numpy() )
sus_cpu=sus.detach().cpu().numpy()

#%%


import matplotlib.pyplot as plt



fig=plt.figure(figsize=(20,20))

plot=fig.add_subplot(1,2,1)
plt.imshow(x_k_cpu[0][0][:,:,80],cmap='gray', clim=(-0.1,0.1))
plot.set_title('Generated')
plot=fig.add_subplot(1,2,2)
plt.imshow(sus_cpu[0][0][:,:,80],cmap='gray', clim=(-0.1,0.1))
plot.set_title('COSMOS')
            
#%%
