#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 11:41:13 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:37:27 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:51:56 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:43:36 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 12:32:23 2021

@author: venkatesh
"""


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
from modules.utils import *


#%%

from modules.QSMnetDataset_updated import mydataloader
from modules.loss_updated import *
from modules.config import Config
from modules.model_for_dw_unet import QSMnet
from modules.model_for_dw_deepqsm import DeepQSM
from modules.model_for_dw_simple_cnn import Dw
from modules.WideResnet import WideResNet

#%%


# paramaters for dipole kernel
matrix_size = [64, 64, 64]
voxel_size = [1,  1,  1]



#for restoring weights
restore=True
K_unrolling=3
batch_size=2
epoch=0
device_id=0
is_data_normalized=True
break_amount=1000
model='WideResNet'

#%%
# making directory for sving models
print ("-"*120)
start_time=time.time()
experiments_folder="savedModels/MODL_QSM_MODELS_dw_UNet_loss_l1/experiments_on_given_data//dw_WideResNet/"
experiment_name=datetime.now().strftime("%b_%d_%I_%M_%P_")+"model_K_"+ str(K_unrolling)+"_B_"+str(batch_size)+"_N_"+str(batch_size*break_amount)
cwd=os.getcwd()
directory=experiments_folder+"/"+experiment_name+"/"
print(directory)
print('Model will be saved to  :', directory)
try:
    os.makedirs(directory)
except:
    print("Exception...")


#%%


data_source='given_data'
Training_patient_no=4
data_source_no=4

if(data_source=='generated_data'):

    raw_data_path='../../QSM_data/data_for_experiments/generated_data/raw_data/'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1/'

elif(data_source=='generated_noisy_data'):
    raw_data_path='../../QSM_data/data_for_experiments/generated_data/raw_data_noisy_sigma_0.01/'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1_sigma_0.01/'

elif(data_source=='given_data'):

    raw_data_path='../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

    if(data_source_no==1):
        patients_list =[7,8,9,10,11,12]
        data_path='../../QSM_data/data_for_experiments/given_data/data_source_1/'

    elif(data_source_no==2):
        patients_list =[10,11,12,1,2,3]
        data_path='../../QSM_data/data_for_experiments/given_data/data_source_2/'

    elif(data_source_no==3):
        patients_list =[1,2,3,4,5,6]
        data_path='../../QSM_data/data_for_experiments/given_data/data_source_3/'

    elif(data_source_no==4):
        patients_list =[4,5,6,7,8,9]
        data_path='../../QSM_data/data_for_experiments/given_data/data_source_4/'

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

        
elif(data_source=='generated_single_patient_data'):
        data_path='../../QSM_data/data_for_experiments/generated_data/single_patient_patches/patient_2/'

import os
print(os.listdir(data_path))
print(os.listdir(raw_data_path))
print('data_path:',data_path)




#%%

# data loading

loader = mydataloader(data_path+'/csv_files/train.csv', data_path+'/Training_Data/')
trainloader = DataLoader(loader, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)

valdata    = mydataloader(data_path+'/csv_files/val.csv', data_path+'/Validation_Data/')
valloader  = DataLoader(valdata, batch_size = batch_size, shuffle=True, num_workers=1,drop_last=True)

print('no of training batches',len(trainloader))
print('no of validation batches',len(valloader))

print("-"*120)

if(is_data_normalized):
    stats = scipy.io.loadmat(data_path+'/Training_Data/tr-stats.mat')
    sus_mean= stats['out_mean'][0][0]
    sus_std = stats['out_std' ][0][0]
else:
    print("Data was not normalized...")
    sus_mean=0
    sus_std=1
    print(sus_mean,sus_std)


#%%





# making a log file........
import logging

logging.basicConfig(filename=directory+'app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
logging.info('batch_size: '+str(batch_size))
logging.info('K_unrolling: '+str(K_unrolling))

logging.info('is_dara_normalized: '+str(is_data_normalized))
logging.info('restore: '+str(restore))

logging.info('data_path:'+data_path)
logging.info('model_name:'+directory)

logging.info('no of training patches: '+str(len(trainloader)))
logging.info('no of validation patches: '+str(len(valloader)))

logging.warning('data_source:'+str(data_source))
logging.warning('data_path:'+(data_path))


#%%
# making a dipole kenel..........


dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk = torch.unsqueeze(dk, dim=0)
#print(dk.shape)

dk=dk.cuda(device_id)
Dk_square=dk * dk

ss = sobel_kernel()
ss = ss.cuda(device_id)
print(ss.shape)

#%%



# we need to write the code for the training here........



if(model=='deepqsm'):
    dw = DeepQSM().cuda(device_id)
elif(model=='QSMnet'):
    dw=QSMnet().cuda(device_id)
elif(model=='WideResNet'):
    dw=WideResNet().cuda(device_id)
elif(model=='simple_cnn'):
    dw=Dw().cuda(device_id)
#%%
dw1 = torch.nn.DataParallel(dw, device_ids=[device_id])  
print("we have defined the model in the GPU")




if restore:
    restore_weights_path='savedModels/MODL_QSM_MODELS_dw_UNet_loss_l1/experiments_on_given_data/dw_WideResNet/Dec_17_07_28_pm_model_K_1_B_2_N_2000_data_source_4/Modl_QSM_model_31_.pth'
    dw.load_state_dict(torch.load(restore_weights_path))
    print("we have restored weights of the model:",restore_weights_path)
    
    
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
    
    x_0 = torch.zeros(size=(1, 1, 64, 64, 64)).cuda(device_id)

    r_0 = b_gpu(local_field_gpu, dw.lambda_val,z_k_gpu)-A_gpu(x_0,dw.lambda_val)
    
    #print('r_0.shape',r_0.shape)
    p_0 = r_0

    #print('r_0.shape', r_0.shape)
    #print('P_0 shape', p_0.shape)

    r_old = r_0
    p_old = p_0
    x_old = x_0

    r_stat = []
    
    r_stat.append(torch.sum(r_old * r_old).item())


    for i in range(10):

        # alpha calculation
        r_old_T_r_old = torch.sum(r_old * r_old)
        #print('\t r_old_T_r_old',r_old_T_r_old,r_old_T_r_old.shape)


        if(r_old_T_r_old<1e-10):
            # print('r_stat',r_stat,r_old_T_r_old.item(),'iteration:',len(r_stat))
            
            # logging.warning('r_stat')
            # logging.warning(r_stat)
            # logging.warning(r_old_T_r_old.item())
            # logging.warning('iteration:'+str(len(r_stat)))
            
            return x_old
        
        
        if(r_old_T_r_old.item()>r_stat[-1] and r_stat[-1] < 1e-06):
            print("Convergence issue:",r_old_T_r_old.item(),r_stat[-1])
            
            logging.warning("Convergence issue:")
            logging.warning(r_old_T_r_old.item())
            logging.warning(r_stat[-1])
            return x_old


        r_stat.append( torch.sum(r_old * r_old).item())

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

        r_stat.append(torch.norm(r_new, p=2).item())
        
        beta = r_new_T_r_new/r_old_T_r_old

        # new direction p calculationubu 
        p_new = r_new+beta*p_old

        # preparing for the new iteration...

        r_old = r_new
        p_old = p_new
        x_old = x_new

   # print(x_new.shape)

    return x_new
#%%

optimizer = optim.Adam(dw.parameters(), lr=1e-4)

loss_Train=[]
loss_Val=[]
lambda_val=[]

# k is for unrolling 
epoch=0
#%%
for epoch in range(100):
    runningLoss = 0
    dw.train() 
    print('epoch---',epoch)


    for i, data in tqdm.tqdm(enumerate(trainloader)):
            if(i==break_amount):
                break
            phs, msk, sus,file_name = data

            dk_repeat = dk.repeat(batch_size,1,1,1,1)
            phs=phs.cuda(device_id)
            msk=msk.cuda(device_id)
            sus=sus.cuda(device_id)
            
            
            # initialization ....
            
            phs_F = torch.fft.fftn(phs,dim=[2,3,4])
            phs_F = dk_repeat * phs_F
            
            x_0 = torch.real(torch.fft.ifftn(phs_F,dim=[2,3,4]))
                        
            # print(sus_mean,sus_std)
            x_k=x_0
            x_k = x_k * msk

            for k in range(K_unrolling):
                x_k=(x_k-sus_mean)/sus_std
                z_k = dw1(x_k)
                z_k=z_k*sus_std+sus_mean
                solutions=[]
                for j in range(batch_size):
                    cg_output=CG_GPU(phs[j,:,:,:,:].unsqueeze(0),z_k[j,:,:,:,:].unsqueeze(0))
                    solutions.append(cg_output)
                    
                
                x_k_new=torch.cat(solutions,dim=0)
                x_k=x_k_new
                x_k = x_k * msk
                
            x_k = x_k * msk
            
            
            #print("Shape",x_k.shape)
            
            #print('x_k.shape',x_k.shape)
            optimizer.zero_grad()
            #loss=loss_fn(x_k,sus)
            loss=total_loss_l1(chi=x_k, y=sus, b=phs, d=dk, m=msk, sobel=ss)
            loss.backward()

            nn.utils.clip_grad_value_(dw.parameters(), clip_value=1.0)

            optimizer.step()
            #print(dw.lambda_val,dw.lambda_val.grad)
            runningLoss += loss.item()

    import time
    loss_Train.append(runningLoss/len(trainloader))
    print("_"*120)
    print('Training_loss:', loss_Train)    
    print("Lambda : ", dw.lambda_val.item())

    torch.save(dw.state_dict(), directory+'Modl_QSM_model_'+str(epoch)+'_.pth')
    time.sleep(10)

    # validation code is here
    running_val_Loss=0
    dw.eval()
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(valloader)):
            try:
                phs, msk, sus,file_name = data
                dk_repeat = dk.repeat(batch_size,1,1,1,1)

                phs=phs.cuda(device_id)
                msk=msk.cuda(device_id)
                sus=sus.cuda(device_id)

           
                # initialization ....
                
                phs_F = torch.fft.fftn(phs,dim=[2,3,4])
                phs_F = dk_repeat * phs_F
                
                x_0 = torch.real(torch.fft.ifftn(phs_F,dim=[2,3,4]))
                            
                
                x_k=x_0
                
                x_k = x_k * msk
                
                # we are running for the k=10 (unrolling for k=10)
                for k in range(K_unrolling):
                    x_k=(x_k-sus_mean)/sus_std
                    z_k = dw1(x_k)
                    z_k=z_k*sus_std+sus_mean
                    solutions=[]
                    for j in range(batch_size):
                        cg_output=CG_GPU(phs[j,:,:,:,:].unsqueeze(0),z_k[j,:,:,:,:].unsqueeze(0))
                        solutions.append(cg_output)

                    x_k_new=torch.cat(solutions,dim=0)
                    x_k=x_k_new
                    x_k = x_k * msk
                    
                x_k = x_k * msk
    
                loss=total_loss_l1(chi=x_k, y=sus, b=phs, d=dk, m=msk, sobel=ss)
                running_val_Loss += loss.item()
    
            except:
                print('error at',i)

    loss_Val.append(running_val_Loss/len(valloader))
    lambda_val.append(dw.lambda_val.item())

    print('Validation_loss:',loss_Val)
    print('Checking what is the K value:',K_unrolling)
    print("-"*120)

    # saving loss    
    loss_dic={"train_loss":loss_Train,"valid_loss":loss_Val}
    scipy.io.savemat(directory+"/loss.mat", loss_dic)

    import pandas as pd
    model_details={"train_loss":loss_Train,"valid_loss":loss_Val,'lambda':lambda_val}
    df = pd.DataFrame.from_dict(model_details) 

    path=directory+'/model_details.csv'
    df.to_csv (path, index = False, header=True)
    
    logging.info('epoch:'+str(epoch))
    logging.info('training loss:'+str(runningLoss/len(trainloader)))
    logging.info('validation loss:'+str(running_val_Loss/len(valloader)))
    logging.info('lambda:'+str(dw.lambda_val.item()))

#%%

#plotting the loss
import math
plt.plot(np.log(np.array(loss_Train)))
plt.plot(np.log(np.array(loss_Val)))
plt.show()
#%%

#%%
import scipy.io
loss_dic={"train_loss":loss_Train,"valid_loss":loss_Val}
scipy.io.savemat(directory+"/loss.mat", loss_dic)



