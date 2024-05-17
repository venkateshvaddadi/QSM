import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import lr_scheduler


import numpy as np
import torch.optim as optim
import time
import tqdm
import scipy.io
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time


#%%
from modules.dipole_kernel_updaed import dipole_kernel
from modules.QSMnetDataset_updated import mydataloader
from modules.loss_updated import *
from modules.model_for_dw import Dw
from modules.config import Config
from modules.model_for_dw_unet import QSMnet
from modules.deepqsm import DeepQSM
from modules.WideResnet import WideResNet
from modules.qsmnet import QSMnet

#%%
def z_real_to_z_complex(z_real):
  z_complex_recon=torch.complex(z_real[:,0,:,:,:].unsqueeze(1),z_real[:,1,:,:,:].unsqueeze(1))
  # print('\n at z_real_to_z_complex sum',torch.sum(z_real[:,1,:,:,:].unsqueeze(1)))
  
  return z_complex_recon

def z_complex_to_z_real(z_complex):
  z_real=z_complex.real
  z_imag=z_complex.imag
  # print('\n at z_complex_to_z_real sum',torch.sum(z_complex.imag))
  z_real_recon=torch.cat([z_real,z_imag],axis=1)
  return z_real_recon
#%%

K_unrolling=3
device_id=0
batch_size=2
No_samples=16800
epoch=0
epsilon=1e-5
learning_rate=1e-4
break_amount=1000
is_data_normalized=True
restore=False;
model='WideResNet'
loss_function='MSE'



#%%

import os

data_source='given_data'
data_source_no=2


if(data_source=='generated_data'):
    raw_data_path='../../QSM_data/data_for_experiments/generated_data/raw_data/patient_'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1/'
    
elif(data_source=='given_data'):
    raw_data_path='../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
    
    if(data_source_no==1):
        patients_list =[7,8,9,10,11,12]
        csv_path='../../QSM_data/data_for_experiments/given_data/data_source_1//'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==2):
        patients_list =[10,11,12,1,2,3]
        csv_path='../../QSM_data/data_for_experiments/given_data/data_source_2//'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==3):
        patients_list =[1,2,3,4,5,6]
        csv_path='../../QSM_data/data_for_experiments/given_data/data_source_3//'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'

    elif(data_source_no==4):
        patients_list =[4,5,6,7,8,9]
        csv_path='../../QSM_data/data_for_experiments/given_data/data_source_4//'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'


elif(data_source=='generated_noisy_data'):
    raw_data_path='../../QSM_data/data_for_experiments/generated_data/raw_data_noisy_sigma_0.01/'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1_sigma_0.01/'
    patients_list =[7,32,9,10]

 
print(os.listdir(data_path))

import os
print(os.listdir(raw_data_path))
print(os.listdir(data_path))
print('csv_path:',csv_path)
print("data_path:",data_path)
#%%
if(not is_data_normalized):
    sus_mean=0
    sus_std=1
    print('\n\n data is not normalized..................\n\n ')

else:
    stats = scipy.io.loadmat(csv_path+'/csv_files/tr-stats.mat')
    sus_mean= torch.tensor(stats['out_mean']).cuda(device_id)
    sus_std = torch.tensor(stats['out_std' ]).cuda(device_id)
    print(sus_mean,sus_std)


#%%

loader = mydataloader(csv_path+'/csv_files/train.csv', data_path)
trainloader = DataLoader(loader, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)

valdata    = mydataloader(csv_path+'/csv_files/val.csv', data_path)
valloader  = DataLoader(valdata, batch_size = batch_size, shuffle=True, num_workers=1,drop_last=True)

#%%
# making directory for sving models
print ('*******************************************************')
start_time=time.time()
experiments_folder="savedModels/LPCNN_QSM_MODELS/experiments_on_given_data/full_sampling_models/"
experiment_name=datetime.now().strftime("%b_%d_%I_%M_%P_")+"model_K_"+ str(K_unrolling)+"_model_"+model;
cwd=os.getcwd()

directory=experiments_folder+"/"+experiment_name+"/"
print(directory)
print('Model will be saved to  :', directory)

#%%

try:
    os.makedirs(directory)
except:
    print("Exception...")

# making a log file........
import logging

logging.basicConfig(filename=directory+'app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('BATCH SIZE:'+str(batch_size))
logging.warning('K_UNROLLING:'+str(K_unrolling))
logging.warning('restore:'+str(restore))
logging.warning('No of training samples:'+str(len(loader)))
logging.warning('No of Validation samples:'+str(len(valdata)))
logging.warning('No of training batches:'+str(len(trainloader)))
logging.warning('No of Validation batches:'+str(len(valloader)))

logging.warning('learning_rate:'+str(learning_rate))
logging.warning('break_amount:'+str(break_amount))
logging.warning('data_normalized:'+str(is_data_normalized))
logging.warning('model:'+(model))
logging.warning('data_path:'+(data_path))
logging.warning('data_source:'+(data_source))

logging.warning('loss_function:'+(loss_function))
#%%
# paramaters for dipole kernel
matrix_size = [64, 64, 64]
voxel_size = [1,  1,  1]


dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk = torch.unsqueeze(dk, dim=0)
dk=dk.cuda(device_id)

ss = sobel_kernel()
ss = ss.cuda(device_id)
# print(ss.shape)



#%%

if(model=='deepqsm'):
    net = DeepQSM().cuda(device_id)
elif(model=='QSMnet'):
    net=QSMnet().cuda(device_id)
elif(model=='WideResNet'):
    net=WideResNet().cuda(device_id)


net_par = torch.nn.DataParallel(net, device_ids=[device_id ])  














#%%
if(restore):
    restore_path='savedModels/LPCNN_QSM_MODELS/Oct_28_07_19_pm_model_K_3_model_WideResNet_MSE_data_source_patient_2/Spinet_QSM_model_6_.pth'
    net.load_state_dict(torch.load(restore_path))
    logging.warning('restore path:'+str(restore_path))
    print('restore path:'+str(restore_path))


#print(net)

#%%
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

loss_critearia=nn.MSELoss();
#%%
loss_Train=[]
loss_Val=[]
aplha_list=[]
#%%
for epoch in range(25):
    runningLoss = 0

    print('\n-------------------------------------------------------------------------------------------------\n')
    print('epoch---',epoch)

    net.train() 

    for i, data in tqdm.tqdm(enumerate(trainloader)):
            phs, msk, sus,file_name = data
            
            if(i==break_amount):
                break
            
            dk_repeat = dk.repeat(batch_size,1,1,1,1)
            dk_repeat=dk_repeat.cuda(device_id)
            phs=phs.cuda(device_id)
            msk=msk.cuda(device_id)
            sus=sus.cuda(device_id)
            
            # calculating term1=alpha(phi^H)y
            
            term_1_complex = torch.fft.ifftn(dk_repeat*torch.fft.fftn(phs,dim=[2,3,4]),dim=[2,3,4])
            term_1_complex=net.alpha*(term_1_complex)
            term_1_complex=term_1_complex.cuda(device_id)
            # initialize with zeros....
            x_0_complex=torch.zeros_like(term_1_complex)
            x_k_complex=x_0_complex;
            
            
            for k in range(K_unrolling):
                # print(k)
                term_2_complex=torch.fft.ifftn(dk_repeat*dk_repeat*torch.fft.ifftn(x_k_complex,dim=[2,3,4]),dim=[2,3,4])
                
                # calculating (I-(Phi)^H(Phi))x
                term_3_complex=x_k_complex-(net.alpha)*term_2_complex
                
                # calculating alpha*(Phi^H)y+(I-alpha*(Phi)^H(Phi))x
                term_4_complex=term_1_complex+term_3_complex
                term_4_real=z_complex_to_z_real(term_4_complex)
                term_4_real=term_4_real*msk
                
                term_4_real=(term_4_real-sus_mean)/sus_std
                x_k_real=net_par(term_4_real)
                x_k_real=x_k_real*sus_std+sus_mean
                x_k_real=x_k_real*msk
                
                x_k_complex=z_real_to_z_complex(x_k_real)
                x_k_complex=x_k_complex*msk
            x_k_complex = x_k_complex * msk
        
            optimizer.zero_grad()
            #loss=total_loss_l1(chi=x_k_complex.real, y=sus.float(), b=phs, d=dk, m=msk, sobel=ss)
            if(loss_function=='MSE'):
                loss=loss_critearia(x_k_complex.real, sus.float())
            elif(loss_function=='full_loss_with_l1'):
                loss=total_loss_l1(chi=x_k_complex.real, y=sus.float(), b=phs, d=dk, m=msk, sobel=ss)
                
            
            loss.backward()
            
            #nn.utils.clip_grad_value_(dw.parameters(), clip_value=1.0)
            
            optimizer.step()
            
            # print('lambda',dw.lambda_val.item(),dw.lambda_val.grad.item())
            # print('p',dw.p.item(),dw.p.grad.item())
            runningLoss += loss.item()

    # printing the training loss.........
    
    loss_Train.append(runningLoss/len(trainloader))
    print('Training_loss:', loss_Train)    
    print('---------------------------------------------------------')
    print('apha',net.alpha.item())
    print('---------------------------------------------------------')

    torch.save(net.state_dict(), directory+'Spinet_QSM_model_'+str(epoch)+'_.pth')
    time.sleep(10)

    running_val_Loss=0
    net.eval()

    for i, data in tqdm.tqdm(enumerate(valloader)):
            phs, msk, sus,file_name = data

            dk_repeat = dk.repeat(batch_size,1,1,1,1)
            dk_repeat=dk_repeat.cuda(device_id)
            phs=phs.cuda(device_id)
            msk=msk.cuda(device_id)
            sus=sus.cuda(device_id)
            
            # calculating term1=alpha(phi^H)y
            
            term_1_complex = torch.fft.ifftn(dk_repeat*torch.fft.fftn(phs,dim=[2,3,4]),dim=[2,3,4])
            term_1_complex=net.alpha*(term_1_complex)
            
            # initialize with zeros....
            x_0_complex=torch.zeros_like(term_1_complex)
            x_k_complex=x_0_complex;
            
            
            for k in range(K_unrolling):
                # print(k)
                term_2_complex=torch.fft.ifftn(dk_repeat*dk_repeat*torch.fft.ifftn(x_k_complex,dim=[2,3,4]),dim=[2,3,4])
                
                # calculating (I-(Phi)^H(Phi))x
                term_3_complex=x_k_complex-(net.alpha)*term_2_complex
                
                # calculating 
                term_4_complex=term_1_complex+term_3_complex
                term_4_real=z_complex_to_z_real(term_4_complex)
                term_4_real=term_4_real*msk

                
                term_4_real=(term_4_real-sus_mean)/sus_std
                x_k_real=net_par(term_4_real)
                x_k_real=x_k_real*sus_std+sus_mean
                x_k_real=x_k_real*msk

                
                
                x_k_complex=z_real_to_z_complex(x_k_real)
                x_k_complex=x_k_complex*msk
                
    
            x_k_complex = x_k_complex * msk
            
            if(loss_function=='MSE'):
                loss=loss_critearia(x_k_complex.real, sus.float())
            elif(loss_function=='full_loss_with_l1'):
                loss=total_loss_l1(chi=x_k_complex.real, y=sus.float(), b=phs, d=dk, m=msk, sobel=ss)

            #loss=total_loss_l1(chi=x_k_complex.real, y=sus.float(), b=phs, d=dk, m=msk, sobel=ss)
            running_val_Loss+=loss.item()

    loss_Val.append(running_val_Loss/len(valloader))
    print('Validation_loss:',loss_Val)

    # printing the validation loss...
    aplha_list.append(net.alpha.item())
    
    print('Checking what is the K value:',K_unrolling)

    # saving mdodel details    
    import pandas as pd
    
    model_details={"train_loss":loss_Train,"valid_loss":loss_Val,'aplha':aplha_list}
    df = pd.DataFrame.from_dict(model_details) 
    
    path=directory+'model_details.csv'
    df.to_csv (path, index = False, header=True)
    scipy.io.savemat(directory+"/model_details.mat", model_details)
    
    print('\n-------------------------------------------------------------------------------------------------\n')
