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
from deepqsm import DeepQSM
from WideResnet  import WideResNet
from qsmnet import QSMnet

#%%
def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")
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
batch_size=1
epsilon=1e-5
is_data_normalized=True
model='WideResNet'

#%%
epoch=48

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="epoch number for testing ",type=int,default=epoch)
args = parser.parse_args()

epoch=args.epoch;

print(epoch)




#%%
data_source='given_single_patient_data'
data_source_no=1
Training_patient_no=1

if(data_source=='generated_data'):
    raw_data_path='../../QSM_data/data_for_experiments/generated_data/raw_data/'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1/'
    patients_list =[7,32,9,10]

elif(data_source=='generated_noisy_data'):
    raw_data_path='../../QSM_data/data_for_experiments/generated_data/raw_data_noisy_sigma_0.03/'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1_sigma_0.03/'
    patients_list =[7,32,9,10]

elif(data_source=='generated_undersampled_data'):
    raw_data_path='../../QSM_data/data_for_experiments/generated_data/sampling_data/sampled_0.2//'
    data_path='../../QSM_data/data_for_experiments/generated_data/data_source_1/'
    patients_list =[7,32,9,10]




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


elif(data_source=='given_single_patient_data'):
    raw_data_path='../../QSM_data/data_for_experiments/given_data/raw_data_names_modified/'

    if(Training_patient_no==1):
        csv_path='../../QSM_data/data_for_experiments/given_data/single_patient/patient_1'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[2,3,4,5,6,7,8,9,10,11,12]

    elif(Training_patient_no==2):
        csv_path='../../QSM_data/data_for_experiments/given_data/single_patient/patient_2'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[1,3,4,5,7,8,9,10,11,12]

    elif(Training_patient_no==3):
        csv_path='../../QSM_data/data_for_experiments/given_data/single_patient/patient_3'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[1,2,4,5,7,8,9,10,11,12]

    if(Training_patient_no==4):
        csv_path='../../QSM_data/data_for_experiments/given_data/single_patient/patient_4'
        data_path='../../QSM_data/data_for_experiments/given_data/data_as_patches/'
        patients_list =[1,2,3,5,7,8,9,10,11,12]



import os
print(os.listdir(raw_data_path))

print('data_path:',data_path)
print('csv_path:',csv_path)





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




#%%
# paramaters for dipole kernel
matrix_size = [176, 176, 160]
voxel_size = [1,  1,  1]


dk = dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1])
dk = torch.unsqueeze(dk, dim=0)
dk=dk.cuda(device_id)

ss = sobel_kernel()
ss = ss.cuda(device_id)
# print(ss.shape)

#%%

experiments_folder="savedModels/LPCNN_QSM_MODELS/full_sampling_models/single_patient//"
experiment_name="Aug_03_04_10_pm_model_K_3_model_WideResNet_patient_1/"

model_name="Spinet_QSM_model_"+str(epoch)+"_.pth"
model_path=experiments_folder+"/"+experiment_name+"/"+model_name
print('model_path:',model_path)

try:
    os.makedirs(experiments_folder+"/"+experiment_name+"/output_csv")
except:
    print("Exception...")

last_string=model_path.split("/")[-1]
directory=model_path.replace(last_string,"")

print('directory:',directory)
print(os.listdir(directory))

#%%

if(model=='deepqsm'):
    net = DeepQSM().cuda(device_id)
elif(model=='QSMnet'):
    net=QSMnet().cuda(device_id)
elif(model=='WideResNet'):
    net=WideResNet().cuda(device_id)

#net= torch.nn.DataParallel(net, device_ids=[device_id])  
net.load_state_dict(torch.load(model_path))
net.eval()


print(net)

#%%

outdir = directory+'predictions_'+str(epoch)+"/"
print(outdir)
import os
try:
    os.makedirs(outdir)
except:
    print("aalready tested..")
#%%

#%%
# Hemorrhage_path='../../QSM_data/data_for_experiments/modl_qsm_data/data/Hemorrhage/';

# import scipy.io
# temp=scipy.io.loadmat(Hemorrhage_path+"/test_data.mat")

# phs=temp['phi']
# msk=temp['mask']
# sus=temp['star_qsm']

#%%
# MS_data_path='../QSM_data/data_for_experiments/modl_qsm_data/data/MS_data/'

# import scipy.io
# temp=scipy.io.loadmat(MS_data_path+"/test_data.mat")

# phs=temp['phi']
# msk=temp['mask']
# sus=temp['star_qsm']

#%%

# Prisma_data_1='../QSM_data/data_for_experiments/modl_qsm_data/data/Prisma_data/sub1/'

# import scipy.io
# temp=scipy.io.loadmat(Prisma_data_1+"/test_data.mat")

# phs=temp['phi']
# msk=temp['mask']
# sus=temp['labels'][:,:,:,0]
# sus=temp['labels'][:,:,:,1]
#%%

# Prisma_data_2='../QSM_data/data_for_experiments/modl_qsm_data/data/Prisma_data/sub2/'

# import scipy.io
# temp=scipy.io.loadmat(Prisma_data_2+"/test_data.mat")

# phs=temp['phi']
# msk=temp['mask']
# sus=temp['labels'][:,:,:,0]

   
import datetime;
 
# ct stores current time
#%%
with torch.no_grad():


    for i in patients_list:
        print("Patinte:"+str(i)+"\n")
        for j in range(1,6):
            print('orientation:',j);
            phs=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/phs'+str(j)+'.mat')['phs']
            sus=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/cos'+str(j)+'.mat')['cos']
            msk=scipy.io.loadmat(raw_data_path+'/patient_'+str(i)+'/msk'+str(j)+'.mat')['msk']
    
    
            phs=torch.unsqueeze(torch.unsqueeze(torch.tensor(phs),0),0)
            sus=torch.unsqueeze(torch.unsqueeze(torch.tensor(sus),0),0)
            msk=torch.unsqueeze(torch.unsqueeze(torch.tensor(msk),0),0)
    

            phs=phs.cuda(device_id)
            sus=sus.cuda(device_id)
            msk=msk.cuda(device_id)

            dk_repeat = dk.repeat(batch_size,1,1,1,1)
            dk_repeat=dk_repeat.cuda(device_id)
            phs=phs.cuda(device_id).float()
            msk=msk.cuda(device_id).float()
            sus=sus.cuda(device_id).float()
            
            tic()            
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
                
                # calculating alpha*(Phi^H)y+(I-alpha*(Phi)^H(Phi))x
                term_4_complex=term_1_complex+term_3_complex
                term_4_real=z_complex_to_z_real(term_4_complex)
                term_4_real=term_4_real*msk
                
                term_4_real=(term_4_real-sus_mean)/sus_std
                x_k_real=net(term_4_real)
                x_k_real=x_k_real*sus_std+sus_mean
                x_k_real=x_k_real*msk

                x_k_complex=z_real_to_z_complex(x_k_real)
                x_k_complex=x_k_complex*msk
            x_k_complex = x_k_complex * msk
            toc()            
            x_k_cpu=(x_k_complex.real.detach().cpu().numpy())*(msk.detach().cpu().numpy() )
            mdic  = {"modl" : x_k_cpu}
            filename  = outdir + 'modl-net-'+ str(i)+'-'+str(j)+'.mat'
            scipy.io.savemat(filename, mdic)

            
