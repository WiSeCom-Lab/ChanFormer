import pywarraychannels.src.pywarraychannels as pywarraychannels
# from pywarraychannels import *
import numpy as np
import json
import scipy, os
import scipy.io as sio
import MOMP.src.MOMP as MOMP
from time import time
import pandas as pd
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import *

# TODO: Params needs input
Dataset_id = 5
dropDoA = 1 # 1: use the drop-DoA method; 0: original method

# Other params
set = f"Yun{Dataset_id}"         # Dataset
link = "up"             # Whether it's up-link or down-link
method = "MOMP"         # Channel estimation method (MOMP or OMP)
K_res = 128             # Method's dictionary resolution
K_res_lr = 4            # Method's dictionary low resolution
index = 0               # Complementary to the dataset, you should be able to eliminate this variable if you're loading different data

# Power
p_t_dBm = 40            # dBm
# Noise related
T = 15                  # C
k_B = 1.38064852e-23    # Boltzmanz's constant
# Speed of light
c = 299792458                 # m/s
# Antennas
N_UE = 8                # Number of UE antennas in each dimension
N_AP = 16                # Number of AP antennas in each dimension
N_RF_UE = 4             # Number of UE RF-chains in total
N_RF_AP = 8             # Number of AP RF-chains in total
N_M_UE = 8              # Number of UE measurements in each dimension
N_M_AP = 16              # Number of AP measurements in each dimension
orientations_UE = [
    pywarraychannels.uncertainties.Static(tilt=-np.pi/2),
    pywarraychannels.uncertainties.Static(tilt=np.pi/2),
    pywarraychannels.uncertainties.Static(roll=np.pi/2),
    pywarraychannels.uncertainties.Static(roll=-np.pi/2)
    ]
orientations_AP = [
    orientations_UE[3]
    ]
# Carriers
f_c = 73                # GHz
B = 1                   # GHz
K = 64                  # Number of delay taps
Q = 64                  # Length of the training pilot
# Estimation
N_est = 5               # Number of estimated paths
Simple_U = False        # Wether to apply SVD reduction to the measurements

if not dropDoA:
    N_UE = 4                # Number of UE antennas in each dimension
    N_AP = 8                # Number of AP antennas in each dimension
    N_RF_UE = 2             # Number of UE RF-chains in total
    N_RF_AP = 4             # Number of AP RF-chains in total
    N_M_UE = 4              # Number of UE measurements in each dimension
    N_M_AP = 8              # Number of AP measurements in each dimension
    K_res_lr = 2
    K_res = 64
    Q = 32                 # Length of the training pilot


# Load true data
with open("data/{}/AP_pos.txt".format(set)) as f:
    AP_pos_all = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
with open("data/{}/UE_pos.txt".format(set)) as f:
    UE_pos_all = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
with open("data/{}/AP_selected.txt".format(set, index)) as f:
    AP_selected_all = [int(a) for a in f.read().split("\n")[1].split()]
with open("data/{}/Info_selected.txt".format(set, index)) as f:
    Rays_all = [pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in ue_block.split("\n")], bool_flip_RXTX=link=="up") for ue_block in f.read()[:-1].split("\n<ue>\n")]
# chan_ids = np.squeeze(sio.loadmat(f'../Dataset/StrongestChanID_Set{Dataset_id}.mat')['chan_ids'] - 1)

# Load estimates from saved file
ue_number = 2000 if (Dataset_id != 1) & (Dataset_id != 2) else 1996
folder_to_load = 'paths-retDoA' if dropDoA else 'paths-all'
with open(os.getcwd()+ f'/data/Yun{Dataset_id}/{folder_to_load}/single_{method}_{N_M_UE}_{N_M_AP}_{int(p_t_dBm)}dBm_{10*K_res}_{ue_number}ue.json', 'r') as f:
    estimation = json.loads(f.read())

estimations = np.zeros([1, 11])
AVG_delay_err = np.zeros([len(estimation)]) - 1
AVG_ang_err = np.zeros([len(estimation), 2])-1
num_usable_paths = []

mapped_DL_DoA = []
mapped_DL_DoD = []
mapped_DL_TDoA = []

for est_id in range(len(estimation)):
# for est_id in [0,5,25]:
    ray_cur = Rays_all[est_id].ray_info
    true_aoas, true_aods, true_toas = np.array(pywarraychannels.em.polar2cartesian(np.deg2rad(ray_cur[:, 3]), np.deg2rad(ray_cur[:, 4]))).T, np.array(pywarraychannels.em.polar2cartesian(np.deg2rad(ray_cur[:, 5]), np.deg2rad(ray_cur[:, 6]))).T, ray_cur[:, 1] - min(ray_cur[:, 1])
    
    estimation_cur = estimation[est_id]
    ested_aoas, ested_aods, ested_toas = np.array(estimation_cur['DoA']), np.array(estimation_cur['DoD']), np.array(estimation_cur["DDoF"])/c
    # print(ested_aoas,'\n\n', true_aoas[:10, :], '\n---------\n')

    estimation_concat = np.hstack([ested_aoas, ested_aods, np.reshape(np.array(estimation_cur['Power']), [-1, 1]), np.reshape(ested_toas, [-1, 1]), np.tile(np.array(UE_pos_all)[est_id,:], (N_est, 1))])
    estimations = np.vstack([estimations, estimation_concat])

    # print('Estimated paths [DoA, DoD, TDoA]')
    # print(pd.DataFrame(np.hstack([np.around(np.reshape(ested_aoas, [-1,3]), 6), np.around(np.reshape(ested_aods, [-1,3]), 6), np.reshape(ested_toas, [-1, 1])])))
    # print('True paths [DoA, DoD, TDoA]')
    # print(pd.DataFrame(np.hstack([np.reshape(true_aoas, [-1, 3]), np.reshape(true_aods, [-1, 3]), np.reshape(true_toas, [-1, 1])]))[:8])



    try: 
        final_id, ested_aoas, ested_aods, ested_toas, true_aoas, true_aods, true_toas, avg_ang_err = ested_paths_v1(ested_aoas, ested_aods, ested_toas, true_aoas, true_aods, true_toas)
        # print('Estimated paths [DoA, DoD, TDoA]')
        # print(pd.DataFrame(np.hstack([np.reshape(ested_aoas, [-1,3]), np.reshape(ested_aods, [-1,3]), np.reshape(ested_toas, [-1, 1]), np.reshape(final_id, [-1,1])]) ))
        # print('True paths [DoA, DoD, TDoA]')
        # print(pd.DataFrame(np.hstack([np.reshape(true_aoas, [-1, 3]), np.reshape(true_aods, [-1, 3]), np.reshape(true_toas, [-1, 1])])))
        # print(f'doa err: {avg_ang_err[1]: .4f} deg; dod err: {avg_ang_err[0]: .4f} deg;') # uplink
        # print(f'dod err: {avg_ang_err[1]: .4f} deg; doa err: {avg_ang_err[0]: .4f} deg;') # downlink
        # AVG_delay_err[est_id] = np.mean(abs((np.squeeze(ested_toas) - np.squeeze(true_toas)) / (np.squeeze(true_toas)+1e-8)))
        AVG_ang_err[est_id, :] = avg_ang_err
        num_usable_paths.append(np.shape(ested_aoas)[0])
        
        # 📝 journal paper
        mapped_DL_DoA.append(np.hstack([np.reshape(ested_aods, [-1,3]), np.reshape(true_aods, [-1,3])]))
        mapped_DL_DoD.append(np.hstack([np.reshape(ested_aoas, [-1,3]), np.reshape(true_aoas, [-1,3])]))
        mapped_DL_TDoA.append(np.hstack([np.reshape(ested_toas, [-1,1]), np.reshape(true_toas, [-1,1])]))
        
        
    except Exception as e:
        print(e)
        num_usable_paths.append(0)
        
        # 📝 journal paper
        mapped_DL_DoA.append(0)
        mapped_DL_DoD.append(0)
        mapped_DL_TDoA.append(0)
        
        continue
    
sio.savemat(f"./Chan est performance/ChanEst_vs_True_Set{Dataset_id}.mat", {'mapped_DL_DoA':  mapped_DL_DoA, 'mapped_DL_DoD': mapped_DL_DoD, 'mapped_DL_TDoA': mapped_DL_TDoA})

    
estimations = estimations[1:, :]
