# import pywarraychannels.src.pywarraychannels as pywarraychannels
import pywarraychannels as pywarraychannels
# from pywarraychannels import *
import numpy as np
import json, argparse
import scipy, sys, os
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

def ested_paths_v1(ested_aoas, ested_aods, ested_toas, true_aoas, true_aods, true_toas):
    '''Find the corresponding true paths in the channel'''
    # ested_aoas, ested_aods, ested_toas: shape: N_est x 3; N_est x 3; N_est x 1
    
    ested_angs = np.hstack([ested_aoas, ested_aods])
    true_angs = np.hstack([true_aoas, true_aods])

    ang_diff_mat = np.dot(ested_angs, true_angs.T)
    toa_diff_mat = np.abs(np.reshape(ested_toas, [-1, 1]) - np.reshape(true_toas, [1, -1]))

    ang_id = ang_diff_mat.argmax(axis=1)

    ang_pass_id = ang_diff_mat.max(axis=1) >= 1.99
    toa_id = toa_diff_mat[range(ested_angs.shape[0]), ang_id] <= 0.2e-8
    pass_id = ang_pass_id & toa_id
    
    final_id = ang_id[pass_id]

    avg_ang_err = [] # in degree

    if np.any(pass_id):
        ested_aoas = ested_aoas[pass_id, :]
        ested_aods = ested_aods[pass_id, :]
        ested_toas = ested_toas[pass_id]

        true_aoas = true_aoas[final_id, :]
        true_aods = true_aods[final_id, :]
        # true_toas = true_toas[toa_id[final_id]]
        true_toas = true_toas[final_id]
        true_toas = np.reshape(true_toas, [-1, 1])

        est_err_aod = np.rad2deg(np.arccos(np.diag(np.dot(ested_aods, true_aods.T))))
        avg_ang_err.append(np.mean(est_err_aod))
        est_err_aoa = np.rad2deg(np.arccos(np.diag(np.dot(ested_aoas, true_aoas.T))))
        avg_ang_err.append(np.mean(est_err_aoa))
        
        avg_ang_err = np.array(avg_ang_err)


        return final_id, ested_aoas, ested_aods, ested_toas, true_aoas, true_aods, true_toas, avg_ang_err
    else:
        return None

data_id = '3'
Pt_dBm = 40
num_est_path = 5

parser = argparse.ArgumentParser(description='channel estimation (drop DoA then retrive DoA)')
parser.add_argument('--dataset', '-ds', action='store', required=False,
                    default = f'{data_id}')
parser.add_argument('--power_dBm', '-p', action='store', required=False,
                    default = f'{Pt_dBm}')
parser.add_argument('--num_path', '-n', action='store', required=False,
                    default = f'{num_est_path}')
cmd_args = parser.parse_args()

# Params
Dataset_id = int(cmd_args.dataset)
set = f"Yun{Dataset_id}"         # Dataset
link = "up"             # Whether it's up-link or down-link
method = "MOMP"         # Channel estimation method (MOMP or OMP)
K_res = 128             # Method's dictionary resolution
K_res_lr = 4            # Method's dictionary low resolution
index = 0               # Complementary to the dataset, you should be able to eliminate this variable if you're loading different data

# Power
p_t_dBm = float(cmd_args.power_dBm)            # dBm
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
N_est = int(cmd_args.num_path)               # Number of estimated paths
Simple_U = False        # Wether to apply SVD reduction to the measurements

filter = pywarraychannels.filters.RCFilter(early_samples=8, late_samples=8)

# Pilot signals
if link == "up":
    Pilot = np.concatenate([scipy.linalg.hadamard(Q)[:N_RF_UE], np.zeros((N_RF_UE, K//2))], axis=1)
else:
    Pilot = np.concatenate([scipy.linalg.hadamard(Q)[:N_RF_AP], np.zeros((N_RF_AP, K//2))], axis=1)
P_len = Pilot.shape[1]
D = K+filter.early_samples+filter.late_samples
Pilot = np.concatenate([np.zeros((Pilot.shape[0], D)), Pilot], axis=1)          # Zero-padding

# Define antennas
antenna_UE = pywarraychannels.antennas.RectangularAntenna((N_UE, N_UE))
antenna_AP = pywarraychannels.antennas.RectangularAntenna((N_AP, N_AP))

# test antenna 
expect_facing = [[1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0]]
for ii in range(len(orientations_UE)):
    antenna_UE.uncertainty = orientations_UE[ii]
    facing = antenna_UE.uncertainty.apply([0, 0, 1])
    assert (np.around(facing, 4) == np.around(expect_facing[ii], 4)).all()
    
# Define codebooks
antenna_UE.set_reduced_codebook((N_M_UE, N_M_UE))
antenna_AP.set_reduced_codebook((N_M_AP, N_M_AP))

# Split codebooks according to number of RF-chains
cdbks_UE = np.transpose(np.reshape(antenna_UE.codebook, [N_UE**2, -1, N_RF_UE]), [1, 0, 2])
cdbks_AP = np.transpose(np.reshape(antenna_AP.codebook, [N_AP**2, -1, N_RF_AP]), [1, 0, 2])

# Transform params to natural units
f_c *= 1e9
B *= 1e9
T += 273.1
p_t = np.power(10, (p_t_dBm-30)/10)

# Compute noise level
p_n = k_B*T*B
print("Noise level: {:.2f}dBm".format(10*np.log10(p_n)+30))


if link == "up":
    channel_Geometric = pywarraychannels.channels.Geometric(
        antenna_AP, antenna_UE, f_c=f_c,
        B=B, K=K, filter=filter, bool_sync=True)
else:
    channel_Geometric = pywarraychannels.channels.Geometric(
        antenna_UE, antenna_AP, f_c=f_c,
        B=B, K=K, filter=filter, bool_sync=True)
channel_MIMO = pywarraychannels.channels.MIMO(channel_Geometric, pilot=Pilot)
channel = pywarraychannels.channels.AWGN(channel_MIMO, power=p_t, noise=p_n)

# Whitening matrices
if link == "up":
    LLinv = [np.linalg.inv(np.linalg.cholesky(np.dot(np.conj(cdbk.T), cdbk))) for cdbk in cdbks_AP]
else:
    LLinv = [np.linalg.inv(np.linalg.cholesky(np.dot(np.conj(cdbk.T), cdbk))) for cdbk in cdbks_UE]

# Measurement matrix
if link == "up":
    """
    L_invW = []
    for cdbk_AP, Linv in zip(cdbks_AP, LLinv):
        L_invW.append(np.dot(Linv, np.conj(cdbk_AP.T)))
    L_invW = np.concatenate(L_invW, axis=0)     # N_M_RX  x  N_RX
    """
    FE_conv = []
    for cdbk_UE in cdbks_UE:
        FE = np.dot(cdbk_UE, Pilot)
        FE_conv.append(np.zeros((N_M_UE**2, D, P_len), dtype="complex128"))
        for k in range(D):
            FE_conv[-1][:, k, :] = FE[:, D-k:P_len+D-k]
    FE_conv = np.concatenate(FE_conv, axis=2)
    FE_conv = FE_conv.transpose([2, 0, 1])      # (P_len*N_M_TX/N_RF_TX)  x  N_TX x D

    if Simple_U:
        #L_invW_U, _, _ = np.linalg.svd(L_invW, full_matrices=False)
        FE_conv_U, _, _, = np.linalg.svd(FE_conv.reshape([-1, N_UE*N_UE*D]), full_matrices=False)

        #L_invW_x_U = np.tensordot(L_invW_U.conj(), L_invW, axes=(0, 0))
        FE_conv_x_U = np.tensordot(FE_conv_U.conj(), FE_conv, axes=(0, 0))

        A = FE_conv_x_U.reshape((-1, N_UE, N_UE, D))
    else:
        A = FE_conv.reshape((-1, N_UE, N_UE, D))
else:
    """
    L_invW = []
    for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
        L_invW.append(np.dot(Linv, np.conj(cdbk_UE.T)))
    L_invW = np.concatenate(L_invW, axis=0)     # N_M_RX  x  N_RX
    """
    FE_conv = []
    for cdbk_AP in cdbks_AP:
        FE = np.dot(cdbk_AP, Pilot)
        FE_conv.append(np.zeros((N_M_AP**2, D, P_len), dtype="complex128"))
        for k in range(D):
            FE_conv[-1][:, k, :] = FE[:, D-k:P_len+D-k]
    FE_conv = np.concatenate(FE_conv, axis=2)
    FE_conv = FE_conv.transpose([2, 0, 1])      # (P_len*N_M_TX/N_RF_TX)  x  N_TX x D

    if Simple_U:
        #L_invW_U, _, _ = np.linalg.svd(L_invW, full_matrices=False)
        FE_conv_U, _, _, = np.linalg.svd(FE_conv.reshape([-1, N_AP*N_AP*D]), full_matrices=False)

        #L_invW_x_U = np.tensordot(L_invW_U.conj(), L_invW, axes=(0, 0))
        FE_conv_x_U = np.tensordot(FE_conv_U.conj(), FE_conv, axes=(0, 0))

        #A = L_invW_x_U.reshape((-1, 1, N_UE, N_UE, 1, 1, 1)) * FE_conv_x_U.reshape((1, -1, 1, 1, N_AP, N_AP, D))
        A = FE_conv_x_U.reshape((-1, N_AP, N_AP, D))
    else:
        FE_conv_x_U.reshape((-1, N_AP, N_AP, D))

angles_AP = np.linspace(-np.pi, np.pi, int(N_M_AP*K_res))
angles_UE = np.linspace(-np.pi, np.pi, int(N_M_UE*K_res))
A_AP = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP[np.newaxis, :])
A_UE = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE[np.newaxis, :])
delays = np.linspace(0, K, int(K*K_res))
A_time = filter.response(K, delays)

# Sparse decomposition components
angles_AP_lr = np.linspace(-np.pi, np.pi, int(N_M_AP*K_res_lr))
angles_UE_lr = np.linspace(-np.pi, np.pi, int(N_M_UE*K_res_lr))
A_AP_lr = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP_lr[np.newaxis, :])
A_UE_lr = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE_lr[np.newaxis, :])
delays_lr = np.linspace(0, K, int(K*K_res_lr))
A_time_lr = filter.response(K, delays_lr)

# Dictionaries
if link == "up":
    X = [
        np.conj(A_UE),
        np.conj(A_UE),
        A_time
    ]
    X_lr = [
        np.conj(A_UE_lr),
        np.conj(A_UE_lr),
        A_time_lr
    ]
else:
    X = [
        np.conj(A_AP),
        np.conj(A_AP),
        A_time
    ]
    X_lr = [
        np.conj(A_AP_lr),
        np.conj(A_AP_lr),
        A_time_lr
    ]

# Define decomposition algorithm
stop = MOMP.stop.General(maxIter=N_est)     # Stop when reached the desired number of estimated paths
if method == "OMP":
    X_kron = A.copy()
    for x in X:
        X_kron = np.tensordot(X_kron, x, axes = (1, 0))
    X_kron = np.reshape(X_kron, [X_kron.shape[0], -1])
    proj = MOMP.proj.OMP_proj(X_kron)
else:
    proj_init = MOMP.proj.MOMP_greedy_proj(A, X, X_lr, normallized=False)
    proj = MOMP.proj.MOMP_proj(A, X, initial=proj_init, normallized=False)
alg = MOMP.mp.OMP(proj, stop)

with open("data/{}/AP_pos.txt".format(set)) as f:
    AP_pos_all = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
with open("data/{}/UE_pos.txt".format(set)) as f:
    UE_pos_all = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
with open("data/{}/AP_selected.txt".format(set, index)) as f:
    AP_selected_all = [int(a) for a in f.read().split("\n")[1].split()]
with open("data/{}/Info_selected.txt".format(set, index)) as f:
    Rays_all = [pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in ue_block.split("\n")], bool_flip_RXTX=link=="up") for ue_block in f.read()[:-1].split("\n<ue>\n")]
chan_ids = np.squeeze(sio.loadmat(f'../Dataset/StrongestChanID_Set{Dataset_id}.mat')['chan_ids'] - 1)
# print(Rays_all[0], len(Rays_all))

for dir_name in [f'./data/{set}/paths-dropDoA', f'./data/{set}/paths-retDoA']:
        if not os.path.exists(f'{dir_name}'):
            os.makedirs(f'{dir_name}')


samples = len(chan_ids)    # Number of samples from the dataset to evaluate
samp_start = 0
# Crop data
UE_pos, AP_selected, Rays = [X[samp_start:samp_start+samples] for X in [UE_pos_all, AP_selected_all, Rays_all]]

# Build channels and decompose them
estimation = []
for rays, ue_pos, ap, ii_ue in tqdm(zip(Rays, UE_pos, AP_selected, range(len(UE_pos)))):
    antenna_UE.uncertainty = orientations_UE[int(chan_ids[samp_start+ii_ue] % 4)]
    antenna_AP.uncertainty = orientations_AP[0]
    channel.build(rays)
    MM = []
    # print("Iteration: {}/{}".format(ii_ue, samples))
    if link == "up":
        for cdbk_AP, Linv in zip(cdbks_AP, LLinv):
            MMM = []
            antenna_AP.set_codebook(cdbk_AP)
            channel.set_corr(np.dot(np.conj(antenna_AP.codebook.T), antenna_AP.codebook))
            for cdbk_UE in cdbks_UE:
                antenna_UE.set_codebook(cdbk_UE)
                MMM.append(np.dot(Linv, channel.measure()))
            MM.append(MMM)
    else:
        for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
            MMM = []
            antenna_UE.set_codebook(cdbk_UE)
            channel.set_corr(np.conj(antenna_UE.codebook.T), antenna_UE.codebook)
            for cdbk_AP in cdbks_AP:
                antenna_AP.set_codebook(cdbk_AP)
                MM.append(np.dot(Linv, channel.measure()))
            MM.append(MMM)
    M = np.concatenate([np.concatenate(MMM, axis=1) for MMM in MM], axis=0)
    #M_U_U = np.tensordot(np.tensordot(M, L_invW_U.conj(), axes=(0, 0)), FE_conv_U.conj(), axes=(0, 0))
    if Simple_U:
        M_U_U = np.tensordot(M, FE_conv_U.conj(), axes=(1, 0))
    else:
        M_U_U = M
    tic = time()
    I, alpha = alg(M_U_U.T)
    toc = time()-tic
    if method == "OMP":
        I = [list(np.unravel_index(ii, [X.shape[1] for X in X_components])) for ii in I]
    Alpha = []
    Power = []
    DoA = []
    DoD = []
    ToF = []
    for a, iii in zip(alpha, I):
        Alpha.append(a)
        Power.append(20*np.log10(np.linalg.norm(a)))
        ii_component = 0
        """
        if link == "up":
            xoa, yoa = [angles_AP[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
        else:
            xoa, yoa = [angles_UE[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
        zoa = xoa**2 + yoa**2
        if zoa > 1:
            xoa, yoa = xoa/np.sqrt(zoa), yoa/np.sqrt(zoa)
            zoa = 0
        else:
            zoa = np.sqrt(1-zoa)
        doa = np.array([xoa, yoa, -zoa])
        DoA.append(doa)
        ii_component += 2
        """
        if link == "up":
            xod, yod = [angles_UE[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
        else:
            xod, yod = [angles_AP[iiii]/np.pi for iiii in iii[ii_component:ii_component+2]]
        zod = xod**2 + yod**2
        if zod > 1:
            xod, yod = xod/np.sqrt(zod), yod/np.sqrt(zod)
            zod = 0
        else:
            zod = np.sqrt(1-zod)
        dod = np.array([xod, yod, zod])
        DoD.append(dod)
        ii_component += 2
        tof = delays[iii[ii_component]]
        ToF.append(tof)
    Alpha = np.array(Alpha)
    Power = np.array(Power)
    #DoA = np.array(DoA)
    DoD = np.array(DoD)
    if link == "up":
        #DoA = antenna_AP.uncertainty.apply(DoA)
        DoD = antenna_UE.uncertainty.apply(DoD)
    else:
        #DoA = antenna_UE.uncertainty.apply(DoA)
        DoD = antenna_AP.uncertainty.apply(DoD)
    ToF = np.array(ToF)/B
    TDoF = ToF - ToF[0]
    DDoF = TDoF*c
    
    # ray_info_rad = np.deg2rad(rays.ray_info[:, -2:].copy())
    # print(np.hstack([np.cos(ray_info_rad[:, -1]).reshape([-1, 1]) * np.cos(ray_info_rad[:, -2]).reshape([-1, 1]), np.cos(ray_info_rad[:, -1]).reshape([-1, 1]) * np.sin(ray_info_rad[:, -2]).reshape([-1, 1]), np.sin(ray_info_rad[:, -1]).reshape([-1, 1])]))
    # print(f'Est DoD: {DoD}')
    
    estimation.append({
        "Alpha_r": np.real(Alpha).tolist(), "Alpha_i": np.imag(Alpha).tolist(), "Power": Power.tolist(),
        "DoA": None, "DoD": DoD.tolist(),
        "DDoF": DDoF.tolist(), "CTime": toc})

with open(f"./data/{set}/paths-dropDoA/single_{method}_{N_M_UE}_{N_M_AP}_{int(p_t_dBm)}dBm_{10*K_res}_{len(estimation)}ue.json", 'w') as f:
    f.write(json.dumps(estimation))

# ## Retrieving the estimated DoA from the estimated $\beta$
# Measurement matrix
if link == "up":
    L_invW = []
    for cdbk_AP, Linv in zip(cdbks_AP, LLinv):
        L_invW.append(np.dot(Linv, np.conj(cdbk_AP.T)))
    L_invW = np.concatenate(L_invW, axis=0).reshape([N_M_AP**2, N_AP, N_AP])    # N_M_RX  x  N_RX
else:
    L_invW = []
    for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
        L_invW.append(np.dot(Linv, np.conj(cdbk_UE.T)))
    L_invW = np.concatenate(L_invW, axis=0).reshape([N_M_UE**2, N_UE, N_UE])    # N_M_RX  x  N_RX

estimation_all = []
for estimation_new, rays, ue_pos, ap, ii_ue in zip(estimation, Rays, UE_pos, AP_selected, range(len(UE_pos))):
    antenna_AP.uncertainty = orientations_AP[0]
    Beta = np.asarray(estimation_new["Alpha_r"])+1j*np.asarray(estimation_new["Alpha_i"])
    DoD = np.asarray(estimation_new["DoD"])
    DDoF = np.asarray(estimation_new["DDoF"])
    DoA = []
    Alpha = []
    Power = []
    for beta in zip(Beta):
        if link == "up":
            Proj = np.tensordot(np.tensordot(
                np.tensordot(np.conj(beta[0]), L_invW, axes=(0, 0)),
                A_AP, axes=(0, 0)), A_AP, axes=(0, 0))
            iii1, iii2 = np.unravel_index(np.argmax(np.abs(Proj)), Proj.shape)
            xoa = angles_AP[iii1]/np.pi
            yoa = angles_AP[iii2]/np.pi
        else:
            Proj = np.tensordot(np.tensordot(
                np.tensordot(np.conj(beta[0]), L_invW, axes=(0, 0)),
                A_UE, axes=(0, 0)), A_UE, axes=(0, 0))
            iii1, iii2 = np.unravel_index(np.argmax(np.abs(Proj)), Proj.shape)
            xoa = angles_UE[iii1]/np.pi
            yoa = angles_UE[iii2]/np.pi
        Alpha.append(Proj[iii1, iii2])
        Power.append(20*np.log10(np.abs(Proj[iii1, iii2])))
        zoa = xoa**2 + yoa**2
        if zoa > 1:
            xoa, yoa = xoa/np.sqrt(zoa), yoa/np.sqrt(zoa)
            zoa = 0
        else:
            zoa = np.sqrt(1-zoa)
        doa = np.array([xoa, yoa, zoa])
        DoA.append(doa)
    Alpha = np.array(Alpha)
    Power = np.array(Power)
    DoA = np.array(DoA)
    if link == "up":
        DoA = antenna_AP.uncertainty.apply(DoA)
    else:
        DoA = antenna_UE.uncertainty.apply(DoA)
  
    if samples <= 5:
        DoA_azi = np.rad2deg(np.angle(DoA[:, 0:1] + 1j * DoA[:, 1:2]))
        DoD_azi = np.rad2deg(np.angle(DoD[:, 0:1] + 1j * DoD[:, 1:2]))

        DoA_ele = np.rad2deg(np.arcsin(DoA[:, 2:]))
        DoD_ele = np.rad2deg(np.arcsin(DoD[:, 2:]))
        print(np.hstack([DoA_azi, DoA_ele, DoD_azi, DoD_ele]))
        print(rays)

    estimation_all.append({
        "Alpha_r": np.real(Alpha).tolist(), "Alpha_i": np.imag(Alpha).tolist(), "Power": Power.tolist(),
        "DoA": DoA.tolist(), "DoD": DoD.tolist(),
        "DDoF": DDoF.tolist(), "CTime": toc})
   
with open(f"./data/{set}/paths-retDoA/single_{method}_{N_M_UE}_{N_M_AP}_{int(p_t_dBm)}dBm_{10*K_res}_{len(estimation_all)}ue.json", 'w') as f:
    f.write(json.dumps(estimation_all))
print(f'single_{method}_{N_M_UE}_{N_M_AP}_{int(p_t_dBm)}dBm_{10*K_res}_{len(estimation_all)}ue finished!')




