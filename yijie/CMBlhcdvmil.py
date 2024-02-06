import numpy as np
from mpi4py import MPI
import scipy
import sys, os
import camb
import scipy.linalg
from camb import model, initialpower
from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_ranks = comm.Get_size()

file_name       = './YZ_samples/LHS/coslhc'
cosmology_file  = file_name + '.npy'
output_file     = file_name + '_testoutput' + '.npy'

camb_accuracy_boost   = 1#test 2.5
camb_l_sampple_boost  = 1# test50    # 50 = every ell is computed
camb_ell_min          = 2#30
camb_ell_max          = 5000
camb_ell_range        = camb_ell_max  - camb_ell_min 
camb_num_spectra      = 4

total_num_dvs  = 0

if rank == 0:
    param_info_total = np.load(
        cosmology_file,
        allow_pickle = True
    )
    total_num_dvs = len(param_info_total)

    for i in range(num_ranks):
        comm.send(
            param_info_total[i:total_num_dvs:num_ranks], 
            dest = i, 
            tag  = 1
        )


param_info = comm.recv(source = 0, tag = 1)

num_datavector = len(param_info)

total_cls = np.zeros(
        (num_datavector, camb_ell_range, camb_num_spectra), dtype = "float32"
    ) 

camb_params = camb.CAMBparams()

for i in range(num_datavector):

    camb_params.set_cosmology(
        H0      = param_info[i,2], 
        ombh2   = param_info[i,0],
        omch2   = param_info[i,1], 
        mnu     = param_info[i,6], 
        tau     = param_info[i,3],
        omk     = 0
    )

    camb_params.InitPower.set_params(
        As = np.exp(param_info[i,5])/(1e10), 
        ns = param_info[i,4]
    )
    
    camb_params.set_for_lmax(camb_ell_max + 500)
    
    camb_params.DarkEnergy = DarkEnergyPPF(
        w = param_info[i,7], 
        wa = param_info[i,8]
    )

    camb_params.set_accuracy(
        AccuracyBoost  = camb_accuracy_boost, 
        lSampleBoost   = camb_l_sampple_boost, 
        lAccuracyBoost = 1  # we wont change the number of multipoles in the hierarchy
    )

    try:
        results = camb.get_results(camb_params)
        
        powers  = results.get_cmb_power_spectra(
            camb_params, 
            CMB_unit = 'muK',
            raw_cl   = True
        )
    except:
        
        total_cls[i] = np.ones((camb_ell_range, camb_num_spectra)) # put 1s for all   

    else:

        total_cls[i] = powers['total'][camb_ell_min:camb_ell_max]




if rank == 0:
    #on 0th rank, just write to the results
    result_cls = np.zeros((total_num_dvs, camb_ell_range, 3), dtype="float32")
    result_cls[0:total_num_dvs:num_ranks,:,0] = total_cls[:,:,0] ## TT

    result_cls[0:total_num_dvs:num_ranks,:,1] = total_cls[:,:,3] ## TE
        
    result_cls[0:total_num_dvs:num_ranks,:,2] = total_cls[:,:,1] ## EE
    
    for i in range(1,num_ranks):
        #receive data vectors from other ranks
        
        rec = comm.recv(source = i, tag = 0)
        
        result_cls[i:total_num_dvs:num_ranks,:,0] = rec[:,:,0] ## TT

        result_cls[i:total_num_dvs:num_ranks,:,1] = rec[:,:,3] ## TE
        
        result_cls[i:total_num_dvs:num_ranks,:,2] = rec[:,:,1] ## EE

    np.save(output_file, result_cls)
    
else:
    #on other ranks, send the data vectors.
    comm.send(total_cls, dest = 0, tag = 0)

