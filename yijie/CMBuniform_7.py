import numpy as np
import scipy
import camb
import scipy.linalg
from camb import model, initialpower
from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid
n= 6#index for job
begin=int(n*20000)
end=min(int((n+1)*20000),200000)
paraminfo=np.load('CMBSO/uniform2.npy',allow_pickle=True)[begin:end]#watch for the dir
#pick range
pars = camb.CAMBparams()
camb.set_params(DoLensing=False)
accuracy=1
TT=[]
TE=[]
EE=[]
for i in range(len(paraminfo)):


    pars.set_cosmology(H0 = paraminfo[i,2], ombh2 = paraminfo[i,0],
                       omch2 = paraminfo[i,1], mnu = 0.06, tau = 0.06,
                       omk = 0)

    pars.InitPower.set_params(As = np.exp(paraminfo[i,3])/(1e10), ns = paraminfo[i,4])
    pars.set_for_lmax(3100)
    pars.DarkEnergy = DarkEnergyPPF(w=-1,
                                    wa=0)
    pars.set_accuracy(AccuracyBoost = accuracy)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK',raw_cl=True)
    totCL=powers['unlensed_scalar']
    TT.append(totCL[:,0][30:3000])
    TE.append(totCL[:,3][30:3000])
    EE.append(totCL[:,1][30:3000])

np.save('./YZ_CMBuniform2/TT/TT_'+str(n+1)+'.npy',np.array(TT))
np.save('./YZ_CMBuniform2/TE/TE_'+str(n+1)+'.npy',np.array(TE))
np.save('./YZ_CMBuniform2/EE/EE_'+str(n+1)+'.npy',np.array(EE))