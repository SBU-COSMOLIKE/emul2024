import numpy as np
import scipy
import astropy
import camb
import scipy.linalg
import os
import sys
from camb import model, initialpower
from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid

import scipy.integrate as integrate
#import your cosmolike package here
#or you can write your own wrapper for cosmolike

class cambwrap:
	def __init__(self, param,z,accuracy=1.5,signal='BAO'):
		#param: a dictionary for importing input parameters
		#z(redshift) is a variable I had here for SN, you can delete it, or simply feed z=0 if you do not need it.
		pars = camb.CAMBparams()
		pars.set_cosmology(H0 = param['H0'], ombh2 = param['omega_b_h2'],
        omch2 = param['omega_c_h2'], mnu = param['mnu'], tau = param['tau'],
        nnu = param['N_eff'] if 'N_eff' in list(param.keys()) else 3.046, YHe = param['Yhe'],
        omk = param['omk'] if 'omk' in list(param.keys()) else 0.0)


		pars.InitPower.set_params(As = np.exp(param['lA_s'])/(1e10), ns = param['n_s'])
		pars.set_for_lmax(3100, lens_potential_accuracy=10)
		pars.DarkEnergy = DarkEnergyPPF(w=param['w'] if 'w' in list(param.keys()) else -1.2,
        wa=param['wa'] if 'wa' in list(param.keys()) else 0.0)
		pars.set_accuracy(AccuracyBoost = accuracy, lSampleBoost = accuracy, lAccuracyBoost = accuracy,)
		results = camb.get_results(pars)
		powers =results.get_cmb_power_spectra(pars, CMB_unit='muK',raw_cl=True)
		#unlensedCL=powers['unlensed_scalar']
		totCL=powers['total']
		if signal=='BAO':
			self.signal=results.get_BAO(z,pars)[:,0]
		elif signal=='H':
			self.signal=results.hubble_parameter(z)
		elif signal=='luminosity':
			
			self.signal=[]
			for zz in z:
				#eta=integrate.quad(lambda x: ((1+x)**2*(1+param['omm']*x)-x*(2+x)*param['omlambda'])**(-0.5), 0, zz)[0]
				#eta=integrate.quad(lambda x: ((1+x)**3*(param['omm'])+(1+x)**(3*(1+param['w']))*(1-param['omm']))**(-0.5), 0, zz)[0]
				#print(eta)
				#omkappa=1-param['omm']-param['omlambda']
				step=2**7
				zlist=np.arange(0,zz+(zz)/(step),(zz)/(step))
				integrand=((1+zlist)**3*(param['omm'])+(1+zlist)**(3*(1+param['w']))*(1-param['omm']))**(-0.5)
				eta=integrate.romb(integrand,zz/step)
				#eta=integrate.trapezoid(integrand,dx=zz/step)
				omkappa=0
				#kappa=np.sqrt(abs(omkappa))
				self.signal.append(5*np.log10(self.Sk(eta,omkappa)*(1+zz)*(3e5)/param['H0']))#c=3e5km/s
			self.signal=np.array(self.signal)
			
			
			#self.signal=5*np.log10(results.luminosity_distance(z))#mu=m-M=5log(dL)
		elif signal=='cmbpol':
			EE=unlensedCL[:,1][30:2017]
			blmin=np.genfromtxt('_external/blmin.dat',dtype=None,delimiter='')[430:629]-4928
			blmax=np.genfromtxt('_external/blmax.dat',dtype=None,delimiter='')[430:629]-4928
			weight=np.genfromtxt('_external/bweight.dat',dtype=None,delimiter='')[-2479:]
			binee=np.array([blmin,blmax]).T
			EEbin=[]
			for b in binee:
				q=0
				for i in range(int(b[0]-30),int(b[1]-29)):
					q+=EE[i]*weight[i]
				EEbin.append(q)
			self.signal=np.array(EEbin)
		elif signal=='cmbplk':
			TT=unlensedCL[:,0][30:2509]
			blmin1=np.genfromtxt('_external/blmin.dat',dtype=None,delimiter='')[:215]+30
			blmax2=np.genfromtxt('_external/blmax.dat',dtype=None,delimiter='')[:215]+30
			weight1=np.genfromtxt('_external/bweight.dat',dtype=None,delimiter='')[:2479]
			bintt=np.array([blmin1,blmax2]).T
			TTbin=[]
			for b in bintt:
				q=0
				for i in range(int(b[0]-30),int(b[1]-29)):
					q+=TT[i]*weight1[i]
				TTbin.append(q)
			TE=unlensedCL[:,3][30:2017]
			blmin3=np.genfromtxt('_external/blmin.dat',dtype=None,delimiter='')[215:414]-2449
			blmax4=np.genfromtxt('_external/blmax.dat',dtype=None,delimiter='')[215:414]-2449
			weight2=np.genfromtxt('_external/bweight.dat',dtype=None,delimiter='')[2479:4958]
			binte=np.array([blmin3,blmax4]).T
			TEbin=[]
			for b in binte:
				q=0
				for i in range(int(b[0]-30),int(b[1]-29)):
					q+=TE[i]*weight2[i]
				TEbin.append(q)
			EE=unlensedCL[:,1][30:2017]
			blmin=np.genfromtxt('_external/blmin.dat',dtype=None,delimiter='')[430:629]-4928
			blmax=np.genfromtxt('_external/blmax.dat',dtype=None,delimiter='')[430:629]-4928
			weight=np.genfromtxt('_external/bweight.dat',dtype=None,delimiter='')[-2479:]
			binee=np.array([blmin,blmax]).T
			EEbin=[]
			for b in binee:
				q=0
				for i in range(int(b[0]-30),int(b[1]-29)):
					q+=EE[i]*weight[i]
				EEbin.append(q)
			self.signal=np.concatenate((np.array(TTbin),np.array(TEbin),np.array(EEbin)))
			#self.signal=np.array(TTbin)
		elif signal=='cmbSO':
			TT=totCL[:,0][30:3000]
			TE=totCL[:,3][30:3000]
			EE=totCL[:,1][30:3000]
			self.signal=np.concatenate((TT, TE, EE))
		elif signal=='test':
			self.signal=np.sin(param['test'])
		else:
			print('not yet')

	def Sk(self,x,kappa):
		k=np.sqrt(abs(kappa))
		if kappa==0:
			return x
		elif kappa>0:
			return np.sin(k*x)/k
		else:
			return np.sinh(k*x)/k


