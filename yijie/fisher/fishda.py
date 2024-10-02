import numpy as np
import scipy
import astropy
import camb
import scipy.linalg
import os
import sys
from camb import model, initialpower
import cambwrap
from scipy.stats import multivariate_normal
import itertools
class FishDa:
	def __init__(self, fid,z,step,sigma,signal='3x2'):
		#fid: a dictionary with param and fiducial quantity
		#step: a dictionary with param and steps
		#z: an array for redshift
		#sigma: this variable is for bao, a percentage variance. You can just put sigma=0 if not needed

		#signal: This flag is for picking the corresponding signal model inside the wrapper
		self.cosmofid=cambwrap.cambwrap(fid,z,signal=signal)
		self.signalflag=signal
		self.sigfid=np.copy(self.cosmofid.signal)
		#self.Hfid=self.cosmofid.H
		self.fid=fid
		self.z=z
		self.step=step
		if signal=='BAO':
			self.sig=sigma*self.sigfid/100#for bao
			self.cov=np.linalg.inv(np.diag(self.sig))
		elif signal=='luminosity':
			self.cov=sigma
		elif signal=='H':
			self.cov=sigma
		elif signal=='cmbpol':
			
			self.cov= sigma
		elif signal=='cmbplk':
			
			self.cov= sigma
		elif signal=='cmbSO':
			
			self.cov= sigma
			
		elif signal=='test':
			self.cov=0

		elif signal=='3x2':
			self.cov=0#specify your cov mat here.	
		else:
			print('not yet')

		#self.cov=np.linalg.inv(np.diag(self.sig))#cov matrix of the data

	def firstderivnew(self,par,pointpar,derivmethod='1st'):
		#par is a string
		#derivmethod:
		#	1st: f'=[f(x+h)-f(x-h)]/2h
		#	2nd: f'=[-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)]/12h
		#return first deriv with respect to par
		up=pointpar.copy()
		down=pointpar.copy()
		up[par]+=self.step[par]
		baoup=cambwrap.cambwrap(up,self.z,signal=self.signalflag).signal
		down[par]-=self.step[par]
		baodown=cambwrap.cambwrap(down,self.z,signal=self.signalflag).signal
		if derivmethod=='1st':
			return (baoup-baodown)/(2*self.step[par])
		elif derivmethod=='2nd':
			upup=pointpar.copy()
			downdown=pointpar.copy()
			upup[par]+=2*self.step[par]
			baoupup=cambwrap.cambwrap(upup,self.z,signal=self.signalflag).signal

			downdown[par]-=2*self.step[par]
			baodowndown=cambwrap.cambwrap(downdown,self.z,signal=self.signalflag).signal
			
			return (-baoupup+8*baoup-8*baodown+baodowndown)/(12*self.step[par])
		else:
			print('invalid')
			return np.nan

	def secondderivnew(self,par1,par2,pointpar,derivmethod='1st'):
		#par_i is a string
		#derivmethod:
		#	1st: f'=[f(x+h)-f(x-h)]/2h
		#	2nd: f'=[-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)]/12h
		#return first deriv with respect to par
		up=pointpar.copy()
		down=pointpar.copy()
		up[par1]+=self.step[par1]
		baoup=self.firstderivnew(par2,up,derivmethod='2nd')
		down[par1]-=self.step[par1]
		baodown=self.firstderivnew(par2,down,derivmethod='2nd')
		if derivmethod=='1st':
			return (baoup-baodown)/(2*self.step[par1])
		elif derivmethod=='2nd':
			upup=pointpar.copy()
			downdown=pointpar.copy()
			upup[par1]+=2*self.step[par1]
			baoupup=self.firstderivnew(par2,upup,derivmethod='2nd')

			downdown[par1]-=2*self.step[par1]
			baodowndown=self.firstderivnew(par2,downdown,derivmethod='2nd')
			
			return (-baoupup+8*baoup-8*baodown+baodowndown)/(12*self.step[par1])
		else:
			print('invalid')
			return np.nan
	def thirdderivnew(self,par1,par2,par3,pointpar,derivmethod='1st'):
		#par_i is a string
		#derivmethod:
		#	1st: f'=[f(x+h)-f(x-h)]/2h
		#	2nd: f'=[-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)]/12h
		#return first deriv with respect to par
		up=pointpar.copy()
		down=pointpar.copy()
		up[par1]+=self.step[par1]
		baoup=self.secondderivnew(par2,par3,up,derivmethod='2nd')
		down[par1]-=self.step[par1]
		baodown=self.secondderivnew(par2,par3,down,derivmethod='2nd')
		if derivmethod=='1st':
			return (baoup-baodown)/(2*self.step[par1])
		elif derivmethod=='2nd':
			upup=pointpar.copy()
			downdown=pointpar.copy()
			upup[par1]+=2*self.step[par1]
			baoupup=self.secondderivnew(par2,par3,upup,derivmethod='2nd')

			downdown[par1]-=2*self.step[par1]
			baodowndown=self.secondderivnew(par2,par3,downdown,derivmethod='2nd')
			
			return (-baoupup+8*baoup-8*baodown+baodowndown)/(12*self.step[par1])
		else:
			print('invalid')
			return np.nan

	def fourthderivnew(self,par1,par2,par3,par4,pointpar,derivmethod='1st'):
		#par_i is a string
		#derivmethod:
		#	1st: f'=[f(x+h)-f(x-h)]/2h
		#	2nd: f'=[-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)]/12h
		#return first deriv with respect to par
		up=pointpar.copy()
		down=pointpar.copy()
		up[par1]+=self.step[par1]
		baoup=self.thirdderivnew(par2,par3,par4,up,derivmethod='2nd')
		down[par1]-=self.step[par1]
		baodown=self.thirdderivnew(par2,par3,par4,down,derivmethod='2nd')
		if derivmethod=='1st':
			return (baoup-baodown)/(2*self.step[par1])
		elif derivmethod=='2nd':
			upup=pointpar.copy()
			downdown=pointpar.copy()
			upup[par1]+=2*self.step[par1]
			baoupup=self.thirdderivnew(par2,par3,par4,upup,derivmethod='2nd')

			downdown[par1]-=2*self.step[par1]
			baodowndown=self.thirdderivnew(par2,par3,par4,downdown,derivmethod='2nd')
			
			return (-baoupup+8*baoup-8*baodown+baodowndown)/(12*self.step[par1])
		else:
			print('invalid')
			return np.nan



	def getfirstderiv(self,par,pointpar,derivmethod='1st'):
		self.firstderiv={}
		for k in par:
			self.firstderiv[k]=self.firstderivnew(k,pointpar,derivmethod)

	def getsecondderiv(self,par,pointpar,derivmethod='1st'):
		self.secondderiv={}
		for k1 in par:
			for k2 in par:
				if k1+k2 not in self.secondderiv.keys():
					self.secondderiv[k1+k2]=self.secondderivnew(k1,k2,pointpar,derivmethod)
					self.secondderiv[k2+k1]=self.secondderiv[k1+k2]
	def getthirdderiv(self,par,pointpar,derivmethod='1st'):
		self.thirdderiv={}
		for k1 in par:
			for k2 in par:
				for k3 in par:
					if k1+k2+k3 not in self.thirdderiv.keys():
						self.thirdderiv[k1+k2+k3]=self.thirdderivnew(k1,k2,k3,pointpar,derivmethod)
						permut=list(itertools.permutations([k1,k2,k3]))
						for keyname in permut[1:]:
							self.thirdderiv[keyname[0]+keyname[1]+keyname[2]]=self.thirdderiv[k1+k2+k3]
						
	def getfourthderiv(self,par,pointpar,derivmethod='1st'):
		self.fourthderiv={}
		for k1 in par:
			for k2 in par:
				for k3 in par:
					for k4 in par:
						if k1+k2+k3+k4 not in self.fourthderiv.keys():
							self.fourthderiv[k1+k2+k3+k4]=self.fourthderivnew(k1,k2,k3,k4,pointpar,derivmethod)
							permut=list(itertools.permutations([k1,k2,k3,k4]))
							for keyname in permut[1:]:
								self.fourthderiv[keyname[0]+keyname[1]+keyname[2]+keyname[3]]=self.fourthderiv[k1+k2+k3+k4]

	


	def fisher(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_ab=f_{,a}Mf_{,b}
		F=[]
		for k1 in par:
			frow=[]
			
			for k2 in par:
				
				frow.append(np.einsum('i,ij,j',self.firstderiv[k1],np.linalg.inv(self.cov),self.firstderiv[k2]))
			F.append(np.array(frow))
		#myFile = open('F.txt', 'w')

		#np.savetxt(myFile, F)
		return np.array(F)


	def flexion(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_abc=f_{,ab}Mf_{,c}
		F=[]
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
			
				for k3 in par:
				
					fcol.append(np.einsum('i,ij,j',self.secondderiv[k1+k2],np.linalg.inv(self.cov),self.firstderiv[k3]))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)

	def quar(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_abcd=f_{,ab}Mf_{,cd}
		F=[]
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
			
				for k3 in par:
					fnol=[]

					for k4 in par:

				
						fnol.append(np.einsum('i,ij,j',self.secondderiv[k1+k2],np.linalg.inv(self.cov),self.secondderiv[k3+k4]))
					fcol.append(np.array(fnol))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)


	def triple1(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_abcd=f_{,abc}Mf_{,d}
		F=[]
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
			
				for k3 in par:
					fnol=[]

					for k4 in par:

				
						fnol.append(np.einsum('i,ij,j',self.thirdderiv[k1+k2+k3],np.linalg.inv(self.cov),self.firstderiv[k4]))
					fcol.append(np.array(fnol))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)

	def triple2(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_abcde=f_{,abc}Mf_{,de}
		F=[]
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
			
				for k3 in par:
					fnol=[]

					for k4 in par:
						fmol=[]
						for k5 in par:

				
							fmol.append(np.einsum('i,ij,j',self.thirdderiv[k1+k2+k3],np.linalg.inv(self.cov),self.secondderiv[k4+k5]))
						fnol.append(np.array(fmol))
					fcol.append(np.array(fnol))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)

	def triple3(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_abcdef=f_{,abc}Mf_{,def}
		F=[]
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
			
				for k3 in par:
					fnol=[]

					for k4 in par:
						fmol=[]
						for k5 in par:
							flol=[]
							for k6 in par:
								flol.append(np.einsum('i,ij,j',self.thirdderiv[k1+k2+k3],np.linalg.inv(self.cov),self.thirdderiv[k4+k5+k6]))
							fmol.append(np.array(flol))
						fnol.append(np.array(fmol))
					fcol.append(np.array(fnol))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)

	def quartet1(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_abcde=f_{,abcd}Mf_{,e}
		F=[]
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
			
				for k3 in par:
					fnol=[]

					for k4 in par:
						fmol=[]
						for k5 in par:

				
							fmol.append(np.einsum('i,ij,j',self.fourthderiv[k1+k2+k3+k4],np.linalg.inv(self.cov),self.firstderiv[k5]))
						fnol.append(np.array(fmol))
					fcol.append(np.array(fnol))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)

	def quartet2(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_abcdef=f_{,abcd}Mf_{,ef}
		F=[]
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
			
				for k3 in par:
					fnol=[]

					for k4 in par:
						fmol=[]
						for k5 in par:
							flol=[]
							for k6 in par:
								flol.append(np.einsum('i,ij,j',self.fourthderiv[k1+k2+k3+k4],np.linalg.inv(self.cov),self.secondderiv[k5+k6]))
							fmol.append(np.array(flol))
						fnol.append(np.array(fmol))
					fcol.append(np.array(fnol))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)

	def quartet3(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_abcdefgh=f_{,abcd}Mf_{,efgh}
		F=[]
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
			
				for k3 in par:
					fnol=[]

					for k4 in par:
						fmol=[]
						for k5 in par:
							flol=[]
							for k6 in par:
								fpol=[]
								for k7 in par:
									fpol.append(np.einsum('i,ij,j',self.fourthderiv[k1+k2+k3+k4],np.linalg.inv(self.cov),self.thirdderiv[k5+k6+k7]))
								flol.append(np.array(fpol))
							fmol.append(np.array(flol))
						fnol.append(np.array(fmol))
					fcol.append(np.array(fnol))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)

	def quartet4(self,par,derivmethod='1st'):
		#par: a list of string
		#derivmethod: see description in firstderiv() and secondderiv() functions
		#return F_abcdef=f_{,abc}Mf_{,def}
		F=[]
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
			
				for k3 in par:
					fnol=[]

					for k4 in par:
						fmol=[]
						for k5 in par:
							flol=[]
							for k6 in par:
								fpol=[]
								for k7 in par:
									fqol=[]
									for k8 in par:
										fqol.append(np.einsum('i,ij,j',self.fourthderiv[k1+k2+k3+k4],np.linalg.inv(self.cov),self.fourthderiv[k5+k6+k7+k8]))
									fpol.append(np.array(fqol))
								flol.append(np.array(fpol))
							fmol.append(np.array(flol))
						fnol.append(np.array(fmol))
					fcol.append(np.array(fnol))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)


	def extra1(self,par,refsig,derivmethod='1st'):
		F=[]
		diff=self.sigfid-refsig
		for k in par:
			F.append(np.einsum('i,ij,j',self.firstderiv[k],np.linalg.inv(self.cov),diff))
		return np.array(F)

	def extra2(self,par,refsig,derivmethod='1st'):
		F=[]
		diff=self.sigfid-refsig
		for k1 in par:
			frow=[]
			for k2 in par:
				frow.append(np.einsum('i,ij,j',self.secondderiv[k1+k2],np.linalg.inv(self.cov),diff))
			F.append(np.array(frow))
		return np.array(F)

	def extra3(self,par,refsig,derivmethod='1st'):
		F=[]
		diff=self.sigfid-refsig
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
				for k3 in par:
					fcol.append(np.einsum('i,ij,j',self.thirdderiv[k1+k2+k3],np.linalg.inv(self.cov),diff))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)

	def extra4(self,par,refsig,derivmethod='1st'):
		F=[]
		diff=self.sigfid-refsig
		for k1 in par:
			frow=[]
			for k2 in par:
				fcol=[]
				for k3 in par:
					fnol=[]
					for k4 in par:
						fnol.append(np.einsum('i,ij,j',self.fourthderiv[k1+k2+k3+k4],np.linalg.inv(self.cov),diff))
					fcol.append(np.array(fnol))
				frow.append(np.array(fcol))
			F.append(np.array(frow))
		return np.array(F)
"""
	def thirdderiv(self,par1,par2,par3,derivmethod='1st'):
		if (par1==par2) and (par2==par3):
			h2=2*self.step[par1]
			h3=3*self.step[par1]
			h4=4*self.step[par1]
			upx=self.fid.copy()
			upx[par1]+=self.step[par1]
			upxx=self.fid.copy()
			upxx[par1]+=h2
			downx=self.fid.copy()
			downx[par1]-=self.step[par1]
			downxx=self.fid.copy()
			downxx[par1]-=h2
			upxxx=self.fid.copy()
			upxxx[par1]+=h3
			downxxx=self.fid.copy()
			downxxx[par1]-=h3
			upxxxx=self.fid.copy()
			upxxxx[par1]+=h4
			downxxxx=self.fid.copy()
			downxxxx[par1]-=h4

			
				
			baoupx=cambwrap.cambwrap(upx,self.z,signal=self.signalflag).signal#/(self.step[par1]**3)
			baoupxx=cambwrap.cambwrap(upxx,self.z,signal=self.signalflag).signal#/(self.step[par1]**3)
			baodownx=cambwrap.cambwrap(downx,self.z,signal=self.signalflag).signal#/(self.step[par1]**3)
			baodownxx=cambwrap.cambwrap(downxx,self.z,signal=self.signalflag).signal#/(self.step[par1]**3)
			baoupxxx=cambwrap.cambwrap(upxxx,self.z,signal=self.signalflag).signal#/(self.step[par1]**3)
			baoupxxxx=cambwrap.cambwrap(upxxxx,self.z,signal=self.signalflag).signal#/(self.step[par1]**3)
			baodownxxx=cambwrap.cambwrap(downxxx,self.z,signal=self.signalflag).signal#/(self.step[par1]**3)
			baodownxxxx=cambwrap.cambwrap(downxxxx,self.z,signal=self.signalflag).signal#/(self.step[par1]**3)
			

			return (7*baoupxxxx/240-0.3*baoupxxx+169*baoupxx/120-61*baoupx/30+61*baodownx/30-169*baodownxx/120+0.3*baodownxxx-7*baodownxxxx/240)/(self.step[par1]**3)
		else:
			xyz=self.fid.copy()
			xymz=self.fid.copy()
			xmyz=self.fid.copy()
			xmymz=self.fid.copy()
			mxyz=self.fid.copy()
			mxymz=self.fid.copy()
			mxmyz=self.fid.copy()
			mxmymz=self.fid.copy()

			xyz[par1]+=self.step[par1]
			xyz[par2]+=self.step[par2]
			xyz[par3]+=self.step[par3]

			xymz[par1]+=self.step[par1]
			xymz[par2]+=self.step[par2]
			xymz[par3]-=self.step[par3]

			xmyz[par1]+=self.step[par1]
			xmyz[par2]-=self.step[par2]
			xmyz[par3]+=self.step[par3]

			mxyz[par1]-=self.step[par1]
			mxyz[par2]+=self.step[par2]
			mxyz[par3]+=self.step[par3]

			xmymz[par1]+=self.step[par1]
			xmymz[par2]-=self.step[par2]
			xmymz[par3]-=self.step[par3]

			mxymz[par1]-=self.step[par1]
			mxymz[par2]+=self.step[par2]
			mxymz[par3]-=self.step[par3]

			mxmyz[par1]-=self.step[par1]
			mxmyz[par2]-=self.step[par2]
			mxmyz[par3]+=self.step[par3]

			mxmymz[par1]-=self.step[par1]
			mxmymz[par2]-=self.step[par2]
			mxmymz[par3]-=self.step[par3]
			
			baoxyz=cambwrap.cambwrap(xyz,self.z,signal=self.signalflag).signal
			baomxyz=cambwrap.cambwrap(mxyz,self.z,signal=self.signalflag).signal
			baoxmyz=cambwrap.cambwrap(xmyz,self.z,signal=self.signalflag).signal
			baoxymz=cambwrap.cambwrap(xymz,self.z,signal=self.signalflag).signal
			baomxmyz=cambwrap.cambwrap(mxmyz,self.z,signal=self.signalflag).signal
			baomxymz=cambwrap.cambwrap(mxymz,self.z,signal=self.signalflag).signal
			baoxmymz=cambwrap.cambwrap(xmymz,self.z,signal=self.signalflag).signal
			baomxmymz=cambwrap.cambwrap(mxmymz,self.z,signal=self.signalflag).signal


			return (baoxyz-baoxymz-baoxmyz+baoxmymz-baomxyz+baomxymz+baomxmyz-baomxmymz)/(8*self.step[par1]*self.step[par2]*self.step[par3])

"""


"""
	def firstderiv(self,par,derivmethod='1st'):
		#par is a string
		#derivmethod:
		#	1st: f'=[f(x+h)-f(x-h)]/2h
		#	2nd: f'=[-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)]/12h
		#return first deriv with respect to par
		up=self.fid.copy()
		down=self.fid.copy()
		up[par]+=self.step[par]
		baoup=cambwrap.cambwrap(up,self.z,signal=self.signalflag).signal
		down[par]-=self.step[par]
		baodown=cambwrap.cambwrap(down,self.z,signal=self.signalflag).signal
		if derivmethod=='1st':
			return (baoup-baodown)/(2*self.step[par])
		elif derivmethod=='2nd':
			upup=self.fid.copy()
			downdown=self.fid.copy()
			upup[par]+=2*self.step[par]
			baoupup=cambwrap.cambwrap(upup,self.z,signal=self.signalflag).signal

			downdown[par]-=2*self.step[par]
			baodowndown=cambwrap.cambwrap(downdown,self.z,signal=self.signalflag).signal
			
			return (-baoupup+8*baoup-8*baodown+baodowndown)/(12*self.step[par])
		else:
			print('invalid')
			return np.nan
"""


"""
	def secondderiv(self,par1,par2,derivmethod='1st'):
		#par_i is a string
		#derivmethod:
		#	1st: f_{,xx}=[f(x+2h)-2f(x)+f(x-2h)]/4h^2
		#		 f_{,xy}=[f(x+h1,y+h2)-f(x-h1,y+h2)-f(x+h1,y-h2)+f(x-h1,y-h2)]/4h1h2
		#	2nd: f_{,xx}=[-f(x+2h)+16f(x+h)-30f(x)+16f(x-h)-f(x-2h)]/12h^2
		#		 f_{,xy}=[-f(x+2h1.y+2h2)+16f(x+h1,y+h2)
		#				  +f(x+2h1.y-2h2)-16f(x+h1,y-h2)
		#				  +f(x-2h1.y+2h2)-16f(x-h1,y+h2)
		#                 -f(x-2h1.y-2h2)+16f(x-h1,y-h2)]/48h1h2
		#return second deriv with respect to par1 and par2
		upup=self.fid.copy()
		downdown=self.fid.copy()
		updown=self.fid.copy()
		downup=self.fid.copy()
		upup[par1]+=self.step[par1]
		upup[par2]+=self.step[par2]
		baoupup=cambwrap.cambwrap(upup,self.z,signal=self.signalflag).signal

		downdown[par1]-=self.step[par1]
		downdown[par2]-=self.step[par2]
		baodowndown=cambwrap.cambwrap(downdown,self.z,signal=self.signalflag).signal

		updown[par1]+=self.step[par1]
		updown[par2]-=self.step[par2]
		baoupdown=cambwrap.cambwrap(updown,self.z,signal=self.signalflag).signal

		downup[par1]-=self.step[par1]
		downup[par2]+=self.step[par2]
		baodownup=cambwrap.cambwrap(downup,self.z,signal=self.signalflag).signal
		if derivmethod=='1st':
			deriv1=(baoupup-baodownup)/(2*self.step[par1])
			deriv2=(baoupdown-baodowndown)/(2*self.step[par1])
			return (deriv1-deriv2)/(2*self.step[par2])
		elif derivmethod=='2nd':
			if par1==par2:
				h2=2*self.step[par1]
				upx=self.fid.copy()
				upx[par1]+=self.step[par1]
				upxx=self.fid.copy()
				upxx[par1]+=h2
				downx=self.fid.copy()
				downx[par1]-=self.step[par1]
				downxx=self.fid.copy()
				downxx[par1]-=h2
				
				baoupx=cambwrap.cambwrap(upx,self.z,signal=self.signalflag).signal
				baoupxx=cambwrap.cambwrap(upxx,self.z,signal=self.signalflag).signal
				baodownx=cambwrap.cambwrap(downx,self.z,signal=self.signalflag).signal
				baodownxx=cambwrap.cambwrap(downxx,self.z,signal=self.signalflag).signal

				return (-30*self.sigfid+16*baoupx+16*baodownx-baoupxx-baodownxx)/(12*self.step[par1]*self.step[par1])
				
			else:
				upup2=self.fid.copy()
				downdown2=self.fid.copy()
				updown2=self.fid.copy()
				downup2=self.fid.copy()
				upup2[par1]+=2*self.step[par1]
				upup2[par2]+=2*self.step[par2]
				baoupup2=cambwrap.cambwrap(upup2,self.z,signal=self.signalflag).signal

				downdown2[par1]-=2*self.step[par1]
				downdown2[par2]-=2*self.step[par2]
				baodowndown2=cambwrap.cambwrap(downdown2,self.z,signal=self.signalflag).signal

				updown2[par1]+=2*self.step[par1]
				updown2[par2]-=2*self.step[par2]
				baoupdown2=cambwrap.cambwrap(updown2,self.z,signal=self.signalflag).signal

				downup2[par1]-=2*self.step[par1]
				downup2[par2]+=2*self.step[par2]
				baodownup2=cambwrap.cambwrap(downup2,self.z,signal=self.signalflag).signal
				return (-baoupup2+16*baoupup+baoupdown2-16*baoupdown+baodownup2-16*baodownup-baodowndown2+16*baodowndown)/(48*self.step[par1]*self.step[par2])
		else:
			print('invalid')
			return np.nan
"""
	
