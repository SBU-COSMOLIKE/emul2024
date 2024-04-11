import numpy as np
import scipy.integrate as integrate

from scipy.fft import fft, dct,dst
from numpy.polynomial import chebyshev as Chev


def chebypoints(n, lb,ub):
	#n: degree of chebyshev polynomials
	#lb,ub: lower bound, upper bound, ub>lb

	tt=np.linspace(0,np.pi,n+1)
	zz=np.exp(1j*tt)
	ll=zz.real
	ell=(ub-lb)/2*ll+(ub+lb)/2

	return ll, ell #return ll \in [-1,1] and ell \in [lb,ub]

def chebytransform(n,f):
	#n: degree of chebyshev polynomial, we actually do n+1
	#f: the function that we are approximating
	

	coef=dct(f)/(n+1) #normalization
	coef[0]/=2 #a_0 requires another 1/2 for normalization


	chevfunc=Chev.Chebyshev(coef) #return the chebyshev polynomial
	return chevfunc




