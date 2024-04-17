import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fft, dct,dst
from numpy.polynomial import chebyshev as Chev
from scipy.interpolate import interp1d
matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux


n  = 500 #number of Chebyshev points that you can change
tt = np.linspace(0,np.pi,n+1)
zz = np.exp(1j*tt)
ll = zz.real

ub = 5000 #upper bound for ell
lb = 2    #lower bound for ell

ell = (ub-lb)/2*ll+(ub+lb)/2

camb_ell = np.arange(2,5300,1)
TT = np.load('testTT.npy',allow_pickle=True)
covinv = np.load('cosvarinvTT.npy',allow_pickle=True)[:(ub-lb),:(ub-lb)]
TTfunc = interp1d(camb_ell,TT,kind='cubic')

#CB decomposition
f = TTfunc(ell)*ell*(ell+1)*ell**1.5 #Note I have C_l*ell*(ell+1)*ell^1.5 here to help CB decompose more accurately 
#discrete cosine transformation from scipy
coef = dct(f,type=1)/(n)
coef[0]/=2

test_ell = np.arange(lb,ub,1)
test_ll = (test_ell-(ub+lb)/2)/((ub-lb)/2)
testspec = Chev.chebval(test_ll, coef)
cambspec = TT[:(ub-lb)]*test_ell*(test_ell+1)*test_ell**1.5

plt.plot(test_ell,testspec/test_ell**1.5,'.',label='cheb')
plt.plot(test_ell,cambspec/test_ell**1.5,label='camb')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.ylabel(r'$D_\ell^{TT}$')
plt.xlabel(r'$\ell$')
plt.savefig("cheby"+str(n)+".pdf", format="pdf", bbox_inches="tight")

plt.clf()

diff = (testspec-cambspec)/test_ell/(test_ell+1)/test_ell**1.5
chi = diff*np.sqrt(np.diag(covinv))
chi2 = chi**2
chil = []
for i in range(len(chi2)):
    chil.append(np.sum(chi2[:i]))
chil = np.array(chil)
plt.plot(test_ell,chil,label=str(n)+' points')
#plt.xscale('log')
plt.legend()
#plt.ylim(-0.003,0.003)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\chi^2(\ell)$')
plt.savefig("cheby"+str(n)+"error.pdf", format="pdf", bbox_inches="tight")

