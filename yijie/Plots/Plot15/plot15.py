import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'


plt.figure(figsize = (5,5))


ell=np.arange(2,5000,1)

q1TT=np.load('q1TT.npy',allow_pickle=True)
q2TT=np.load('q2TT.npy',allow_pickle=True)
q3TT=np.load('q3TT.npy',allow_pickle=True)

q1TE=np.load('q1TE.npy',allow_pickle=True)
q2TE=np.load('q2TE.npy',allow_pickle=True)
q3TE=np.load('q3TE.npy',allow_pickle=True)

q1EE=np.load('q1EE.npy',allow_pickle=True)
q2EE=np.load('q2EE.npy',allow_pickle=True)
q3EE=np.load('q3EE.npy',allow_pickle=True)

plt.fill_between(ell, q3TT, color='blue',label=r'$99\%$')
plt.fill_between(ell, q2TT, color='red',label=r'$95\%$')
plt.fill_between(ell, q1TT, color='green',label=r'$68\%$')
plt.legend(loc=9)
plt.ylabel(r'$|\Delta C_{\ell}^{TT}|/\sigma_{\ell}^{TT}$',fontsize=18)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlabel(r'$\ell$',fontsize=18)

plt.savefig("TTq.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.05)
plt.savefig("TTq.svg", format="svg", bbox_inches="tight",dpi=300, pad_inches=0.05)

plt.clf()

plt.fill_between(ell, q3TE, color='blue',label=r'$99\%$')
plt.fill_between(ell, q2TE, color='red',label=r'$95\%$')
plt.fill_between(ell, q1TE, color='green',label=r'$68\%$')
plt.legend(loc=9)
plt.ylabel(r'$|\Delta C_{\ell}^{TE}|/\sigma_{\ell}^{TE}$',fontsize=18)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlabel(r'$\ell$',fontsize=18)

plt.savefig("TEq.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.05)
plt.savefig("TEq.svg", format="svg", bbox_inches="tight",dpi=300, pad_inches=0.05)

plt.clf()


plt.fill_between(ell, q3EE, color='blue',label=r'$99\%$')
plt.fill_between(ell, q2EE, color='red',label=r'$95\%$')
plt.fill_between(ell, q1EE, color='green',label=r'$68\%$')
plt.legend(loc=9)
plt.ylabel(r'$|\Delta C_{\ell}^{EE}|/\sigma_{\ell}^{EE}$',fontsize=18)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlabel(r'$\ell$',fontsize=18)

plt.savefig("EEq.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.05)
plt.savefig("EEq.svg", format="svg", bbox_inches="tight",dpi=300, pad_inches=0.05)

plt.clf()