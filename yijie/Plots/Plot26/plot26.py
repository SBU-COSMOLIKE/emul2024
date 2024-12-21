
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# GENERAL PLOT OPTIONS
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
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16

matplotlib.use('TKAgg')

fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex='all')

ell=np.arange(2,5000,1)

q1tail=np.load('q1tail.npy',allow_pickle=True)
q2tail=np.load('q2tail.npy',allow_pickle=True)
q3tail=np.load('q3tail.npy',allow_pickle=True)

q1ori=np.load('q1ori.npy',allow_pickle=True)
q2ori=np.load('q2ori.npy',allow_pickle=True)
q3ori=np.load('q3ori.npy',allow_pickle=True)

axs[0].fill_between(ell, q3tail, color='blue',label=r'$99\%$')
axs[0].fill_between(ell, q2tail, color='red',label=r'$95\%$')
axs[0].fill_between(ell, q1tail, color='green',label=r'$68\%$')
axs[0].minorticks_on()
axs[0].text(1500, 35, 'Removal', fontsize = 22)
axs[0].legend(loc=9,fontsize=15)
axs[0].set_ylabel(r'$|\Delta C_{\ell}^{TT}|/\sigma_{\ell}^{TT}$'+r'$\%$',fontsize=15)
axs[0].set_xlabel(r'$\ell$',fontsize=15)
axs[0].set_ylim(0,89)
axs[0].set_xlim(2,5000)

axs[1].fill_between(ell, q3ori, color='blue',label=r'$99\%$')
axs[1].fill_between(ell, q2ori, color='red',label=r'$95\%$')
axs[1].fill_between(ell, q1ori, color='green',label=r'$68\%$')
axs[1].minorticks_on()
axs[1].text(500, 37, 'No Removal', fontsize = 22)
axs[1].legend(loc=9,fontsize=15)
axs[1].set_ylabel(r'$|\Delta C_{\ell}^{TT}|/\sigma_{\ell}^{TT}$'+r'$\%$',fontsize=15)
axs[1].set_xlabel(r'$\ell$',fontsize=15)
axs[1].set_ylim(0,89)
axs[1].set_xlim(2,5000)
plt.subplots_adjust(hspace=0)
fig.savefig("tailoriq.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.01)
