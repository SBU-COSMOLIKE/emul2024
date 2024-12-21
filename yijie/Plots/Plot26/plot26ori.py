
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

plt.figure(figsize = (3.5,3.5))

ell=np.arange(2,5000,1)

q1=np.load('q1ori.npy',allow_pickle=True)
q2=np.load('q2ori.npy',allow_pickle=True)
q3=np.load('q3ori.npy',allow_pickle=True)

plt.fill_between(ell, q3, color='blue',label=r'$99\%$')
plt.fill_between(ell, q2, color='red',label=r'$95\%$')
plt.fill_between(ell, q1, color='green',label=r'$68\%$')
plt.minorticks_on()
plt.text(500, 37, 'No Removal', fontsize = 22)
plt.legend(loc=9,fontsize=15)
plt.ylabel(r'$|\Delta C_{\ell}^{TT}|/\sigma_{\ell}^{TT}$'+r'$\%$',fontsize=15)
plt.xlabel(r'$\ell$',fontsize=15)
plt.ylim(0,83)
plt.xlim(2,5000)
plt.savefig("oriq.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.01)
