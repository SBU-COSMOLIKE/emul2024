
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

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

chi2out=np.load('chiout.npy',allow_pickle=True)
outlier=np.load('outliers.npy',allow_pickle=True)

matplotlib.use('TKAgg')

plt.figure(figsize = (4,3.5))
name=[r'$\Omega_b$',r'$\Omega_c$',r'$H_0$',r'$\tau$',r'$n_s$',r'$\log{10^{10}A_s}$']
norm = plt.Normalize(chi2out.min(),chi2out.max())
cmap = plt.get_cmap("viridis")

plt.scatter(outlier[:,5], outlier[:,4],
           linewidths=1, alpha=0.7,
           edgecolor='k',
           s = 20,
           c=chi2out,cmap='viridis')

plt.xlabel(name[5],fontsize=14)
plt.ylabel(name[4],fontsize=14)
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)
plt.xticks([2,2.5,3,3.5,4])
plt.colorbar()

plt.savefig("outuni.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.01)