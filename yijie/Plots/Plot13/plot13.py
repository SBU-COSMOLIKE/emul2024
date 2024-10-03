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


d1=np.load('cos_pkc_T256_cut.npy',allow_pickle=True)[:5000]
d2=np.load('cos_pkc_T128_cut.npy',allow_pickle=True)[:5000]
d3=np.load('cos_pkc_T64_cut.npy',allow_pickle=True)[:5000]

z=np.ones(5000)
fig, axes = plt.subplots(figsize=(35, 35), sharex=False, sharey=False, ncols=3, nrows=3)
name=[r'$\Omega_bh^2$',r'$\Omega_ch^2$',r'$H_0$',r'$\tau$',r'$n_s$',r'$\log{10^{10}A_s}$']
norm = plt.Normalize(z.min(), z.max())
cmap = plt.get_cmap("viridis")
q=np.array([0,2,5])

for i in range(3):
    for j in range(3):
        if i<=j:
            axes[j, i].axis('off')
        else:
            scatter=axes[j, i].scatter(d1[:,q[j]], d1[:,q[i]],
           linewidths=1, alpha=0.7,
           edgecolor='k',
           s = 20,
           c=z,cmap='viridis')
            scatter2=axes[j, i].scatter(d2[:,q[j]], d2[:,q[i]],
           linewidths=1, alpha=0.7,
           edgecolor='r',
           s = 20,
           c=z,cmap='viridis')
            scatter3=axes[j, i].scatter(d3[:,q[j]], d3[:,q[i]],
           linewidths=1, alpha=0.7,
           edgecolor='b',
           s = 20,
           c=z,cmap='viridis')
            
            axes[j,i].set_xlabel(name[q[j]],fontsize=54)
            axes[j,i].set_ylabel(name[q[i]],fontsize=54)
            axes[j,i].tick_params(axis='x', labelsize=50)
            axes[j,i].tick_params(axis='y', labelsize=50)
            scatter.set_rasterized(True)
            scatter2.set_rasterized(True)
            scatter3.set_rasterized(True)
            


#cbar = fig.colorbar(sm, ax=axes[:,1],location='left')
#cbar.ax.tick_params(labelsize=35)
#cbar.ax.yaxis.get_offset_text().set_fontsize(25)   

fig.savefig("trainvstest.pdf", format="pdf", bbox_inches="tight", dpi=200, pad_inches=0.05)
