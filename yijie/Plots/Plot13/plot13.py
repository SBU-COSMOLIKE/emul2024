import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux

d1=np.load('cos_pkc_T256_cut.npy',allow_pickle=True)[:5000]
d2=np.load('cos_pkc_T128_cut.npy',allow_pickle=True)[:5000]
d3=np.load('cos_pkc_T64_cut.npy',allow_pickle=True)[:5000]

z=np.ones(5000)
fig, axes = plt.subplots(figsize=(50, 50), sharex=False, sharey=False, ncols=6, nrows=6)
name=[r'$\Omega_b$',r'$\Omega_c$',r'$H_0$',r'$\tau$',r'$n_s$',r'$\log{10^{10}A_s}$']
norm = plt.Normalize(z.min(), z.max())
cmap = plt.get_cmap("viridis")


for i in range(6):
    for j in range(6):
        if i<=j:
            axes[j, i].axis('off')
        else:
            scatter=axes[j, i].scatter(d1[:,j], d1[:,i],
           linewidths=1, alpha=0.7,
           edgecolor='k',
           s = 20,
           c=z,cmap='viridis')
            scatter2=axes[j, i].scatter(d2[:,j], d2[:,i],
           linewidths=1, alpha=0.7,
           edgecolor='r',
           s = 20,
           c=z,cmap='viridis')
            scatter3=axes[j, i].scatter(d3[:,j], d3[:,i],
           linewidths=1, alpha=0.7,
           edgecolor='g',
           s = 20,
           c=z,cmap='viridis')
            
            axes[j,i].set_xlabel(name[j],fontsize=24)
            axes[j,i].set_ylabel(name[i],fontsize=24)
            axes[j,i].tick_params(axis='x', labelsize=20)
            axes[j,i].tick_params(axis='y', labelsize=20)
            scatter.set_rasterized(True)
            scatter2.set_rasterized(True)
            scatter3.set_rasterized(True)
            


#cbar = fig.colorbar(sm, ax=axes[:,1],location='left')
#cbar.ax.tick_params(labelsize=35)
#cbar.ax.yaxis.get_offset_text().set_fontsize(25)   

fig.savefig("trainvstest.pdf", format="pdf", bbox_inches="tight")
