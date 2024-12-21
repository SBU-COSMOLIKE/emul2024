
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator


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

pred=np.load('cos_TE_pred.npy',allow_pickle=True)
camb=np.load('cos_TE_true.npy',allow_pickle=True)
ell=np.arange(2,5000,1)
factor=ell*(ell+1)/2/np.pi

fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex='all')
axs[1].plot(ell,1/ell)
color=['firebrick','grey','steelblue','darkmagenta','yellowgreen','brown',
		'olivedrab','plum','seagreen','royalblue','moccasin','tan',
		'aqua','navy','crimson','coral','deeppink']
for i in range(len(pred)):
	axs[0].plot(ell,pred[i]*factor,linestyle='dashed',c=color[i],lw=2.5)
	axs[0].plot(ell,camb[i]*factor,c=color[i])
	#axs[1].plot(ell,abs((pred[i]-camb[i])/camb[i]))

for i in range(len(pred)):
	
	axs[1].plot(ell,abs((pred[i]-camb[i])/camb[i]),c=color[i],linestyle='dashed',lw=2.5)

arr=np.array([0])
axs[0].plot(arr,arr,c='black',linestyle='dashed',label='Emul',lw=2.5)
axs[0].plot(arr,arr,c='black',label='CAMB')
axs[1].text(2000, 1e-3, '$1/\ell$', fontsize = 22)
axs[0].set_yscale('symlog')
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_xlim(2,5000)
#axs[0].set_xticklabels([])

axs[0].label_outer()
axs[1].set_yticks([1e-5,1e-4,1e-3])
axs[1].set_ylim(1e-5,1e-2)
axs[1].tick_params(axis='x',which='both', direction='inout', labelsize=16,top=True, labeltop=False, bottom=True, labelbottom=True)
axs[0].tick_params(axis='x',which='both', direction='inout', labelsize=16,top=True, labeltop=False, bottom=True, labelbottom=True)
axs[0].tick_params(axis='y',which='both', direction='inout', left=True,right=True,labelright=False)
axs[1].tick_params(axis='y',which='both', direction='inout', left=True,right=True,labelright=False)
axs[0].legend(fontsize=16)
axs[0].yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=1, base=10,subs = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)))
#axs[1].yaxis.set_minor_locator(AutoMinorLocator())
axs[0].set_ylabel(r'$\ell(\ell+1)C^{\rm TE}_{\ell}/2\pi[\mu K^2]$',fontsize=16)
axs[1].set_ylabel(r'$\Delta C^{\rm TE}_{\ell}/C_{\ell}^{\rm TE,CAMB}$',fontsize=16)
axs[1].set_xlabel(r'$\ell$',fontsize=14)

#plt.setp(axs[0].get_xticklabels(),visible=True)
plt.subplots_adjust(hspace=0)

fig.savefig("TEcomp.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.01)