import numpy as np
import matplotlib
from matplotlib import pyplot as plt

#matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux

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
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{tipa}'

plt.figure(figsize = (3.5,3.5))

med128=np.array([0.215, 0.038, 0.019 ,0.011,0.006])

med1283b=np.array([0.217,0.035,0.012,0.012,0.005])

pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])

pertail1283b=np.array([0.5168,0.1645,0.0782,0.0614,0.0567])




n_train=np.array([100,200,300,400,530])
plt.plot(n_train,pertail1283b,c='pink',lw=4.00,label='$N_{\\rm TRF}=3$',marker='o', markersize=10)
plt.plot(n_train,pertail128,c='black',lw=1.50,label='$N_{\\rm TRF}=1$',marker='D', markersize=6)
plt.plot(n_train,med1283b,c='pink',linestyle=(0, (1, 1),),lw=4.00,label='_nolegend_',marker='o', markersize=10)
plt.plot(n_train,med128,c='black',linestyle=(0, (1, 1),),lw=1.50,label='_nolegend_',marker='D', markersize=6)

arr=np.array([0,0])
plt.plot(arr,arr,c='black',linestyle=(0, (1, 1)),label='$\langle\Delta\chi^2\\rangle_{\\rm med}$')
plt.plot(arr,arr,c='black',label=r'\texthtbardotlessj($\Delta\chi^2>0.2$)')
plt.xlabel('$N_{\\rm train} / 1000 $',fontsize=15)
#plt.ylabel(r'\texthtbardotlessj($\Delta\chi^2>0.2$)',fontsize=18)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
fs = 13
l = plt.legend(
    fontsize = fs,
    ncol=1,
    loc='upper right',
    frameon=False,
    labelspacing=0.65,
    handletextpad=0.4,
    handlelength=2,
    columnspacing=0.4,
)
plt.xlim(90,550)
plt.xticks([100,200,300,400, 500])
plt.ylim(0.02,0.8)
plt.savefig("fractionvsnumTRF.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.01)
#plt.savefig("fractionvsnumTRF.svg", format="svg", bbox_inches="tight",dpi=300, pad_inches=0.05)