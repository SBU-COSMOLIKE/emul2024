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


pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])

pertail1283b=np.array([0.5168,0.1645,0.0782,0.0614,0.0567])




n_train=np.array([100,200,300,400,530])

plt.plot(n_train,pertail128,'b-',lw=3.50,label='TRF 1Block, T=128',marker='D', markersize=5)

plt.plot(n_train,pertail1283b,'r-',lw=3.50,label='TRF 3Block, T=128',marker='D', markersize=5)

plt.xlabel('$N_{\\rm train} / 1000 $')
plt.ylabel(r'\texthtbardotlessj($\Delta\chi^2>0.2$)',fontsize=18)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
fs = 11
l = plt.legend(
    fontsize = fs,
    ncol=1,
    loc='upper right',
    frameon=False,
    labelspacing=0.25,
    handletextpad=0.4,
    handlelength=2,
    columnspacing=0.4,
)
plt.xlim(101,499)
plt.ylim(0.05,0.6)
plt.savefig("fractionvsnumTRF.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.05)
#plt.savefig("fractionvsnumTRF.svg", format="svg", bbox_inches="tight",dpi=300, pad_inches=0.05)