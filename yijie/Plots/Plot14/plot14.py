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

plt.figure(figsize = (4,4))
tail=np.array([0.0607, 0.0295, 0.0161 ,0.0102 ,0.0045 ,0.0033 ,0.0006 ,0.0001])


temp=np.array([128,64,32,16,8,4,2,1])

plt.plot(temp, tail,'b-',marker = 'D', markersize=5)

plt.xlabel('Temperature',fontsize=18)
plt.ylabel(r'\texthtbardotlessj($\Delta\chi^2>0.2$)',fontsize=18)
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.0001,0.1)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)

plt.savefig("fractionvsT.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.05)