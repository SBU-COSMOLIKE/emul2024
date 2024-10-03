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


pertail128res4=np.array([0.6832,0.4409,0.3210,0.3296,0.2731])

pertail128res5=np.array([0.6712,0.4411,0.3931,0.3266,0.1854])

pertail128res6=np.array([1,0.3804,0.3796,0.2889,0.2499])




n_train=np.array([100,200,300,400,530])

plt.plot(n_train,pertail128res4,'b-',lw=2.5,label='ResMLP 4Layer, T=128')

plt.plot(n_train,pertail128res5,'r-',lw=2.5,label='ResMLP 5Layer, T=128')

plt.plot(n_train,pertail128res6,'y-',lw=2.5,label='ResMLP 6Layer, T=128')

plt.xlabel('$N_{\\rm train} / 1000 $',fontsize=18)
plt.ylabel(r'Fraction of points with $\chi^2>0.2$',fontsize=18)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
fs = 9
l = plt.legend(
    fontsize = fs,
    ncol=2,
    loc='upper right',
    frameon=False,
    labelspacing=0.25,
    handletextpad=0.4,
    handlelength=2,
    columnspacing=0.4,
)
plt.xlim(101,499)
plt.ylim(0.15,1)
plt.savefig("fractionvsnumRes.pdf", format="pdf", bbox_inches="tight",dpi=300, pad_inches=0.05)
plt.savefig("fractionvsnumRes.svg", format="svg", bbox_inches="tight",dpi=300, pad_inches=0.05)