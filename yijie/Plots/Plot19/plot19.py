
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
#matplotlib.rcParams['text.usetex'] = True

med128=np.array([0.215, 0.038, 0.019 ,0.011,0.006])
med128sqrt=np.array([42.169,3.538,0.332,0.321,0.157])
pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])
pertail128sqrt=np.array([1,1,0.7095,0.6601,0.4204])
med64=np.array([0.138,0.029,0.015,0.008,0.004])
med64sqrt=np.array([26.443,2.081,0.225,0.208,0.123])
pertail64=np.array([.3907,0.0552,0.0399,0.0315,0.0295])
pertail64sqrt=np.array([1,1,0.5630,0.5147,0.3011])
n_train=np.array([100,200,300,400,530])
matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux

plt.figure(figsize = (3.5,3.5))

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.plot(n_train, 
    pertail128, 
    c='blue',
    marker = 'D', markersize=5,
    alpha=1.0,
    lw=3.50,
    label='$L=\sqrt{\Delta\widetilde{\chi}^2}$')

plt.plot(n_train, 
    pertail128sqrt,
    c='blue', 
    marker = 'D', markersize=5,
    alpha=1.0,
    lw=1.50,
    label='$L=\sqrt{1+2\Delta\chi^2}$')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------



plt.plot(n_train,med128,
    c='blue',
    linestyle=(0, (1, 1)),
    marker = 'D', markersize=5,
    alpha=0.5,
    lw=3.50,
    label='_nolegend_')

plt.plot(n_train,med128sqrt,
    c='blue', 
    linestyle=(0, (1, 1)),
    marker = 'D', markersize=5,
    alpha=0.5,
    lw=1.50,
    label='_nolegend_')

fs = 12
l = plt.legend(
    fontsize = fs,
    ncol=1,
    loc='upper right',
    frameon=False,
    labelspacing=0.25,
    handletextpad=0.4,
    handlelength=2,
    columnspacing=0.35,
)
#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.xlabel('$N_{\\rm train} / 1000 $',fontsize=15)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlim(101,499)
plt.ylim(0.0099,4.9)
#plt.legend()
plt.savefig("rescalevssqrtT128.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("rescalevssqrtT128.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("rescalevssqrtT128.jpg", format="jpg", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.clf()


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.plot(n_train, pertail64, 
    c='firebrick',
    marker = 'x', markersize=5,
    alpha=0.7,
    lw=3.50,
    label='_nolegend_')

plt.plot(n_train, 
    pertail64sqrt, 
    'firebrick',
    marker = 'x', markersize=5,
    alpha=0.7,
    lw=1.50,
    label='_nolegend_')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.plot(n_train,med64,
    c='firebrick',
    linestyle=(0, (1, 1)),
    marker = 'x', markersize=5,
    alpha=0.3, 
    lw=3.50,
    label='_nolegend_')


plt.plot(n_train,med64sqrt,
    c='firebrick', 
    linestyle=(0, (1, 1)),
    marker = 'x', markersize=5,
    alpha=0.3,
    lw=1.50,
    label='_nolegend_')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.xlabel('$N_{\\rm train} / 1000 $',fontsize=15)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlim(101,499)
plt.ylim(0.0099,4.9)
#plt.legend()
plt.savefig("rescalevssqrtT64.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("rescalevssqrtT64.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("rescalevssqrtT64.jpg", format="jpg", bbox_inches="tight", dpi=300, pad_inches=0.05)
