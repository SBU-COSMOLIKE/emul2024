
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

med128=np.array([0.215, 0.038, 0.019 ,0.011,0.006])
med128cnn=np.array([2.844,0.118,0.021,0.014,0.005])
pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])
pertail128cnn=np.array([0.9982,0.382,0.1232,0.0932,0.0582])
med64=np.array([0.138,0.029,0.015,0.008,0.004])
med64cnn=np.array([1.708,0.083,0.014,0.010,0.004])
pertail64=np.array([0.3907,0.0552,0.0399,0.0315,0.0295])
pertail64cnn=np.array([0.9996,0.2316,0.0392,0.0327,0.0291])
n_train=np.array([100,200,300,400,530])

matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux

plt.figure(figsize = (3.5,3.5))

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.plot(n_train, 
    pertail128, 
    c='blue',
    marker='D',
    alpha=1.0,
    lw=3.50,
    label='TRF', markersize=5)

plt.plot(n_train, 
    pertail128cnn,
    c='blue', 
    marker='D',
    alpha=1.0,
    lw=1.25,
    label='CNN', markersize=5)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


plt.plot(n_train,med128,
    c='blue',
    linestyle=(0, (1, 1),),
    marker='D',
    alpha=0.5,
    lw=3.50,
    label='_nolegend_', markersize=5)

plt.plot(n_train,med128cnn,
    c='blue', 
    linestyle=(0, (1, 1)),
    marker='D',
    alpha=0.5,
    lw=1.25,
    label='_nolegend_', markersize=5)


fs = 12
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
#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.xlabel('$N_{\\rm train} / 1000 $',fontsize=14)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlim(101,499)
plt.ylim(0.0099,1.01)
#plt.legend()
plt.savefig("TRFCNNT128.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("TRFCNNT128.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.clf()

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
plt.figure(figsize = (3.5,3.5))
plt.plot(n_train,med64,
    c='firebrick',
    linestyle=(0, (1, 1)),
    marker='x',
    alpha=0.5, 
    lw=3.50,
    label='_nolegend_', markersize=5)


plt.plot(n_train,med64cnn,
    c='firebrick', 
    linestyle=(0, (1, 1)),
    marker='x',
    alpha=0.5,
    lw=1.25,
    label='_nolegend_', markersize=5)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
plt.plot(n_train, pertail64, 
    c='firebrick',
    marker='x',
    alpha=1.0,
    lw=3.50,
    label='_nolegend_', markersize=5)

plt.plot(n_train, 
    pertail64cnn, 
    'firebrick',
    marker='x',
    alpha=1.0,
    lw=1.25,
    label='_nolegend_', markersize=5)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.xlabel('$N_{\\rm train} / 1000 $',fontsize=14)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xlim(101,499)
plt.ylim(0.0099,1.01)
#plt.legend()
plt.savefig("TRFCNNT64.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("TRFCNNT64.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
