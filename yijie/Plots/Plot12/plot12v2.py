
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
med128res=np.array([0.389,0.123 ,0.089,0.080,0.049 ])
pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])
pertail128res=np.array([0.6712,0.3804,0.3210,0.2889,0.1854])
med64=np.array([0.138,0.029,0.015,0.008,0.004])
med64res=np.array([0.261,0.088,0.064,0.055,0.037])
pertail64=np.array([0.3907,0.0552,0.0399,0.0315,0.0295])
pertail64res=np.array([0.5939,0.2408,0.1725,0.1495,0.0676])
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
    pertail128res,
    c='blue', 
    marker='D',
    alpha=1.0,
    lw=1.25,
    label='ResMLP', markersize=5)

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

plt.plot(n_train,med128res,
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
plt.savefig("plot12T128.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("plot12T128.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
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


plt.plot(n_train,med64res,
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
    pertail64res, 
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
plt.savefig("plot12T64.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("plot12T64.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
