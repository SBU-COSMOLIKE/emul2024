
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

#matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{tipa}'

xmarkersz = 11
Dmarkersz = 7
yupperlim = 3
ydownlim = 0.008
xupperlim = 550
xdownlim = 90
fig, axs = plt.subplots(1, 1, figsize=(3.5, 3.5),sharey=True) 
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

axs.plot(n_train, 
    pertail128, 
    c='blue',
    marker='o',
    alpha=1.0,
    lw=3.50,
    label='TRF', markersize=xmarkersz)

axs.plot(n_train, 
    pertail128cnn,
    c='firebrick', 
    marker='D',
    alpha=1.0,
    lw=1.25,
    label='CNN', markersize=Dmarkersz)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


axs.plot(n_train,med128,
    c='blue',
    linestyle=(0, (1, 1),),
    marker='o',
    alpha=0.5,
    lw=3.50,
    label='_nolegend_', markersize=xmarkersz)

axs.plot(n_train,med128cnn,
    c='firebrick', 
    linestyle=(0, (1, 1)),
    marker='D',
    alpha=0.5,
    lw=1.25,
    label='_nolegend_', markersize=Dmarkersz)

arr=np.array([0,0])
axs.plot(arr,arr,c='black',linestyle=(0, (1, 1)),label='$\langle\Delta\chi^2\\rangle_{\\rm med}$')
axs.plot(arr,arr,c='black',label=r'\texthtbardotlessj($\Delta\chi^2>0.2$)')
fs = 13
l = axs.legend(
    fontsize = fs,
    ncol=1,
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

axs.set_xlabel('$N_{\\rm train} / 1000 $',fontsize=14)
axs.set_yscale('log')
axs.tick_params(axis='x', labelsize=16)
axs.tick_params(axis='y', labelsize=16)
axs.set_xlim(xdownlim,xupperlim)
axs.set_ylim(ydownlim,yupperlim)
axs.text(0.05, 0.05, '$T_{\\rm test} = 128$', transform=axs.transAxes, fontsize=15,
        verticalalignment='bottom', bbox=None,c='black')
#plt.legend()


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
"""
axs[1].plot(n_train,med64,
    c='firebrick',
    linestyle=(0, (1, 1)),
    marker='o',
    alpha=0.5, 
    lw=3.50,
    label='_nolegend_', markersize=xmarkersz)


axs[1].plot(n_train,med64sqrt,
    c='firebrick', 
    linestyle=(0, (1, 1)),
    marker='D',
    alpha=0.5,
    lw=1.25,
    label='_nolegend_', markersize=Dmarkersz)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
axs[1].plot(n_train, pertail64, 
    c='firebrick',
    marker='o',
    alpha=1.0,
    lw=3.50,
    label='_nolegend_', markersize=xmarkersz)

axs[1].plot(n_train, 
    pertail64sqrt, 
    'firebrick',
    marker='D',
    alpha=1.0,
    lw=1.25,
    label='_nolegend_', markersize=Dmarkersz)
axs[1].text(0.75, 0.05, '$T = 64$', transform=axs[1].transAxes, fontsize=18,
        verticalalignment='bottom', bbox=None,c='firebrick')
arr=np.array([0,0])
axs[1].plot(arr,arr,c='firebrick',linestyle=(0, (1, 1)),label='$\langle\Delta\chi^2\\rangle_{med}$')
axs[1].plot(arr,arr,c='firebrick',label=r'\texthtbardotlessj($\Delta\chi^2>0.2$)')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

l = axs[1].legend(
    fontsize = fs,
    ncol=1,
    loc='upper right',
    frameon=False,
    labelspacing=0.25,
    handletextpad=0.4,
    handlelength=2,
    columnspacing=0.4,
)
axs[1].set_xlabel('$N_{\\rm train} / 1000 $',fontsize=14)
axs[1].set_yscale('log')
axs[1].tick_params(axis='x', labelsize=16)
axs[1].tick_params(axis='y', labelsize=16)
axs[1].set_xlim(xdownlim,xupperlim)
axs[1].set_ylim(ydownlim,yupperlim)
#plt.legend()
"""
fig.savefig("plot22new.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
fig.savefig("plot22new.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
