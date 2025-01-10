
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
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{tipa}'


med128l4=np.array([0.215, 0.038, 0.019 ,0.011,0.006])
med128l3=np.array([42.169,3.538,0.332,0.321,0.157])
pertail128l4=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])
pertail128l3=np.array([1,1,0.7095,0.6601,0.4204])
med128l2=np.array([0.898,0.039,0.037,0.036,0.018])
pertail128l2=np.array([0.9634,0.1374,0.1344,0.1174,0.0721])
med128l5=np.array([1.946,0.226,0.120,0.042,0.035])
pertail128l5=np.array([0.9894,0.5292,0.3861,0.1765,0.1534])
n_train=np.array([100,200,300,400,530])
#matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux

xmarkersz = 11
Dmarkersz = 7
yupperlim = 1.25
ydownlim = 0.05
xupperlim = 550
xdownlim = 90
fig, axs = plt.subplots(1, 1, figsize=(3.5, 3.5),sharey=True) 
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

axs.plot(n_train, 
    pertail128l2, 
    c='blue',
    marker='o',
    alpha=1.0,
    lw=3.50,
    markersize=xmarkersz)

axs.plot(n_train, 
    pertail128l3,
    c='firebrick', 
    marker='D',
    alpha=1.0,
    lw=3,
    markersize=Dmarkersz)
axs.plot(n_train, 
    pertail128l4, 
    c='grey',
    marker='o',
    alpha=1.0,
    lw=2.50,
    markersize=xmarkersz)

axs.plot(n_train, 
    pertail128l5,
    c='pink', 
    marker='D',
    alpha=1.0,
    lw=2,
    markersize=Dmarkersz)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

axs.set_xlabel('$N_{\\rm train} / 1000 $',fontsize=14)
axs.set_ylabel(r'\texthtbardotlessj($\Delta\chi^2>0.2$)',fontsize=18)
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
fig.savefig("plot19newtail.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
fig.savefig("plot19newtail.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
