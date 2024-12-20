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
xmarkersz = 11
Dmarkersz = 7
yupperlim = 3.6
ydownlim = 0.008
xupperlim = 1700
xdownlim = 80
med128=np.array([0.215, 0.038, 0.019 ,0.011,0.006])

pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])


med128_uni=np.array([12.743,0.125,0.024,0.015,0.016])
pertail128_uni=np.array([1,0.3676,0.0959,0.0574,0.0656])



n_train=np.array([100,200,300,400,530])

n_train_uni=np.array([100,500,1000,1300,1500])
plt.figure(figsize = (3.5,3.5))

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.plot(n_train, 
    pertail128, 
    c='black',
    marker = 'D',
    alpha=1.0,
    lw=3.50,
    label='Gaussian', markersize=Dmarkersz)

plt.plot(n_train_uni, 
    pertail128_uni, 
    c='orchid',
    marker = 'o',
    alpha=1.0,
    lw=3.50,
    label='Uniform', markersize=xmarkersz)

plt.plot(n_train,med128,
    c='black',
    linestyle=(0, (1, 1)),
    marker = 'D',
    alpha=0.5,
    lw=3.50,
    label='_nolegend_', markersize=Dmarkersz)

plt.plot(n_train_uni,med128_uni,
    c='orchid',
    linestyle=(0, (1, 1)),
    marker = 'o',
    alpha=0.5,
    lw=3.50,
    label='_nolegend_', markersize=xmarkersz)


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
arr=np.array([0,0])
plt.plot(arr,arr,c='black',linestyle=(0, (1, 1)),label='$\langle\Delta\chi^2\\rangle_{\\rm med}$')
plt.plot(arr,arr,c='black',label=r'\texthtbardotlessj($\Delta\chi^2>0.2$)')
fs = 13
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
#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.xlabel('$N_{\\rm train} / 1000 $',fontsize=15)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.minorticks_on()
plt.xlim(xdownlim,xupperlim)
plt.ylim(ydownlim,yupperlim)
#plt.legend()
plt.savefig("gaussianvsuni.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("gaussianvsuni.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("gaussianvsuni.jpg", format="jpg", bbox_inches="tight", dpi=300, pad_inches=0.05)