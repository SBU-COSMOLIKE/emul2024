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



pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])
pertail128_lin=np.array([0.9846, 0.63,0.3675 , 0.2103, 0.2138])
pertail128_lsh=np.array([0.9988,0.9423,0.3008,0.147,0.0963])
pertail128_aft=np.array([0.6033,0.2176,0.1098,0.0938,0.0575])
n_train=np.array([100,200,300,400,530])

plt.figure(figsize = (3.5,3.5))

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
marksz=9
plt.plot(n_train, 
    pertail128, 
    c='black',
    marker = 'D',
    alpha=1.0,
    lw=3.50,
    label='Dot Product', markersize=marksz)

plt.plot(n_train, 
    pertail128_lin, 
    c='orchid',
    marker = '*',
    alpha=1.0,
    lw=0.50,
    label='Linear', markersize= marksz)

plt.plot(n_train, 
    pertail128_lsh, 
    c='purple',
    marker = '^',
    alpha=1.0,
    lw=1.50,
    label='LSH', markersize=marksz)

plt.plot(n_train, 
    pertail128_aft, 
    c='darkcyan',
    marker = 'o',
    alpha=1.0,
    lw=2.50,
    label='AFT', markersize=7)


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

fs = 11
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
#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.xlabel('$N_{\\rm train} / 1000 $',fontsize=15)
plt.ylabel(r'\texthtbardotlessj($\Delta\chi^2>0.2$)',fontsize=18)
plt.yscale('log')
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.xticks([100,200,300,400, 500])
plt.ylim(0.05,1.6)
#plt.legend()
plt.savefig("att.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("att.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)
plt.savefig("att.jpg", format="jpg", bbox_inches="tight", dpi=300, pad_inches=0.05)