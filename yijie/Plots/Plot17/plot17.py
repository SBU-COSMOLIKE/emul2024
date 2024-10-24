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


med128=np.array([0.215, 0.038, 0.019 ,0.011,0.006])

pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])


med128_uni=np.array([0.562,0.380,0.172,0.107])
pertail128_uni=np.array([0.8383,0.6708,0.4666,0.3581])



n_train=np.array([100,200,300,400,530])

n_train_uni=np.array([200,300,430,530])
plt.figure(figsize = (3.5,3.5))

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.plot(n_train, 
    pertail128, 
    c='blue',
    marker = None,
    alpha=1.0,
    lw=3.50,
    label='Gaussian')

plt.plot(n_train_uni, 
    pertail128_uni, 
    c='red',
    marker = None,
    alpha=1.0,
    lw=3.50,
    label='Uniform')

plt.plot(n_train,med128,
    c='blue',
    linestyle=(0, (1, 1)),
    marker = None,
    alpha=0.5,
    lw=3.50,
    label='_nolegend_')

plt.plot(n_train_uni,med128_uni,
    c='red',
    linestyle=(0, (1, 1)),
    marker = None,
    alpha=0.5,
    lw=3.50,
    label='_nolegend_')


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

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
#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.xlabel('$N_{\\rm train} / 1000 $')
plt.yscale('log')
plt.xlim(201,499)
plt.ylim(0.0099,1.01)
#plt.legend()
plt.savefig("gaussianvsuni.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
