
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
med128tanh=np.array([8.369,2.145,0.605,0.188,0.615])
pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])
pertail128tanh=np.array([1,0.9643,0.7521,0.4879,0.7275])
med64=np.array([0.138,0.029,0.015,0.008,0.004])
med64tanh=np.array([6.214,1.605,0.445,0.142,0.449])
pertail64=np.array([.3907,0.0552,0.0399,0.0315,0.0295])
pertail64tanh=np.array([1,0.9823,0.7319,0.4140,0.6920])
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
    label='H(x)')

plt.plot(n_train, 
    pertail128tanh,
    c='blue', 
    marker = 'D', markersize=5,
    alpha=1.0,
    lw=1.50,
    label='Tanh')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------



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

plt.plot(n_train,med128tanh,
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
    columnspacing=0.4,
)
#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.xlabel('$N_{\\rm train} / 1000 $')
plt.yscale('log')
plt.xlim(101,499)
plt.ylim(0.0099,2.2)
#plt.legend()
plt.savefig("tanhvshxT128.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)

plt.savefig("tanhvshxT128.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)

plt.savefig("tanhvshxT128.jpg", format="jpg", bbox_inches="tight", dpi=300, pad_inches=0.05)

plt.clf()

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


plt.plot(n_train,med64tanh,
    c='firebrick', 
    linestyle=(0, (1, 1)),
    marker = 'x', markersize=5,
    alpha=0.3,
    lw=1.50,
    label='_nolegend_')

plt.plot(n_train, pertail64, 
    c='firebrick',
    marker = 'x', markersize=5,
    alpha=0.7,
    lw=3.50,
    label='_nolegend_')

plt.plot(n_train, 
    pertail64tanh, 
    'firebrick',
    marker = 'x', markersize=5,
    alpha=0.7,
    lw=1.50,
    label='_nolegend_')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

#for t in l.get_texts(): t.set_va('center_baseline')

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

plt.xlabel('$N_{\\rm train} / 1000 $')
plt.yscale('log')
plt.xlim(101,499)
plt.ylim(0.0099,2.2)
#plt.legend()
plt.savefig("tanhvshxT64.pdf", format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)

plt.savefig("tanhvshxT64.svg", format="svg", bbox_inches="tight", dpi=300, pad_inches=0.05)

plt.savefig("tanhvshxT64.jpg", format="jpg", bbox_inches="tight", dpi=300, pad_inches=0.05)
