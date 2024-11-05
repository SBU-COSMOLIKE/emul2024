import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
import getdist
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt

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

d1=np.load('cos_pkc_T256_cut.npy',allow_pickle=True)[:,:6]
d2=np.load('cos_pkc_T128_cut.npy',allow_pickle=True)[:,:6]
d3=np.load('cos_pkc_T64_cut.npy',allow_pickle=True)[:,:6]

labels=[r'\Omega_bh^2',r'\Omega_ch^2',r'H_0',r'\tau',r'n_s',r'\log{(10^{10}A_s)}']
names=['omb','omc','H0','tau','ns','logAs']

samples1 = MCSamples(samples=d1,names = names, labels = labels,label='T=256'
                     ,ranges={'H0':(25, 100),'tau':(0.007, 0.16)
                              ,'ns':(0.65, 1.35),'logAs':(1.45,4.65)},settings={'smooth_scale_2D':1.5})
samples2 = MCSamples(samples=d2,names = names, labels = labels,label='T=128'
                     ,ranges={'H0':(d2[:,2].min(), d2[:,2].max()),'tau':(d2[:,3].min(), d2[:,3].max())
                              ,'ns':(d2[:,4].min(), d2[:,4].max()),'logAs':(d2[:,5].min(), d2[:,5].max())},settings={'smooth_scale_2D':0.8})
samples3 = MCSamples(samples=d3,names = names, labels = labels,label='T=64'
                     ,ranges={'H0':(d3[:,2].min(), d3[:,2].max()),'tau':(d3[:,3].min(), d3[:,3].max())
                              ,'ns':(d3[:,4].min(), d3[:,4].max()),'logAs':(d3[:,5].min(), d3[:,5].max())},settings={'smooth_scale_2D':0.8})

g = plots.get_single_plotter(width_inch=5,ratio=1)
g.plot_2d([samples1,samples2,samples3],['omb','omc'],filled=True)
g.add_legend(['T=256', 'T=128','T=64'], legend_loc='lower right',fontsize=13)
g.fig.savefig('OMBOMC.pdf',format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)

g = plots.get_single_plotter(width_inch=5,ratio=1)
g.plot_2d([samples1,samples2,samples3],['H0','ns'],filled=True)
g.add_legend(['T=256', 'T=128','T=64'], legend_loc='upper left',fontsize=13)
g.fig.savefig('H0NS.pdf',format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)

g = plots.get_single_plotter(width_inch=5,ratio=1)
g.plot_2d([samples1,samples2,samples3],['H0','logAs'],filled=True)
g.add_legend(['T=256', 'T=128','T=64'], legend_loc='upper left',fontsize=13)
g.fig.savefig('H0AS.pdf',format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)