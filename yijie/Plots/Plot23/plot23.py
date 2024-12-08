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

d1=np.load('cos_uniform_cut_0.npy',allow_pickle=True)[:,:6]
d2=np.load('cos_uniform_cut_testset.npy',allow_pickle=True)[:,:6]


labels=[r'\Omega_bh^2',r'\Omega_ch^2',r'H_0',r'\tau',r'n_s',r'\log{(10^{10}A_s)}']
names=['omb','omc','H0','tau','ns','logAs']

samples1 = MCSamples(samples=d1,names = names, labels = labels,label='Uniform Train'
                     ,ranges={'H0':(20, 120),'tau':(0.005, 0.16)
                              ,'ns':(0.65, 1.35),'logAs':(1.45,4.65)},settings={'smooth_scale_2D':0.5})
samples2 = MCSamples(samples=d2,names = names, labels = labels,label='$Uniform Test'
                     ,settings={'smooth_scale_2D':0.5})
#samples3 = MCSamples(samples=d3,names = names, labels = labels,label='T=64'
 #                    ,ranges={'H0':(d3[:,2].min(), d3[:,2].max()),'tau':(d3[:,3].min(), d3[:,3].max())
  #                            ,'ns':(d3[:,4].min(), d3[:,4].max()),'logAs':(d3[:,5].min(), d3[:,5].max())},settings={'smooth_scale_2D':0.8})


g = plots.get_single_plotter(width_inch = 4.75, ratio=1.0483870)
g.settings.alpha_filled_add = 0.7
g.settings.lw_contour = 1.3
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.subplot_size_ratio = 1
g.settings.legend_frame = False
g.settings.legend_rect_border = False
g.settings.legend_fontsize = 13.5
g.settings.axes_fontsize  = 18
g.settings.axes_labelsize = 20
g.plot_2d([samples1,samples2],['omb','omc'],filled=[False,True],colors=['dimgray','firebrick'],contour_lws=[2,1.2])
g.add_legend(['Uniform Train', 'Uniform Test'], legend_loc='lower right',fontsize=15)
g.fig.savefig('OMBOMCuni.pdf',format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)

g = plots.get_single_plotter(width_inch = 4.75, ratio=1.0483870)
g.settings.alpha_filled_add = 0.7
g.settings.lw_contour = 1.3
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.subplot_size_ratio = 1
g.settings.legend_frame = False
g.settings.legend_rect_border = False
g.settings.legend_fontsize = 13.5
g.settings.axes_fontsize  = 18
g.settings.axes_labelsize = 20
g.plot_2d([samples1,samples2],['H0','ns'],filled=[False,True],colors=['dimgray','firebrick'],contour_lws=[2,1.2])
#g.add_legend(['T=256', 'T=128','T=64'], legend_loc='upper left',fontsize=13)
g.fig.savefig('H0NSuni.pdf',format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)

g = plots.get_single_plotter(width_inch = 4.75, ratio=1.0483870)
g.settings.alpha_filled_add = 0.7
g.settings.lw_contour = 1.3
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.subplot_size_ratio = 1
g.settings.legend_frame = False
g.settings.legend_rect_border = False
g.settings.legend_fontsize = 13.5
g.settings.axes_fontsize  = 18
g.settings.axes_labelsize = 20
g.plot_2d([samples1,samples2],['H0','logAs'],filled=[False,True],colors=['dimgray','firebrick'],contour_lws=[2,1.2])
#g.add_legend(['T=256', 'T=128','T=64'], legend_loc='upper left',fontsize=13)
g.fig.savefig('H0ASuni.pdf',format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)