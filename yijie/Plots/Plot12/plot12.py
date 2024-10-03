import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux

fig, ax1 = plt.subplots()

med128=np.array([0.215, 0.038, 0.019 ,0.011,0.006])

med128res=np.array([0.389,0.123 ,0.089,0.080,0.049 ])

pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])

pertail128res=np.array([0.6712,0.3804,0.3210,0.2889,0.1854])

med64=np.array([0.138,0.029,0.015,0.008,0.004])
med64res=np.array([0.261,0.088,0.064,0.055,0.037])

pertail64=np.array([0.3907,0.0552,0.0399,0.0315,0.0295])

pertail64res=np.array([0.5939,0.2408,0.1725,0.1495,0.0676])




n_train=np.array([100,200,300,400,530])

ax1.plot(n_train,pertail128,'b-',label='TRF, T=128, tail fraction')

ax1.plot(n_train,pertail64,'r-',label='TRF, T=64, tail fraction')

ax1.plot(n_train,pertail128res,'g-',label='ResMLP, T=128, tail fraction')

ax1.plot(n_train,pertail64res,'c-',label='ResMLP, T=64, tail fraction')
ax2 = ax1.twinx()
ax2.plot(n_train,med128,'b--',label=r'TRF, T=128, median $\chi^2$')
ax2.plot(n_train,med64,'r--',label=r'TRF, T=64, median $\chi^2$')

ax2.plot(n_train,med128res,'g--',label=r'ResMLP, T=128, median $\chi^2$')
ax2.plot(n_train,med64res,'c--',label=r'ResMLP, T=64, median $\chi^2$')
ax1.set_xlabel('Number of training points(thousands)',fontsize=18)
ax1.set_ylabel(r'Fraction of points with $\chi^2>0.2$',fontsize=18)
ax2.set_ylabel(r'median $\chi^2$',fontsize=18)
ax1.set_yscale('log')
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
fig.legend(fontsize="10",loc='center left', bbox_to_anchor=(1.05, 0.5))
fig.savefig("fractionvsnum.pdf", format="pdf", bbox_inches="tight",dpi=150, pad_inches=0.5)