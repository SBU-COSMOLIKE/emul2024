import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux

fig, ax1 = plt.subplots()

med128=np.array([0.215, 0.038, 0.019 ,0.011,0.006])

pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])

med64=np.array([0.138,0.029,0.015,0.008,0.004])

pertail64=np.array([0.3907,0.0552,0.0399,0.0315,0.0295])




n_train=np.array([100,200,300,400,530])

ax1.plot(n_train,pertail128,'b-',label='T=128, fraction of tail')

ax1.plot(n_train,pertail64,'r-',label='T=64, fraction of tail')
ax2 = ax1.twinx()
ax2.plot(n_train,med128,'b--',label=r'T=128, median $\chi^2$')
ax2.plot(n_train,med64,'r--',label=r'T=64, median $\chi^2$')
ax1.set_xlabel('Number of training points(thousands)')
ax1.set_ylabel(r'Fraction of points with $\chi^2>0.2$')
ax2.set_ylabel(r'median $\chi^2$')
ax1.set_yscale('log')
fig.legend()
fig.savefig("fractionvsnum.pdf", format="pdf", bbox_inches="tight",dpi=150, pad_inches=0.5)