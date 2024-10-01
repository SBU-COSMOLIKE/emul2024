import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux


pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])

pertail32=np.array([0.2617,0.0217,0.0176,0.0166 ,0.0161])

pertail8=np.array([0.1462,0.0062,0.0043,0.0045,0.0045])

n_train=np.array([100,200,300,400,530])

plt.plot(n_train,pertail8,label='T=8')

plt.plot(n_train,pertail32,label='T=32')

plt.plot(n_train,pertail128,label='T=128')

plt.xlabel('Number of training points(thousands)')
plt.ylabel(r'Fraction of points with $\chi^2>0.2$')

plt.yscale('log')
plt.legend()
plt.savefig("fractionvsnum.pdf", format="pdf", bbox_inches="tight",dpi=150, pad_inches=0.5)
