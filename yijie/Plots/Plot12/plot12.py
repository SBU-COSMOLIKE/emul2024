import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux


pertail128=np.array([0.5169,0.1696,0.1055,0.0807 ,0.0607])
pertail1283b=np.array([0.5168 , 0.1645, 0.0782,0.0680, 0.0567])

pertail32=np.array([0.2617,0.0217,0.0176,0.0166 ,0.0161])
pertail323b=np.array([0.2974,0.0196,0.0170,0.0164,0.0162])

pertail8=np.array([0.1462,0.0062,0.0043,0.0045,0.0045])
pertail83b=np.array([0.1949,0.0053,0.0049,0.0048,0.0044])

n_train=np.array([100,200,300,400,530])

plt.plot(n_train,pertail8,'b-',label='T=8, 1 Block')

plt.plot(n_train,pertail32,'r-',label='T=32, 1 Block')

plt.plot(n_train,pertail128,'g-',label='T=128, 1 Block')

plt.plot(n_train,pertail83b,'b--',label='T=8, 3 Block')

plt.plot(n_train,pertail323b,'r--',label='T=32, 3 Block')

plt.plot(n_train,pertail1283b,'g--',label='T=128, 3 Block')

plt.xlabel('Number of training points(thousands)')
plt.ylabel(r'Fraction of points with $\chi^2>0.2$')

plt.yscale('log')
plt.legend()
plt.savefig("fractionvsnum.pdf", format="pdf", bbox_inches="tight",dpi=150, pad_inches=0.5)
