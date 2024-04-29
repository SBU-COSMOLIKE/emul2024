import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
import scipy.linalg
import rescaling
matplotlib.use('TKAgg')# This is for windows, you may not need this for Mac/Linux


camb_ell_min          = 2#30
camb_ell_max          = 5000
camb_ell_range        = camb_ell_max  - camb_ell_min 

test_samples      = np.load('500sample.npy',allow_pickle=True)#[:100]

test_data_vectors = np.load('500dv.npy',allow_pickle=True)[:,:camb_ell_range] # This is a thousand samples



testing_samples      = []
testing_data_vectors = []

# You can change the box size here to get different span ranges
# For each sample, sample[0]=omega_bh^2,sample[1]=omega_ch^2,sample[2]=H0,sample[3]=tau, sample[4]=n_s, 
# sample[5]=log(A_s*10^10)
for ind in range(len(test_samples)):
    samp = test_samples[ind]
    dv   = test_data_vectors[ind]
    if (0.015<samp[0]<0.03) and (0.005<samp[1]<0.25) and (60<samp[2]<80) and (0.015<samp[3]<0.3):
    # This is just an example of how I change the box size 
        testing_samples.append(samp)
        testing_data_vectors.append(dv)

testing_samples      = np.array(testing_samples)
testing_data_vectors = np.array(testing_data_vectors)

ell = np.arange(camb_ell_min,camb_ell_max,1)
for i in range(len(testing_samples)):
    plt.plot(testing_data_vectors[i]*ell*(ell+1))
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$C_{\ell}$')
plt.xlabel(r'$\ell$')
plt.ylim(1e-1,1e6)
#plt.legend()
#plt.ylim(1e-2,7e5) Please change the ylim according to your need
#plt.title('original')
plt.savefig("original.pdf", format="pdf", bbox_inches="tight",dpi=150, pad_inches=0.5)
plt.clf()


for i in range(len(testing_samples)):
    plt.plot(testing_data_vectors[i]*ell*(ell+1)/(np.exp(testing_samples[i,5]))*np.exp(2*testing_samples[i,3]))
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-1,1e6)
#plt.ylim(1e-1,2e5) Please change the ylim according to your need
#plt.title('divide by A_s, multiply by exp(2tau)')
plt.ylabel(r'$C_{\ell}^{rescale}$')
plt.xlabel(r'$\ell$')
plt.savefig("rescaledAstau.pdf", format="pdf", bbox_inches="tight",dpi=150, pad_inches=0.5)
plt.clf()


rescale = rescaling.par_tot_to_rescale(testing_samples,camb_ell_min,camb_ell_max)

scaled = (testing_data_vectors*ell*(ell+1)/rescale)
for i in range(len(testing_samples)):
    plt.plot(scaled[i])
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-1,1e6)
plt.ylabel(r'$C_{\ell}^{rescale}$')
plt.xlabel(r'$\ell$')
#plt.legend()
#plt.ylim(1e-1,2e5) Please change the ylim according to your need
#plt.title('divide by A_s, multiply by exp(2tau), take out damping from reionization and damping tail, taking out potential envelope')
plt.savefig("rescaled.pdf", format="pdf", bbox_inches="tight",dpi=150, pad_inches=0.5)
