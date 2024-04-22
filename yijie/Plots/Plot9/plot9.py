import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TKAgg')
device = torch.device('cpu') # To do the eval, we need the model and data all on the same device, so either all on cpu or all on gpu
Nlayer=4
intdim=4

camb_ell_min          = 2#30
camb_ell_max          = 5000
camb_ell_range        = camb_ell_max  - camb_ell_min 

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        # This function is designed for the Neuro-network to learn how to normalize the data between
        # layers. we will initiate gains and bias both at 1 
        self.gain = nn.Parameter(torch.ones(1))

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        return x * self.gain + self.bias


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        self.act1 = nn.Tanh()#nn.ReLU()#
        self.act2 = nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2


class ResMLP(nn.Module):

    def __init__(self, input_dim, output_dim, int_dim, N_layer):

        super(ResMLP, self).__init__()

        modules=[]

        # Def: we will set the internal dimension as multiple of 128 (reason: just simplicity)
        int_dim = int_dim * 128

        # Def: we will only change the dimension of the datavector using linear transformations  
        modules.append(nn.Linear(input_dim, int_dim))
        
        # Def: by design, a pure block has the input and output dimension to be the same
        for n in range(N_layer):
            # Def: This is what we defined as a pure MLP block
            # Why the Affine function?
            #   R: this is for the Neuro-network to learn how to normalize the data between layer
            modules.append(ResBlock(int_dim, int_dim))
            #modules.append(nn.Tanh())
        
        # Def: the transformation from the internal dimension to the output dimension of the
        #      data vector we intend to emulate
        modules.append(nn.Tanh())
        modules.append(nn.Linear(int_dim, output_dim))
        modules.append(Affine())
        # NN.SEQUENTIAL is a PYTHORCH function DEFINED AT: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # This function stacks up layers in the modules-list in sequence to create the whole model
        self.resmlp =nn.Sequential(*modules)#

    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        out = self.resmlp(x)

        return out
model = ResMLP(input_dim=9,output_dim=camb_ell_range,int_dim=intdim,N_layer=Nlayer)
model.to(device)
model = nn.DataParallel(model)


covinv         = np.load('cosvarinvTT.npy',allow_pickle=True)[:camb_ell_range,:camb_ell_range]
testing_sample = np.load('testsample.npy',allow_pickle=True)
testing_data_vector = np.load('testdv.npy',allow_pickle=True)


PATH = "chiTTAstauresneti4l4"
extrainfo=np.load(PATH+'.npy',allow_pickle=True)
X_mean=extrainfo.item()['X_mean'].to(device)
X_std=extrainfo.item()['X_std'].to(device)
Y_mean=extrainfo.item()['Y_mean'].to(device)
Y_std=extrainfo.item()['Y_std'].to(device)
model.load_state_dict(torch.load(PATH+'.pt',map_location=device))
model.eval()

def predict(X):
    with torch.no_grad():
        X_norm=((X.float() - X_mean.float()) / X_std.float())
        X_norm[:,6:]=0
        #print(X_norm)
        pred=model(X_norm)
        
        M_pred=pred.to(device)
        y_pred = (M_pred.float() *Y_std.float()+Y_mean.float()).numpy()
        for i in range(len(y_pred)):
            y_pred[i]=y_pred[i]*(np.exp(X[5].float().numpy()))/(np.exp(2*X[3].float().numpy()))
    return y_pred

#just a small trial to see if the pipline works
testing_sample = torch.Tensor(testing_sample)

testing_result = predict(testing_sample)[0]

diff = testing_data_vector - testing_result

ell = np.arange(2,5000,1)


plt.plot(ell,testing_result*ell*(ell+1),'.',label='emulator')
plt.plot(ell,testing_data_vector*ell*(ell+1),label='camb')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell}^{TT}$')
plt.legend()
plt.savefig("resnetwd.pdf", format="pdf", bbox_inches="tight")

plt.clf()


plt.plot(ell,(testing_result-testing_data_vector)*np.sqrt(np.diag(covinv)))
#plt.xscale('log')

#plt.ylim(-0.003,0.003)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\Delta C_{\ell}^{TT}/\sigma_{\ell}^{TT}$')
plt.ylim(-0.05,0.125)
plt.axhline(y = 0, color = 'grey', linestyle = '-') 
#plt.legend()
plt.savefig("resnetwderror.pdf", format="pdf", bbox_inches="tight")

plt.clf()

chi2=((testing_result-testing_data_vector)*np.sqrt(np.diag(covinv)))**2
chil=[]
ell=np.arange(2,5000,1)
for i in range(len(chi2)):
    chil.append(np.sum(chi2[:i]))
chil=np.array(chil)
plt.plot(ell,chil)
#plt.xscale('log')

#plt.ylim(-0.003,0.003)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\chi^2(\ell)$')
#plt.axhline(y = 0, color = 'grey', linestyle = '-') 
plt.savefig("resnetwdchi.pdf", format="pdf", bbox_inches="tight")