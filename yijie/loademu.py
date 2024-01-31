import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        # This function is designed for the Neuro-network to learn how to normalize the data between
        # layers. we will initiate gains and bias both at 1 
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, int_dim, N_layer):
        super(MLP, self).__init__()
        modules=[]

        # Def: we will set the internal dimension as multiple of 128 (reason: just simplicity)
        int_dim = int_dim * 128

        # Def: we will only change the dimension of the datavector using linear transformations  
        modules.append(nn.Linear(input_dim, int_dim))
        
        # Def: by design, a pure block has the input and output dimension to be the same
        for n in range(N_layer):
            # Def: This is what we defined as a pure MLP block
            # Why the Affine function?
            #   R: this is for the Neuro-network to learn how to normalize the data between layers
            modules.append(Affine())
            modules.append(nn.Linear(int_dim, int_dim))
            modules.append(nn.Tanh())
        
        # Def: the transformation from the internal dimension to the output dimension of the
        #      data vector we intend to emulate
        modules.append(nn.Linear(int_dim, output_dim))
        
        # NN.SEQUENTIAL is a PYTHORCH function DEFINED AT: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # This function stacks up layers in the modules-list in sequence to create the whole model
        self.mlp =nn.Sequential(*modules)#

    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        out = self.mlp(x)
        return out

N_thread=10
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    torch.set_num_interop_threads(N_thread) # Inter-op parallelism
    torch.set_num_threads(N_thread) # Intra-op parallelism

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(input_dim=5,output_dim=2970,int_dim=100,N_layer=10).to(device)


#load model
PATH = "./trainedemu/trial1"
extrainfo=np.load(PATH+'.npy',allow_pickle=True)
X_mean=extrainfo.item()['X_mean']
X_std=extrainfo.item()['X_std']
Y_mean=extrainfo.item()['Y_mean']
Y_std=extrainfo.item()['Y_std']
model.load_state_dict(torch.load(PATH+'.pt'))
model.eval()

#set up predict
def predict(X):
    with torch.no_grad():
        y_pred = (model(((X - X_mean) / X_std).to(torch.float32)) *Y_std.to(torch.float32)+Y_mean.to(torch.float32)).numpy()
    return y_pred

#just a small trial to see if the pipline works
X_try=np.array([0.02,0.1,69,3,1.01])
X_try=torch.tensor(X_try)
print(predict(X_try))
