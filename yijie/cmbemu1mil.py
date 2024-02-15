import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

N_thread=10

if torch.cuda.is_available():

    device = 'cuda'

else:

    device = 'cpu'

    torch.set_num_interop_threads(N_thread) # Inter-op parallelism

    torch.set_num_threads(N_thread) # Intra-op parallelism
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#define model
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


#Set up the Covariance matrix
fid=np.load('YZ_samples/LHS/fid.npy',allow_pickle=True)

covinv=np.load('YZ_samples/LHS/cosvarinv.npy',allow_pickle=True)
covinv=torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix

#load in data
samples=np.load('YZ_samples/LHS/coslhc_acc.npy',allow_pickle=True)# This is actually a latin hypercube sampling of 1mil points
input_size=len(samples[0])
data_vectors=np.load('YZ_samples/LHS/coslhc_acc_output.npy',allow_pickle=True)

out_size=3*len(data_vectors[0])
#assign training and validation sets
vnum=0
train_samples=[]
train_data_vectors=[]
validation_samples=[]
validation_data_vectors=[]
for ind in range(len(samples)):
    samp=samples[ind]
    if (0.01<samp[0]<0.035) and (0.005<samp[1]<0.85) and (30<samp[2]<90) and (0.02<samp[3]<0.75) and (0.8<samp[4]<1.2) and (1.7<samp[5]<4.5) and vnum<=int(1e5):
        validation_samples.append(samp)
        validation_data_vectors.append(data_vectors[ind])
        vnum+=1
    else:
        train_samples.append(samp)
        train_data_vectors.append(data_vectors[ind])


train_samples=torch.Tensor(train_samples)
train_data_vectors=torch.Tensor(train_data_vectors)
validation_samples=torch.Tensor(validation_samples)
validation_data_vectors=torch.Tensor(validation_data_vectors)

#normalizing samples and data vectors to mean 0, std 1
X_mean=torch.Tensor(train_samples.mean(axis=0, keepdims=True))
X_std  = torch.Tensor(train_samples.std(axis=0, keepdims=True))
Y_mean=torch.Tensor(train_data_vectors.mean(axis=0, keepdims=True))
Y_std=torch.Tensor(train_data_vectors.std(axis=0, keepdims=True))
X_train=(train_samples-X_mean)/X_std
X_train[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization
y_train=(train_data_vectors-Y_mean)/Y_std

X_validation=(validation_samples-X_mean)/X_std
X_validation[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization
y_validation=(validation_data_vectors-Y_mean)/Y_std

#load the data to batches. Do not send those to device yet to save space
batch_size=256
trainset    = TensorDataset(X_train, y_train)
validset    = TensorDataset(X_validation,y_validation)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

#Set up the model and optimizer
model = MLP(input_dim=input_size,output_dim=out_size,int_dim=128,N_layer=3).to(device)

optimizer = torch.optim.Adam(model.parameters())

reduce_lr = True#reducing learning rate on plateau
if reduce_lr==True:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10)

#training
n_epoch=1000#for trial test purpose
losses_train = []
losses_vali = []

for n in range(n_epoch):
    
    
    losses=[]
    for i, data in enumerate(trainloader):
        model.train()
        X = data[0].to(device)# send to device one by one
        Y_batch = data[1].to(device)# send to device one by one
        Y_pred  = model(X)
        #print((Y_pred), 'pred')

        Y_pred=torch.reshape(Y_pred, (256,4998, 3))
        diff = (Y_batch - Y_pred)*Y_std# Scale back to unit by *Y_std
        

        
        loss1 = torch.einsum('kli,lij,klj->k',diff,covinv,diff)# wil implement with torch.einsum
        
        loss=torch.mean(loss1)# torch.diagonal will only give you the diagonal part of the matrix
        losses.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses_train.append(np.mean(losses))# We take means since a loss function should return a single real number

    with torch.no_grad():
        model.eval()
        
        losses = []
        for i, data in enumerate(validloader):
            X_v       = data[0].to(device)
            Y_v_batch = data[1].to(device)
            Y_v_pred = model(X_v) 
            Y_v_pred=torch.reshape(Y_v_pred, (256,4998, 3))
            v_diff = (Y_v_batch - Y_v_pred )*Y_std
            
            loss1 = torch.einsum('kli,lij,klj->k',v_diff,covinv,v_diff)

            #print(loss)
            loss_vali=torch.mean(loss1)
            losses.append(loss_vali.cpu().detach().numpy())

        losses_vali.append(np.mean(losses))
        if reduce_lr == True:
            print('Reduce LR on plateu: ',reduce_lr)
            scheduler.step(losses_vali[n])


    print('epoch {}, loss={}, validation loss={}, lr={} )'.format(
                        n,
                        losses_train[-1],
                        losses_vali[-1],
                        optimizer.param_groups[0]['lr']
                        
                    ))#, total runtime: {} ({} average))



# Save the model and extra parameters
PATH = "./trainedemu/trial1milaccb256"
torch.save(model.state_dict(), PATH+'.pt')
extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std}
np.save(PATH+'.npy',extrainfo)
np.save(PATH+'losstrain.npy',losses_train)
np.save(PATH+'lossvali.npy',losses_vali)
