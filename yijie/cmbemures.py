import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset

'''
N_thread=10

if torch.cuda.is_available():

    device = 'cuda'# using gpu if gpu is available

else:

    device = 'cpu'

    torch.set_num_interop_threads(N_thread) # Inter-op parallelism

    torch.set_num_threads(N_thread) # Intra-op parallelism

#define model
print(device)
'''


camb_ell_min          = 2#30
camb_ell_max          = 5000
camb_ell_range        = camb_ell_max  - camb_ell_min 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            #   R: this is for the Neuro-network to learn how to normalize the data between layers
            modules.append(Affine())
            modules.append(ResBlock(int_dim, int_dim))
            #modules.append(nn.Tanh())
        
        # Def: the transformation from the internal dimension to the output dimension of the
        #      data vector we intend to emulate
        modules.append(nn.Linear(int_dim, output_dim))
        
        # NN.SEQUENTIAL is a PYTHORCH function DEFINED AT: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # This function stacks up layers in the modules-list in sequence to create the whole model
        self.resmlp =nn.Sequential(*modules)#

    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        out = self.resmlp(x)

        return out

#Set up the Covariance matrix

fid=np.load('YZ_samples/LHS/fid.npy',allow_pickle=True)

covinv=np.load('YZ_samples/LHS/cosvarinvTT.npy',allow_pickle=True)[:camb_ell_range,:camb_ell_range]
covinv=torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix

#load in data
train_samples=np.load('YZ_samples/LHS/coslhc_acc.npy',allow_pickle=True)# This is actually a latin hypercube sampling of 1mil points
vali_samples=np.load('YZ_samples/Uniform/input/cosuni_0.npy',allow_pickle=True)
input_size=len(train_samples[0])
train_data_vectors=np.load('YZ_samples/LHS/coslhc_acc_output.npy',allow_pickle=True)[:,:camb_ell_range,0]

vali_data_vectors=np.load('YZ_samples/Uniform/output/cosuni_0_output.npy',allow_pickle=True)[:,:camb_ell_range,0]
for i in range(1,10):
    samp_new=np.load('YZ_samples/Uniform/input/cosuni_'+str(i)+'.npy',allow_pickle=True)
    dv_new=np.load('YZ_samples/Uniform/output/cosuni_'+str(i)+'_output.npy',allow_pickle=True)[:,:camb_ell_range,0]
    vali_samples=np.vstack((vali_samples,samp_new))
    vali_data_vectors=np.vstack((vali_data_vectors,dv_new))

out_size=1*len(train_data_vectors[0])
#assign training and validation sets
model = ResMLP(input_dim=input_size,output_dim=out_size,int_dim=4,N_layer=4)
optimizer = torch.optim.Adam(model.parameters())

reduce_lr = True#reducing learning rate on plateau
if reduce_lr==True:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10)
model = nn.DataParallel(model)
model.to(device)


validation_samples=[]
validation_data_vectors=[]
for ind in range(len(vali_samples)):
    samp=vali_samples[ind]
    dv=vali_data_vectors[ind]
    if (0.01<samp[0]<0.035) and (0.005<samp[1]<0.85) and (30<samp[2]<90) and (0.02<samp[3]<0.75) and (0.8<samp[4]<1.2) and (1.7<samp[5]<4.5) and (dv[0]!=1) and (dv[1]!=1):
        validation_samples.append(samp)
        validation_data_vectors.append(dv)
        
    else:
        placeholder=1


validation_samples=np.array(validation_samples)
validation_data_vectors=np.array(validation_data_vectors)

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
Y_std=Y_std.to(device)
#load the data to batches. Do not send those to device yet to save space

batch_size=256
trainset    = TensorDataset(X_train, y_train)
validset    = TensorDataset(X_validation,y_validation)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

#Set up the model and optimizer

#training
n_epoch=500#for trial test purpose
losses_train = []
losses_vali = []
losses_train_med = []
losses_vali_med = []

for n in range(n_epoch):
    
    
    losses=[]
    for i, data in enumerate(trainloader):
        model.train()
        X = data[0].to(device)# send to device one by one
        Y_batch = data[1].to(device)# send to device one by one
        Y_pred  = model(X).to(device)
        #print((Y_pred), 'pred')

        #Y_pred=torch.reshape(Y_pred, (batch_size,camb_ell_range, 3))
        diff = (Y_batch - Y_pred)*Y_std# Scale back to unit by *Y_std
        

        
        loss1 = torch.diag(diff @ covinv @ torch.t(diff))# implement with torch.einsum
        loss1,index=loss1.sort()#[:int(-0.02*batch_size)]
        #print(loss1)
        loss1=loss1[:int(-0.04*batch_size)]
        extra=torch.sqrt(torch.diag(diff[index[int(-0.04*batch_size):].cpu().numpy()]@ torch.t(diff[index[int(-0.04*batch_size):].cpu().numpy()])))
        loss1=torch.cat((loss1,extra))
        loss=torch.mean(loss1)
        #print(loss)
        losses.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses_train.append(np.mean(losses))# We take means since a loss function should return a single real number
    losses_train_med.append(np.median(losses))

    with torch.no_grad():
        model.eval()
        
        losses = []
        for i, data in enumerate(validloader):
            X_v       = data[0].to(device)
            Y_v_batch = data[1].to(device)
            Y_v_pred = model(X_v).to(device)
            #Y_v_pred=torch.reshape(Y_v_pred, (batch_size,camb_ell_range, 3))
            v_diff = (Y_v_batch - Y_v_pred )*Y_std
            
            loss1 = torch.diag(v_diff @ covinv @ torch.t(v_diff))# implement with torch.einsum
            loss1,index=loss1.sort()#[:int(-0.02*batch_size)]
            #print(loss1)
            loss1=loss1[:int(-0.04*batch_size)]
            extra=torch.sqrt(torch.diag(v_diff[index[int(-0.04*batch_size):].cpu().numpy()]@ torch.t(v_diff[index[int(-0.04*batch_size):].cpu().numpy()])))
            loss1=torch.cat((loss1,extra))
            loss_vali=torch.mean(loss1)
            losses.append(loss_vali.cpu().detach().numpy())

        losses_vali.append(np.mean(losses))
        losses_vali_med.append(np.median(losses))
        if reduce_lr == True:
            print('Reduce LR on plateu: ',reduce_lr)
            scheduler.step(losses_vali[n])

    #if optimizer.param_groups[0]['lr']<1e-9:



    print('epoch {}, loss={}, validation loss={}, lr={} )'.format(
                        n,
                        losses_train[-1],
                        losses_vali[-1],
                        optimizer.param_groups[0]['lr']
                        
                    ))#, total runtime: {} ({} average))



# Save the model and extra parameters
PATH = "./trainedemu5000/chiTTres"
torch.save(model.state_dict(), PATH+'.pt')
extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std}
np.save(PATH+'.npy',extrainfo)
np.save(PATH+'losstrain.npy',losses_train)
np.save(PATH+'lossvali.npy',losses_vali)
np.save(PATH+'losstrainmed.npy',losses_train_med)
np.save(PATH+'lossvalimed.npy',losses_vali_med)
