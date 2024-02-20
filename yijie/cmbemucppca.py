import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.decomposition import IncrementalPCA

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
camb_ell_max          = 2509
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

#fid=np.load('YZ_samples/LHS/fid.npy',allow_pickle=True)

#covinv=np.load('YZ_samples/LHS/cosvarinv.npy',allow_pickle=True)[:camb_ell_range]
#covinv=torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix

#load in data
samples=np.load('YZ_samples/LHS/coslhc_acc.npy',allow_pickle=True)[:5000]# This is actually a latin hypercube sampling of 1mil points
input_size=len(samples[0])
data_vectors=np.load('YZ_samples/LHS/coslhc_acc_output.npy',allow_pickle=True)[:5000,:camb_ell_range,0]
data_vectors=np.log(data_vectors)
n_pcas=512#testing cosmopower parameter choice
out_size=n_pcas#1*len(data_vectors[0])
#assign training and validation sets
model = MLP(input_dim=input_size,output_dim=out_size,int_dim=4,N_layer=4)
optimizer = torch.optim.Adam(model.parameters())

reduce_lr = True#reducing learning rate on plateau
if reduce_lr==True:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10)
model = nn.DataParallel(model)
model.to(device)

vnum=0
train_samples=[]
train_data_vectors=[]
validation_samples=[]
validation_data_vectors=[]
for ind in range(len(samples)):
    samp=samples[ind]
    if (0.01<samp[0]<0.035) and (0.005<samp[1]<0.85) and (30<samp[2]<90) and (0.02<samp[3]<0.75) and (0.8<samp[4]<1.2) and (1.7<samp[5]<4.5) and vnum<=int(1e2):
        validation_samples.append(samp)
        validation_data_vectors.append(data_vectors[ind])
        vnum+=1
    else:
        train_samples.append(samp)
        train_data_vectors.append(data_vectors[ind])


train_samples=np.array(train_samples)
train_data_vectors=np.array(train_data_vectors)
validation_samples=np.array(validation_samples)
validation_data_vectors=np.array(validation_data_vectors)

train_samples=torch.Tensor(train_samples)
validation_samples=torch.Tensor(validation_samples)

X_mean=torch.Tensor(train_samples.mean(axis=0, keepdims=True))
X_std  = torch.Tensor(train_samples.std(axis=0, keepdims=True))
Y_mean=train_data_vectors.mean(axis=0, keepdims=True)
Y_std=train_data_vectors.std(axis=0, keepdims=True)
X_train=(train_samples-X_mean)/X_std
X_train[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization
y_train=(train_data_vectors-Y_mean)/Y_std

X_validation=(validation_samples-X_mean)/X_std
X_validation[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization
y_validation=(validation_data_vectors-Y_mean)/Y_std
Y_std=torch.tensor(Y_std).to(device)


TTPCA = IncrementalPCA(n_components=n_pcas,batch_size=len(y_train))
#TEPCA = IncrementalPCA(n_components=n_pcas,batch_size=len(train_samples))
#EEPCA = IncrementalPCA(n_components=n_pcas,batch_size=len(train_samples))

TTPCA.partial_fit(y_train)
#TEPCA.partial_fit(train_data_vectors[:,:,1])
#EEPCA.partial_fit(train_data_vectors[:,:,2])
transform_matrix=torch.Tensor(TTPCA.components_).to(device)
#train_pca=np.zeros((len(train_samples), n_pcas, 3), dtype="float32")

train_pca=TTPCA.transform(y_train)
train_pca=torch.Tensor(train_pca)

#train_pca[:,:,1]=TTPCA.transform(train_data_vectors[:,:,1])
#train_pca[:,:,2]=TTPCA.transform(train_data_vectors[:,:,2])

#vali_pca=np.zeros((len(validation_samples), n_pcas, 3), dtype="float32")

vali_pca=TTPCA.transform(y_validation)
vali_pca=torch.Tensor(vali_pca)

#normalizing samples and data vectors to mean 0, std 1

#load the data to batches. Do not send those to device yet to save space


batch_size=512
trainset    = TensorDataset(X_train, train_pca)
validset    = TensorDataset(X_validation,vali_pca)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

#Set up the model and optimizer

#training
n_epoch=400#for trial test purpose
losses_train = []
losses_vali = []

for n in range(n_epoch):
    
    
    losses=[]
    for i, data in enumerate(trainloader):
        model.train()
        X = data[0].to(device)# send to device one by one
        Y_batch = data[1].to(device)# send to device one by one
        Y_pred  = model(X).to(device)
        diff_b=(Y_batch-Y_pred).cpu().detach().numpy()
        #Y_pred_unpca=torch.Tensor(TTPCA.inverse_transform(diff_b)).to(device)#print((Y_pred), 'pred')

        #Y_pred=torch.reshape(Y_pred, (batch_size,n_pcas,3))
        
        diff =TTPCA.inverse_transform(diff_b)# Scale back to unit by *Y_std
        diff=torch.Tensor(diff).to(device)
        diff=diff*Y_std
        diff.requires_grad_(True)
        loss1 = torch.sqrt(torch.einsum('kl,kl->k',diff,diff))# implement with torch.einsum
        #print(loss1)
        loss=torch.mean(loss1)

        optimizer.zero_grad()
        loss.backward()
        #print(loss)
        losses.append(loss.cpu().detach().numpy())
        
        
        optimizer.step()

    losses_train.append(np.mean(losses))# We take means since a loss function should return a single real number

    with torch.no_grad():
        model.eval()
        
        losses = []
        for i, data in enumerate(validloader):
            X_v       = data[0].to(device)
            Y_v_batch = data[1].to(device)
            Y_v_pred = model(X_v).to(device)
            diff_v_b=(Y_v_batch-Y_v_pred).cpu().detach().numpy()
            v_diff =TTPCA.inverse_transform(diff_v_b)# Scale back to unit by *Y_std
            v_diff=torch.Tensor(v_diff).to(device)
            v_diff=v_diff*Y_std
            v_diff.requires_grad_(True)

            #v_diff_b=np.zeros((batch_size, camb_ell_range, 3), dtype="float32")
            #v_diff_b=np.dot(v_diff, transform_matrix)
            #v_diff_b[:,:,1]=np.dot(v_diff[:,:,1], TEPCA.components_)
            #v_diff_b[:,:,2]=np.dot(v_diff[:,:,2], EEPCA.components_)
            
            loss1 = torch.sqrt(torch.einsum('kl,kl->k',v_diff,v_diff))

            #print(loss)
            loss_vali=torch.mean(loss1)
            losses.append(loss_vali.cpu().detach().numpy())

        losses_vali.append(np.mean(losses))
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
PATH = "./trainedemucp/pca"+str(batch_size)
torch.save(model.state_dict(), PATH+'.pt')
extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std}
np.save(PATH+'.npy',extrainfo)
np.save(PATH+'losstrain.npy',losses_train)
np.save(PATH+'lossvali.npy',losses_vali)
np.save(PATH+'TTpca.npy',TTPCA.components_)
#np.save(PATH+'TEpca.npy',TEPCA.components_)
#np.save(PATH+'EEpca.npy',EEPCA.components_)
