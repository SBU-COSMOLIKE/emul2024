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

        o1 = self.act1(self.layer1(self.norm1(x)))
        o2 = self.act2(self.layer2(self.norm2(o1))) + xskip

        return o2

class CNNBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(CNNBlock, self).__init__()
        
        self.cnn=nn.Conv1d(in_channels=16, out_channels=8, kernel_size=5)
        self.pool=nn.MaxPool1d(kernel_size=5, stride=4)

    def forward(self, x):

        o1 = self.cnn(x)
        o2 = self.pool(o1)

        return o2


class CNNMLP(nn.Module):

    def __init__(self, input_dim, output_dim, int_dim, N_layer):

        super(CNNMLP, self).__init__()

        modules=[]
        self.outdim=output_dim
        # Def: we will set the internal dimension as multiple of 128 (reason: just simplicity)
        int_dim = int_dim * 128
        self.norm = Affine()
        # Def: we will only change the dimension of the datavector using linear transformations  
        self.input_layer = nn.Linear(input_dim, int_dim)
        
        # Def: by design, a pure block has the input and output dimension to be the same
        self.Res1 = ResBlock(int_dim, int_dim)

        
        # Def: the transformation from the internal dimension to the output dimension of the
        #      data vector we intend to emulate
        self.Act = nn.Tanh()
        self.CNN = CNNBlock(int_dim,int_dim)

        # NN.SEQUENTIAL is a PYTHORCH function DEFINED AT: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # This function stacks up layers in the modules-list in sequence to create the whole model
    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        x = self.input_layer(x)
        x = self.Res1(x)
        x = self.Act(x)
        x = self.Res1(x)
        x = self.Act(x)
        l = len(x)
        w = len(x[0])
        n = 64
        c = 16
        self.trans1 = nn.Linear(in_features = w,out_features = n*c).to(device)
        x = self.trans1(x).view (-1, c, n)
        x = self.CNN(x)
        n_new = len(x[0,0])
        c_new = len(x[0])
        x = x.view(l,n_new*c_new)#.to(device)
        self.Res2 = ResBlock(n_new*c_new, n_new*c_new).to(device)
        x = self.Res2(x)
        x = self.Act(x)
        #x = self.Res2(x)
        #x = self.Act(x)
        self.out_layer = nn.Linear(n_new*c_new, self.outdim).to(device)
        out = self.out_layer(x)
        out = self.norm(out)
        #print(out.size())
        return out


#Set up the Covariance matrix



covinv=np.load('extra/cosvarinvTT.npy',allow_pickle=True)[:camb_ell_range,:camb_ell_range]*4/np.exp(4*0.06)
covinv=torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix

#load in data
train_samples=np.load('/home/grads/extra_data/yijie/YZ_samples/cos3milsamps.npy',allow_pickle=True)#.astype('float32')# This is actually a latin hypercube sampling of 1mil points
validation_samples=np.load('/home/grads/data/yijie/mltrial/parametersamples/cosuni_acc_10.npy',allow_pickle=True)#.astype('float32')
lhc_len=len(train_samples)
input_size=len(train_samples[0])




validation_data_vectors=np.load('/home/grads/extra_data/yijie/YZ_samples/Uniform/output/cosuni_10_output_acc_vali.npy',allow_pickle=True)[:,:camb_ell_range]

#train_data_vectors=torch.zeros((lhc_len,camb_ell_range),dtype=torch.float32)
y_train=torch.from_numpy(np.load('/home/grads/yijie/cos3mildvs.npy',allow_pickle=True,mmap_mode='r+')[:,:camb_ell_range])


for i in range(len(validation_data_vectors)):
    validation_data_vectors[i]=validation_data_vectors[i]/(np.exp(validation_samples[i,5]))*(np.exp(2*validation_samples[i,3]))

out_size=4998#1*len(train_data_vectors[0])
#assign training and validation sets

train_samples=torch.from_numpy(train_samples)#.to(device)
#train_data_vectors=torch.from_numpy(train_data_vectors)#.to(device)
validation_samples=torch.from_numpy(validation_samples)#.to(device)
validation_data_vectors=torch.from_numpy(validation_data_vectors)#.to(device)

extrainfo=np.load("/home/grads/data/yijie/mltrial/extra/3milextra.npy",allow_pickle=True)
X_mean=extrainfo.item()['X_mean']#.to(device)
X_std=extrainfo.item()['X_std']#.to(device)
Y_mean=extrainfo.item()['Y_mean']#.to(device)
Y_std=extrainfo.item()['Y_std'].to('cpu')#.to(device)

#normalizing samples and data vectors to mean 0, std 1

X_train=(train_samples-X_mean)/X_std
X_train[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization
#y_train=(train_data_vectors-Y_mean)/Y_std

#print(train_data_vectors.dtype)
#print(X_train.dtype)
X_validation=(validation_samples-X_mean)/X_std
X_validation[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization
y_validation=(validation_data_vectors-Y_mean)/Y_std
Y_std=Y_std.to(device)
X_train=X_train.to(torch.float32)
X_validation=X_validation.to(torch.float32)
#load the data to batches. Do not send those to device yet to save space
#del train_data_vectors
del validation_data_vectors
batch_size=512
trainset    = TensorDataset(X_train, y_train)
validset    = TensorDataset(X_validation,y_validation)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)


#Set up the model and optimizer

#training

n_epoch=900#for trial test purpose

intdim=4
Nlayer=2

model = CNNMLP(input_dim=input_size,output_dim=out_size,int_dim=intdim,N_layer=Nlayer)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.001)

model = nn.DataParallel(model)
model.to(device)


losses_train = []
losses_vali = []
losses_train_med = []
losses_vali_med = []


reduce_lr = True#reducing learning rate on plateau
if reduce_lr==True:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=15)

for n in range(n_epoch):
    losses=[]
    for i, data in enumerate(trainloader):
        model.train()
        X = data[0].to(device)# send to device one by one
        Y_batch = data[1].to(device)# send to device one by one
        Y_pred  = model(X).to(device)
        diff = (Y_batch - Y_pred)*Y_std# Scale back to unit by *Y_std
        

        
        loss1 = torch.diag(diff @ covinv @ torch.t(diff))# implement with torch.einsum
        loss1 = torch.sqrt(1+2*loss1)
        loss=torch.mean(loss1)
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
            v_diff = (Y_v_batch - Y_v_pred )*Y_std
            
            loss1 = torch.diag(v_diff @ covinv @ torch.t(v_diff))# implement with torch.einsum
            loss1 = torch.sqrt(1+2*loss1)
            loss_vali=torch.mean(loss1)
            losses.append(loss_vali.cpu().detach().numpy())

        losses_vali.append(np.mean(losses))
        losses_vali_med.append(np.median(losses))

        if reduce_lr == True:
            print('Reduce LR on plateu: ',reduce_lr)
            scheduler.step(losses_vali[n])

    print('epoch {}, loss={}, validation loss={}, lr={}, wd={})'.format(
                        n,
                        losses_train[-1],
                        losses_vali[-1],
                        optimizer.param_groups[0]['lr'],
                        optimizer.param_groups[0]['weight_decay']
                        
                    ))#, total runtime: {} ({} average))



PATH = "./trainedemu5000cnn/chiTTAstaucnn3mil"#rename as drop0 afterward
torch.save(model.state_dict(), PATH+'.pt')
#extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std}
#np.save(PATH+'.npy',extrainfo)
np.save(PATH+'losstrain.npy',losses_train)
np.save(PATH+'lossvali.npy',losses_vali)
np.save(PATH+'losstrainmed.npy',losses_train_med)
np.save(PATH+'lossvalimed.npy',losses_vali_med)
