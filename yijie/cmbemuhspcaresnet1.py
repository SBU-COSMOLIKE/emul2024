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


class Supact(nn.Module):
    # New activation function, returns:
    # f(x)=(gamma+(1+exp(-beta*x))^(-1)*(1-gamma))*x
    # gamma and beta are trainable parameters.
    # I chose the initial value for gamma to be all 1, and beta to be all 0
    def __init__(self, in_size):
        super(Supact, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(in_size))
        self.beta = nn.Parameter(torch.zeros(in_size))
        self.m = nn.Sigmoid()
    def forward(self, x):
        inv = self.m(torch.mul(self.beta,x))
        fac = 1-self.gamma
        mult = self.gamma + torch.mul(inv,fac)
        return torch.mul(mult,x)

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

        self.act1 = Supact(in_size)#nn.Tanh()#nn.ReLU()#
        self.act2 = Supact(in_size)#nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.layer1(self.norm1(x)))
        o2 = self.act2(self.layer2(self.norm2(o1))) + xskip

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
            modules.append(Supact(int_dim))
        
        # Def: the transformation from the internal dimension to the output dimension of the
        #      data vector we intend to emulate
        
        modules.append(nn.Linear(int_dim, output_dim))
        modules.append(Affine())
        # NN.SEQUENTIAL is a PYTHORCH function DEFINED AT: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # This function stacks up layers in the modules-list in sequence to create the whole model
        self.resmlp =nn.Sequential(*modules)#

    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        out = self.resmlp(x)

        return out


#Set up the Covariance matrix



covinv=np.load('extra/cosvarinvTT.npy',allow_pickle=True)[:camb_ell_range,:camb_ell_range]#*4/np.exp(4*0.06)
covinv=torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix
transform_matrix=np.load('extra/PCAmat_hs_5mil.npy',allow_pickle=True)[:512]
transform_matrix = torch.Tensor(transform_matrix).to(device)
#load in data
train_samples=np.load('parametersamples/hypersphere/coshs_train.npy',allow_pickle=True)#.astype('float32')# This is actually a latin hypercube sampling of 1mil points

input_size=len(train_samples[0])

validation_samples=np.load('parametersamples/hypersphere/coshs_vali.npy',allow_pickle=True)

train_data_vectors=np.load('datavectors/hypersphere/coshs_TT_train.npy',allow_pickle=True)

validation_data_vectors=np.load('datavectors/hypersphere/coshs_TT_vali.npy',allow_pickle=True)
#train_data_vectors=torch.zeros((lhc_len,camb_ell_range),dtype=torch.float32)

out_size=512#1*len(train_data_vectors[0])
#assign training and validation sets

train_samples=torch.Tensor(train_samples)#.to(device)
train_data_vectors=torch.Tensor(train_data_vectors)#.to(device)
validation_samples=torch.Tensor(validation_samples)#.to(device)
validation_data_vectors=torch.Tensor(validation_data_vectors)#.to(device)

extrainfo=np.load("extra/extrainfo_hs_5mil.npy",allow_pickle=True)
X_mean=torch.Tensor(extrainfo.item()['X_mean'])#.to(device)
X_std=torch.Tensor(extrainfo.item()['X_std'])#.to(device)
Y_mean=torch.Tensor(extrainfo.item()['Y_mean']).to(device)
Y_std=torch.Tensor(extrainfo.item()['Y_std']).to(device)
Y_mean2=torch.Tensor(extrainfo.item()['Y_mean2']).to(device)
Y_std2=torch.Tensor(extrainfo.item()['Y_std2']).to(device)

#normalizing samples and data vectors to mean 0, std 1

X_train=(train_samples-X_mean)/X_std
X_train[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization
#y_train=(train_data_vectors-Y_mean)/Y_std

#print(train_data_vectors.dtype)
#print(X_train.dtype)
X_validation=(validation_samples-X_mean)/X_std
X_validation[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization
#y_validation=(validation_data_vectors-Y_mean)/Y_std
X_train=X_train.to(torch.float32)
X_validation=X_validation.to(torch.float32)
#load the data to batches. Do not send those to device yet to save space

X_mean=X_mean.to(device)
X_std=X_std.to(device)

batch_size=512
trainset    = TensorDataset(X_train, train_data_vectors)
validset    = TensorDataset(X_validation,validation_data_vectors)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

#Set up the model and optimizer

#training

n_epoch=900#for trial test purpose
iset=[4,8]

Nlayer=4
for intdim in iset:
    #PATH = "./trainedemu5000resnetnew/chiTTAstauresneti"+str(intdim)+"l"+str(Nlayer)
    
    model = ResMLP(input_dim=input_size,output_dim=out_size,int_dim=intdim,N_layer=Nlayer)
    

    model = nn.DataParallel(model)
    model.to(device)
    #model.load_state_dict(torch.load(PATH+'.pt',map_location=device))
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0)


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
            As=torch.exp(X[:,5]*X_std[0,5]+X_mean[0,5])
            exptau=torch.exp(2*X[:,3]*X_std[0,3]+2*X_mean[0,3])

            Y_pred =  Y_pred*Y_std2+Y_mean2# Scale back to unit by *Y_std
            Y_pred = torch.matmul(Y_pred, transform_matrix)*Y_std+Y_mean
            Y_pred = Y_pred *As[:,None]/exptau[:,None]
            diff = Y_pred - Y_batch
        

        
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
                As=torch.exp(X_v[:,5]*X_std[0,5]+X_mean[0,5])
                exptau=torch.exp(2*X_v[:,3]*X_std[0,3]+2*X_mean[0,3])
                Y_v_pred_back = torch.matmul((Y_v_pred*Y_std2+Y_mean2),transform_matrix)*Y_std+Y_mean
                Y_v_pred_back = Y_v_pred_back *As[:,None]/exptau[:,None]
                v_diff = (Y_v_batch - Y_v_pred_back)
                
                
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



    PATH = "./trainedemu5000hypersphere/chiTTAstauresunscalehs512i"+str(intdim)+'l'+str(Nlayer)#rename as drop0 afterward
    torch.save(model.state_dict(), PATH+'.pt')
    #extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std,'Y_mean2':Y_mean2,'Y_std2':Y_std2}
    #np.save(PATH+'.npy',extrainfo)
    np.save(PATH+'losstrain.npy',losses_train)
    np.save(PATH+'lossvali.npy',losses_vali)
    np.save(PATH+'losstrainmed.npy',losses_train_med)
    np.save(PATH+'lossvalimed.npy',losses_vali_med)
