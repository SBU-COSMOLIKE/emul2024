import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset


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

class Better_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = torch.reshape(prod,(batch_size,-1))+x # reshape back to vector

        return out

class Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(Better_Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = nn.Tanh()#nn.ReLU()#
        self.norm         = torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights = nn.Parameter(weights) # turn the weights tensor into trainable weights
        bias = torch.Tensor(in_size)
        self.bias = nn.Parameter(bias) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5)) # matrix weights init 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights) # fan_in in the input size, fan out is the output size but it is not use here
        bound = 1 / np.sqrt(fan_in) 
        nn.init.uniform_(self.bias, -bound, bound) # bias weights init

    def forward(self,x):
        mat = torch.block_diag(*self.weights) # how can I do this on init rather than on each forward pass?
        x_norm = self.norm(x)
        _x = x_norm.reshape(x_norm.shape[0],self.n_partitions,self.int_dim) # reshape into channels
        o = self.act(torch.matmul(x_norm,mat)+self.bias)
        return o+x

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



class TRF(nn.Module):

    def __init__(self, input_dim, output_dim, int_dim, N_layer):

        super(TRF, self).__init__()

        modules=[]

        # Def: we will set the internal dimension as multiple of 128 (reason: just simplicity)
        int_dim = int_dim * 128

        # Def: we will only change the dimension of the datavector using linear transformations  
        
        
        n_channels = 32
        int_dim_trf = 1024

        modules.append(nn.Linear(input_dim, int_dim))
        modules.append(ResBlock(int_dim, int_dim))
        modules.append(nn.Tanh())
        modules.append(ResBlock(int_dim, int_dim))
        modules.append(nn.Tanh())
        modules.append(ResBlock(int_dim, int_dim))
        modules.append(nn.Tanh())
        modules.append(nn.Linear(int_dim, int_dim_trf))
        modules.append(Better_Attention(int_dim_trf, n_channels))
        modules.append(Better_Transformer(int_dim_trf, n_channels))
        modules.append(Better_Attention(int_dim_trf, n_channels))
        modules.append(Better_Transformer(int_dim_trf, n_channels))
        modules.append(Better_Attention(int_dim_trf, n_channels))
        modules.append(Better_Transformer(int_dim_trf, n_channels))
        modules.append(nn.Linear(int_dim_trf, output_dim))
        modules.append(Affine())

        
        self.trf =nn.Sequential(*modules)#

    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        out = self.trf(x)
        return out


#Set up the Covariance matrix
covinv=np.load('YZ_samples/LHS/cosvarinvTT.npy',allow_pickle=True)[:camb_ell_range,:camb_ell_range]*4/np.exp(4*0.06)
covinv=torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix

#load in data
train_samples=np.load('YZ_samples/LHS/coslhc_acc.npy',allow_pickle=True)#.astype('float32')# This is actually a latin hypercube sampling of 1mil points
validation_samples=np.load('YZ_samples/Uniform/input/cosuni_acc_10.npy',allow_pickle=True)#.astype('float32')
lhc_len=len(train_samples)
input_size=len(train_samples[0])




validation_data_vectors=np.load('YZ_samples/Uniform/output/cosuni_10_output_acc_vali.npy',allow_pickle=True)[:,:camb_ell_range]
uniset=[20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
for i in uniset:
    samp_new=np.load('YZ_samples/Uniform/input/cosuni_acc_'+str(i)+'.npy',allow_pickle=True)
    #dv_new=np.load('YZ_samples/Uniform/output/cosuni_'+str(i)+'_output_acc.npy',allow_pickle=True)[:,:camb_ell_range]
    train_samples=np.vstack((train_samples,samp_new))
    #train_data_vectors=np.vstack((train_data_vectors,dv_new))
train_len=len(train_samples)
step=lhc_len
train_data_vectors=torch.zeros((train_len,camb_ell_range),dtype=torch.float32)
train_data_vectors[0:lhc_len,:]=torch.Tensor(np.load('YZ_samples/LHS/coslhc_acc_output_TT.npy',allow_pickle=True)[:,:camb_ell_range])
for i in uniset:
    samp_new=np.load('YZ_samples/Uniform/input/cosuni_acc_'+str(i)+'.npy',allow_pickle=True)
    step_new=step+len(samp_new)
    #train_samples=np.vstack((train_samples,samp_new))
    train_data_vectors[step:step_new,:]=torch.Tensor(np.load('YZ_samples/Uniform/output/cosuni_'+str(i)+'_output_acc.npy',allow_pickle=True)[:,:camb_ell_range])
    step=step_new
for i in range(len(train_data_vectors)):
    train_data_vectors[i]=train_data_vectors[i]/(np.exp(train_samples[i,5]))*(np.exp(2*train_samples[i,3]))
for i in range(len(validation_data_vectors)):
    validation_data_vectors[i]=validation_data_vectors[i]/(np.exp(validation_samples[i,5]))*(np.exp(2*validation_samples[i,3]))

out_size=1*len(train_data_vectors[0])
#assign training and validation sets

train_samples=torch.from_numpy(train_samples)#.to(device)
#train_data_vectors=torch.from_numpy(train_data_vectors)#.to(device)
validation_samples=torch.from_numpy(validation_samples)#.to(device)
validation_data_vectors=torch.from_numpy(validation_data_vectors)#.to(device)

#extrainfo=np.load("./trainedemu5000resnetnew/chiTTAstauresneti4l4.npy",allow_pickle=True)
#X_mean=extrainfo.item()['X_mean']#.to(device)
#X_std=extrainfo.item()['X_std']#.to(device)
#Y_mean=extrainfo.item()['Y_mean']#.to(device)
#Y_std=extrainfo.item()['Y_std'].to('cpu')#.to(device)

#normalizing samples and data vectors to mean 0, std 1
X_mean=torch.Tensor(train_samples.mean(axis=0, keepdims=True))
X_std  = torch.Tensor(train_samples.std(axis=0, keepdims=True))
Y_mean=torch.Tensor(train_data_vectors.mean(axis=0, keepdims=True))
Y_std=torch.Tensor(train_data_vectors.std(axis=0, keepdims=True))
X_train=(train_samples-X_mean)/X_std
X_train[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization
#y_train=(train_data_vectors-Y_mean)/Y_std
for i in range(len(train_data_vectors)):
    train_data_vectors[i]=(train_data_vectors[i]-Y_mean)/Y_std
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
trainset    = TensorDataset(X_train, train_data_vectors)
validset    = TensorDataset(X_validation,y_validation)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)


#Set up the model and optimizer

#training

n_epoch=900#for trial test purpose

intdim=2
Nlayer=3

model = TRF(input_dim=input_size,output_dim=out_size,int_dim=intdim,N_layer=Nlayer)
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



PATH = "./trainedemu5000trf/chiTTAstautrfmore"#rename as drop0 afterward
torch.save(model.state_dict(), PATH+'.pt')
extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std}
np.save(PATH+'.npy',extrainfo)
np.save(PATH+'losstrain.npy',losses_train)
np.save(PATH+'lossvali.npy',losses_vali)
np.save(PATH+'losstrainmed.npy',losses_train_med)
np.save(PATH+'lossvalimed.npy',losses_vali_med)
