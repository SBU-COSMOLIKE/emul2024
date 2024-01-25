import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#define model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear =nn.Sequential(
            nn.Linear(5, 100),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 2970)
        )
    def forward(self, x):
        out = self.linear(x)
        return out

model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters())
fid=np.load('YZ_CMBuniform2/TT/fid.npy',allow_pickle=True)
fid=torch.Tensor(fid)

#load in data
samples=np.load('CMBSO/uniform2.npy',allow_pickle=True)
TT=np.load('YZ_CMBuniform2/TT/TT_1.npy',allow_pickle=True)
#TE=np.load('YZ_CMBuniform2/TE/TE_1.npy',allow_pickle=True)
#EE=np.load('YZ_CMBuniform2/EE/EE_1.npy',allow_pickle=True)

for i in range(2,11):
    TTnew=np.load('YZ_CMBuniform2/TT/TT_'+str(i)+'.npy',allow_pickle=True)
    #TEnew=np.load('YZ_CMBuniform2/TE/TE_'+str(i)+'.npy',allow_pickle=True)
    #EEnew=np.load('YZ_CMBuniform2/EE/EE_'+str(i)+'.npy',allow_pickle=True)
    TT=np.vstack((TT,TTnew))
    #TE=np.vstack((TE,TEnew))
    #EE=np.vstack((EE,EEnew))
data_vectors=TT#np.concatenate((TT,TE,EE),axis=1)

#assign training and validation sets
train_samples=samples[:150000]
train_data_vectors=data_vectors[:150000]
validation_samples=samples[150000:]
validation_data_vectors=data_vectors[150000:]

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
y_train=(train_data_vectors-Y_mean)/Y_std

val_X_mean=torch.Tensor(validation_samples.mean(axis=0, keepdims=True))
val_X_std  = torch.Tensor(validation_samples.std(axis=0, keepdims=True))
val_Y_mean=torch.Tensor(validation_data_vectors.mean(axis=0, keepdims=True))
val_Y_std=torch.Tensor(validation_data_vectors.std(axis=0, keepdims=True))
X_validation=(validation_samples-val_X_mean)/val_X_std
y_validation=(validation_data_vectors-val_Y_mean)/val_Y_std

#load the data to batches
batch_size=2000
trainset    = TensorDataset(X_train, y_train)
validset    = TensorDataset(X_validation,y_validation)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

#training
n_epoch=100#for trial test purpose
losses_train = []
losses_vali = []

for n in range(n_epoch):
    
    model.train()
    losses=[]
    for i, data in enumerate(trainloader):
        X = data[0].to(device)
        Y_batch = data[1].to(device)
        Y_pred  = model(X)
        diff = (Y_batch - Y_pred)*Y_std+Y_mean

        dif=torch.div(diff,fid)
        loss1 = dif@ torch.t(dif)
        #print(loss)
        loss=torch.mean(torch.diag(loss1))
        losses.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses_train.append(np.mean(losses))

    with torch.no_grad():
        model.eval()
        
        losses = []
        for i, data in enumerate(validloader):
            X_v       = data[0].to(device)
            Y_v_batch = data[1].to(device)
            Y_v_pred = model(X_v) 
            v_diff = (Y_v_batch - Y_v_pred )* val_Y_std+val_Y_mean
            v_dif=torch.div(v_diff,fid)
            loss1 = v_dif@ torch.t(v_dif)
            #print(loss)
            loss_vali=torch.mean(torch.diag(loss1))
            losses.append(loss_vali.cpu().detach().numpy())

        losses_vali.append(np.mean(losses))

    print('epoch {}, loss={}, validation loss={}, lr={} )'.format(
                        n,
                        losses_train[-1],
                        losses_vali[-1],
                        optimizer.param_groups[0]['lr']
                        
                    ))#, total runtime: {} ({} average))



# Save
PATH = "./trainedemu/trial1"
torch.save(model.state_dict(), PATH+'.pt')
extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std}
np.save(PATH+'.npy',extrainfo)
