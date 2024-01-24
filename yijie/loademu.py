import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)

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