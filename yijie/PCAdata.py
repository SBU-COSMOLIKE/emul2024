import numpy as np
from sklearn.decomposition import IncrementalPCA

camb_ell_min          = 2#30
camb_ell_max          = 2509
camb_ell_range        = camb_ell_max  - camb_ell_min 

train_samples=np.load('YZ_samples/LHS/coslhc_acc.npy',allow_pickle=True)# This is actually a latin hypercube sampling of 1mil points
vali_samples=np.load('YZ_samples/Uniform/input/cosuni_0.npy',allow_pickle=True)

train_data_vectors=np.load('YZ_samples/LHS/coslhc_acc_output.npy',allow_pickle=True)[:,:camb_ell_range,0]
train_data_vectors=np.log(train_data_vectors)
vali_data_vectors=np.load('YZ_samples/Uniform/output/cosuni_0_output.npy',allow_pickle=True)[:,:camb_ell_range,0]
for i in range(1,10):
    samp_new=np.load('YZ_samples/Uniform/input/cosuni_'+str(i)+'.npy',allow_pickle=True)
    dv_new=np.load('YZ_samples/Uniform/output/cosuni_'+str(i)+'_output.npy',allow_pickle=True)[:,:camb_ell_range,0]
    vali_samples=np.vstack((vali_samples,samp_new))
    vali_data_vectors=np.vstack((vali_data_vectors,dv_new))



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

validation_data_vectors=np.log(validation_data_vectors)

mean=np.mean(train_data_vectors,axis=0)
std=np.std(train_data_vectors,axis=0)
X=(train_data_vectors-mean)/std
vali_X=(validation_data_vectors-mean)/std
n_pca=64
batchsize=43
PCA = IncrementalPCA(n_components=n_pca,batch_size=batchsize)

for batch in np.array_split(X, batchsize):
    PCA.partial_fit(batch)


train_pca=PCA.transform(X)

vali_pca=PCA.transform(vali_X)

extra_info={'sample_mean':mean,'sample_std':std}

np.save('YZ_samples/LHS/coslhc_acc_pca.npy',train_pca)

np.save('YZ_samples/Uniform/input/cosuni_acc_10.npy',validation_samples)

np.save('YZ_samples/Uniform/output/cosuni_10_output_pca.npy',vali_pca)

np.save('YZ_samples/PCAcomp/msett_comp.npy',PCA.components_)

np.save('YZ_samples/PCAcomp/msett_sampleinfo.npy',extra_info)