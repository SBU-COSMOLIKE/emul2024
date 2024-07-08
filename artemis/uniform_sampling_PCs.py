#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np 
import matplotlib.pyplot as pl
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
import time 
#import corner
import h5py
#import ipdb
from scipy import integrate 
#from tqdm import tqdm


# # Basic definitions

# In[2]:


# magnitude of speed of light
cmag = 149896229 / 500


# In[3]:


# Redshifts for the bins that we chose when we did the eigevector analysis with Fisher.
Nx = 500
zzmin = 0.03
zzmax = 1.7
zBinsFisher = [zzmin + i * (zzmax - zzmin) / Nx for i in range(1, Nx + 1)]


# - Import the eigenfunctions from the mathematica output file.

# In[4]:


# ## Add code here to read of the e_i eigenvectors
# eigenvectors = [[0] * 850 for _ in range(500)]
# eigenvectors = np.array(eigenvectors)

eigenvectors = np.loadtxt("/gpfs/home/argiannakopo/cosmo_stuff/eigenvectorsFisherTot.dat")
print('eigenvectors.shape',eigenvectors.shape)


# In[5]:


## Define DE equation of state as a sum over the eigenvectors times some coefficient alpha

# This function returns an array [1, number of z bins] which corresponds DE EoS for each redshift in each bin.
def w(alphas,eigenvectors):
    
    weighted_eigenvectors = alphas[:,np.newaxis] * eigenvectors

    summed_vector = -1 + np.sum(weighted_eigenvectors, axis=0)

    return -1 + summed_vector
    
    


# **IMPORTANT NOTE**
# - **Here I have assumed that I am interested in $z \in [0.03,1.7]$ and defined 500 bins in this range**

def aux(z1, z2, w1):
    return ( (1 + z2) / (1 + z1) ) ** (3*(1 + w1))

def omegade(zbins, alphas, eigenvectors):
    w_values = w(alphas, eigenvectors)
    zbins = np.insert(zbins, 0, 0.03)
    N = len(w_values)
    if len(zbins) != N + 1 :
        raise ValueError("Number of zbins is not correct")
        
    cumulative_product = []
    current_product = 1
        
    for i in range(N):
        current_product *= aux(zbins[i], zbins[i+1], w_values[i])
        cumulative_product.append(current_product)
        
    return np.array(cumulative_product)
    


# - The way I defined things, I have an issue with luminosity distance $d_L$ since my bins from summing the eigenvectors go from $z_{min}$ to $z_{max}$ with $z_{min} \neq 0$. However, in the definition of the luminosity distance I have an integral from 0 to z. To deal with this part of $z < z_{min}$, I will evaluate a constant factor with $w_{DE} = w_{fid} = -1$ and add this to each element of the array within the zbins that I care about.
# 

# In[9]:


## I need to figure out how to define angular distance 

def hubble(zbins, alphas, eigenvectors, H0, Om, Ok ):

    x = 1 - Om - Ok 
    
    if x < 0:
        print("Error: Bad Input, Om, Ok")
    else:
        return np.array( H0 * np.sqrt( Om * ( 1 + zbins) ** 3 + x * omegade(zbins, alphas, eigenvectors) + Ok * (1 + zbins)**2) )



# In[10]:


## Define the comoving distance. 
## Since the zbins start at zmin, I will break the integral in two pieces: [0,zmin) and [zmin,z]

# ## This function return a numpy array that gives the comoving distance at each redshift bin. 
# def comov_dist(zbins, alphas, eigenvectors, H0, Om, Ok):
#     x = 1 - Om - Ok
#     if x < 0:
#         return print('Comoving dist: Bad Input, Om, Ok')
#     else:
#         # Bin the [0, zmin] interval and do the integration
#         x1, x2 = 0, zbins[0]
#         Nx = 5 + 5 * int(abs(x2 - x1) * 100)
#         dx = (x2 - x1) / Nx
#         dz = zbins[1] - zbins[0]
#         less_zmin_arr = Parallel(n_jobs=-1)(delayed(lambda i: dx / (H0 * np.sqrt(Om * (1 + x1 + dx * i)**3 + x * (1+ x1 + dx * i) + Ok * (1 + x1 + dx * i)**2)))(i) for i in range(0, Nx))
#         less_zmin =  np.sum(less_zmin_arr)
        
#         return cmag * less_zmin +  cmag * np.cumsum( dz / hubble(zbins, alphas, eigenvectors, H0, Om, Ok))
    
## Define the comoving distance. 
## Since the zbins start at zmin, I will break the integral in two pieces: [0,zmin) and [zmin,z]

## This function return a numpy array that gives the comoving distance at each redshift bin. 
def comov_dist(zbins, alphas, eigenvectors, H0, Om, Ok):
    x = 1 - Om - Ok
    if x < 0:
        return print('Comoving dist: Bad Input, Om, Ok')
    else:
        # Write the comoving distance function for z < zmin where w_de = -1, so that I can use scipy to do the integration for that part
        def f(x1,x, H0, Om, Ok):
            return 1 / ( H0 * np.sqrt( Om * pow(1+x1 ,3 ) + x * (1 + x1) + Ok * pow(1+x1, 2)))
            
        less_zmin = integrate.quad(f, 0, zbins[0], args=(x, H0, Om, Ok))
        # # Bin the [0, zmin] interval and do the integration
        # x1, x2 = 0, zbins[0]
        # Nx = 5 + 5 * int(abs(x2 - x1) * 100)
        # dx = (x2 - x1) / Nx
        dz = zbins[1] - zbins[0]
        # less_zmin_arr = Parallel(n_jobs=-1)(delayed(lambda i: dx / (H0 * np.sqrt(Om * (1 + x1 + dx * i)**3 + x * (1+ x1 + dx * i) + Ok * (1 + x1 + dx * i)**2)))(i) for i in range(0, Nx))
        # less_zmin =  np.sum(less_zmin_arr)
       
        return cmag * less_zmin[0] +  cmag * np.cumsum( dz / hubble(zbins, alphas, eigenvectors, H0, Om, Ok))
    
    


# In[11]:


## Returns an array of luminosity distances at each redshift z 
def lum_distance(zbins, alphas, eigenvectors, H0, Om, Ok):
    z_shifted = 1 + zbins

    return z_shifted * comov_dist(zbins, alphas, eigenvectors, H0, Om, Ok)
    

# In[13]:

# Define the logarithm of dL
# Returns the logarithm of the array of luminosity distance for each redshift bin
def log_h0dl(zbins, alphas, eigenvectors, H0, Om, Ok):
    return np.log10( (H0 / cmag) * lum_distance(zbins, alphas, eigenvectors, H0, Om, Ok))
    

## Uniform Sampling

# In[23]:


zBinsFisher = np.linspace(0.0334, 1.7, 500)
## In order to define the correct prior for the PC amplitudes, I need to define the allowed min and max values for each alpha_i.
## I can do that outside of the function that I will write below since these things do not change with each random step that the MCMC makes

# Theoretical limits on DE
wfid = -1
# Quintessence
wmin, wmax = -1, 1
# Smooth DE
# wmin, wmax = -5, 3
weight1 = wmin + wmax - 2 * wfid
weight2 = wmax - wmin 

# First I need to calculate each of the two relevant sums in the eq (A10)
first_sum = np.sum(eigenvectors, axis=1)
second_sum = np.sum(np.abs(eigenvectors), axis=1)

## Nz is the number of z bins I used 
Nz = 500

# alpha_max and alpha_min are two arrays that are storing the max and min value of the amplitudes alpha_i 
alpha_max = ( 1 / (2*Nz)) * ( weight1 * first_sum + weight2 * second_sum)
alpha_min = ( 1 / (2*Nz)) * ( weight1 * first_sum - weight2 * second_sum)



# - I will sample only for $\alpha_1$, $\Omega_m$ and $\Omega_m h^2$. I will set to zero the rest of the PCs. Also, note that based on the calculation in the previous cell, I know the min and max values that are allowed for $\alpha_1$.
# - The following should create two .h5 files that contain the cosmological parameters and the respective values for logdL, so that I can use them for training my ResMLP.



## Define the limits of the sampling region -- ([alphas], Omega_m, Omega_mh^2) 
## The fiducial model that I am using has Omega_m = 0.24 and h = 0.73 -- Omega_m h^2 = 0.127896 
low_lim = np.concatenate([alpha_min[:50],[0.14, 0.055566]])
high_lim = np.concatenate([alpha_max[:50],[0.34, 0.234226]])
# low_lim = [alpha_min[0], 0.14 , 0.055566]
# high_lim = [alpha_max[0], 0.34, 0.234226]


# In[28]:


## Will save the uniform sampled cosmological parameters in an hdf5 file
with h5py.File('/gpfs/scratch/argiannakopo/uniform_cosmo_data_50PC.h5', 'w') as f:
    # Will save things in chunks so that it is easier to load and read later
    dataset = f.create_dataset('data', shape=(0, 52), maxshape=(None, 52), dtype=np.float64, chunks=True)

    chunk_size = 100000  # Number of rows per chunk
    for start in range(0, 100000, chunk_size):
        end = min(start + chunk_size, 100000)
        chunk = np.random.uniform(low=low_lim, high=high_lim, size=(end - start, 52))
        dataset.resize((dataset.shape[0] + chunk.shape[0]), axis=0)
        dataset[-chunk.shape[0]:] = chunk


# In[37]:


# Input and output file names
input_filename = '/gpfs/scratch/argiannakopo/uniform_cosmo_data_50PC.h5'
output_filename = '/gpfs/scratch/argiannakopo/uniform_lodL_data_50PC.h5'
zeros = [0] * (500 - 50)

# Read the input file and process each row
with h5py.File(input_filename, 'r') as f_in, h5py.File(output_filename, 'w') as f_out:
    dataset_in = f_in['data']
    num_rows, num_cols = dataset_in.shape
    
    
    dataset_out = f_out.create_dataset('processed_data', shape=(num_rows, 500), dtype=np.float64, chunks=True)
    
    # Process each row and save the result
    for i in range(num_rows):
        row = dataset_in[i]
        alps = np.concatenate([row[:50],zeros])  
        processed_row = log_h0dl(zBinsFisher, alps, eigenvectors, np.sqrt(row[-1]/row[-2])*100, row[-2], 0)
        dataset_out[i] = processed_row

print("Processing complete. Data saved to:", output_filename)


