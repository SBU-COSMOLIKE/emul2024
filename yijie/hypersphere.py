import numpy as np

N = #number of samples

d = #dimension of parameter vectors



up = np.array([]) #upper bounds
low = np.array([]) #lower bounds

def hypesphere(N,d,up,low):
    mean = np.zeros(d)
    cov = np.identity(d)
    S = np.random.multivariate_normal(mean, cov, N)
    Rnew = np.random.uniform(0,1,N)
    for i in range(N):
        S[i] = S[i]/np.linalg.norm(S[i])*Rnew[i]**(1/d)
    a = (up-low)/2
    b = (up+low)/2

    for j in range(d):
        S[:,j] = S[:,j]*a[j]+b[j]
    return S
S=hypesphere(N,d,up,low)# this is good if you do not need correlation


#### if you need to add in correlation between parameters ####
##############################################################

C = # a cov mat
L = np.linalg.cholesky(C)

S_cor = np.matmul(S,L)
