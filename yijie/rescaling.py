import numpy as np
import scipy
import scipy.linalg
from scipy import integrate
#the equations of this code come from https://iopscience.iop.org/article/10.1086/303928/pdf
def par_to_a(par):
    #omega_b=par[0]
    omega_0=par[0]+par[1] #omega_0=
    a1=0.0396*omega_0**(-0.248)*(1+13.6*omega_0**(0.638))
    a2=1480*omega_0**(-0.0606)/(1+10.2*omega_0**(0.553))
    a3=1.03*omega_0**(0.0335)
    a4=-0.0473*omega_0**(-0.0639)
    return np.array([a1,a2,a3,a4])


def par_to_r_theta(par):
    omega_b=par[0]
    omega_0=par[0]+par[1]
    c=3e5
    H0=par[2]
    h=H0/100
    a_eq=(4.17e-5)/omega_0*(2.726/2.728)**4
    b1=0.0783*omega_b**(-0.238)/(1+39.5*omega_b**(0.763))
    b2=0.56/(1+21.1*omega_b**(1.81))
    a_star=1/(1+1048*(1+0.00124*omega_b**(-0.738))*(1+b1*(omega_0**b2)))
    integrand=lambda x:1/H0/np.sqrt(omega_0/h**2*x**(-3)+(1-omega_0/h**2)+2.47e-5/h**2*x**(-4))/x**2
    eta_0=integrate.quad(integrand, 0,1)[0]#2*(omega_0*10000)**(-0.5)*(np.sqrt(1+a_eq)-np.sqrt(a_eq))*(1-0.0841*np.log(omega_0/h**2))
    eta_star=integrate.quad(integrand,0, a_star)[0]#2*(omega_0*10000)**(-0.5)*(np.sqrt(a_star+a_eq)-np.sqrt(a_eq))
    return (eta_0-eta_star)*c
def par_to_lD_m(par):
    omega_b=par[0]
    omega_0=par[0]+par[1]
    a=par_to_a(par)
    r_theta=par_to_r_theta(par)
    l_D=r_theta*a[0]*omega_b**(0.291)*(1+a[1]*omega_b**(1.8))**(-0.2)
    m=a[2]*omega_b**(a[3])*(1+omega_b**(1.8))**(0.2)
    return np.array([l_D,m])

def par_to_D_l(ell,par):
    l_D,m=par_to_lD_m(par)
    return np.exp(-(ell/l_D)**m)


def par_to_lr(par):
    
    
    omega_b=par[0]
    omega_0=par[0]+par[1]
    #c=3e5
    H0=par[2]
    h=H0/100
    a_eq=(4.17e-5)/omega_0*(2.726/2.728)**4

    eta_0=2*(omega_0*10000)**(-0.5)*(np.sqrt(1+a_eq)-np.sqrt(a_eq))*(1-0.0841*np.log(omega_0/h**2))
    eta_r=2*(omega_0*10000)**(-0.5)*(np.sqrt(1/11+a_eq)-np.sqrt(a_eq)) #picked z=10 for reionizatoin

    return (eta_0-eta_r)/eta_r


def par_to_Rl2(ell,par):
    tau=par[3]
    lr=par_to_lr(par)
    x=ell/(lr+1)
    c1=-0.276
    c2= 0.581
    c3=-0.172
    c4= 0.0312
    return np.exp(-2*tau)+(1-np.exp(-2*tau))/(1+c1*x+c2*x**2+c3*x**3+c4*x**4)

def par_to_P_l(ell,par):
    rho_nu=7/8*3.046*(4/11)**(4/3)
    rho_gamma=1
    f_nu=rho_nu/(rho_nu+rho_gamma)
    omega_b=par[0]
    omega_0=par[0]+par[1]
    b1=0.0783*omega_b**(-0.238)/(1+39.5*omega_b**(0.763))
    b2=0.56/(1+21.1*omega_b**(1.81))
    a_star=1/(1+1048*(1+0.00124*omega_b**(-0.738))*(1+b1*(omega_0**b2)))
    R_star=omega_b*30000*a_star
    omega_0=par[0]+par[1]
    c=3e5
    H0=par[2]
    h=H0/100
    r_theta=par_to_r_theta(par)
    a_eq=(4.17e-5)/omega_0
    k_eq=np.sqrt(omega_0*10000*2/a_eq)
    l_eq=r_theta*k_eq/c
    A=25/(1+4/15*f_nu)**2*(1/np.sqrt(1+R_star)+(1+R_star)**(-3/2))/2-1
    #print(l_eq)
    return A*np.exp(-1.4*l_eq/ell)+1

def par_to_rescale(par,camb_ell_min,camb_ell_max):
    ell=np.arange(camb_ell_min,camb_ell_max,1)
    A_s=np.exp(par[5])
    tau=par[3]
    rescale=A_s/np.exp(2*tau)*(par_to_D_l(ell,par))**2*par_to_Rl2(ell,par)*par_to_P_l(ell,par)
    return rescale

def par_tot_to_rescale(partot,camb_ell_min,camb_ell_max):

	camb_ell_range=camb_ell_max-camb_ell_min

	rescale_tot=np.zeros((len(partot),camb_ell_range))
	

	for i in range(len(partot)):
		rescale_tot[i]=par_to_rescale(partot[i],camb_ell_min,camb_ell_max)

	return rescale_tot


