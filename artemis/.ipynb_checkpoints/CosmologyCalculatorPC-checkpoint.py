import numpy as np
from scipy import integrate

class CosmologyCalculatorPC:
    def __init__(self, H0, Om, Ok, alphas, eigenvectors):
        
        """
        Initialize the CosmologyCalculator with cosmological parameters and Principal Componenents.

        Parameters:
        - H0: Hubble constant in km/s/Mpc
        - Om: Matter density parameter
        - Ok: Curvature density parameter
        - alphas: Array of amplitudes for the PCs
        - eigenvectors: Array of PCs

        We have assumed here that ORhh = ONhh + OGhh = 4.18095e-05
        """
        self.H0 = H0
        self.Om = Om
        self.Ok = Ok
        self.alphas = alphas
        self.eigenvectors = eigenvectors
        self.cmag = 149896229 / 500  # Speed of light in km/s/Mpc

    def w(self):
        """
        Calculate the DE equation of state w(z).

        Returns:
        - Array of w(z) values for all the z in PC z range
        """
        weighted_eigenvectors = self.eigenvectors * self.alphas[:, np.newaxis]
        summed_vector = -1 + np.sum(weighted_eigenvectors, axis=0)
        return summed_vector

    def aux(self, z1, z2, w1):
        """
        Compute the auxiliary function used in the omegade calculation.

        Parameters:
        - z1: Starting redshift
        - z2: Ending redshift
        - w1: Weighting factor

        Returns:
        - Computed auxiliary value
        """
        return ((1 + z2) / (1 + z1)) ** (3 * (1 + w1))

    def omegade(self, zbins):
        """
        Compute the dark energy density function over redshift bins.

        Parameters:
        - zbins: Array of redshift bins

        Returns:
        - Dark energy density as a numpy array
        """
        w_values = self.w()
        N = len(w_values)
        if len(zbins) != N + 1:
            raise ValueError("Number of zbins is not correct")

        cumulative_product = []
        current_product = 1

        for i in range(N):
            current_product *= self.aux(zbins[i], zbins[i + 1], w_values[i])
            cumulative_product.append(current_product)

        return np.array(cumulative_product)


    def hubble(self, zbins, zBinsFisher):
        """
        Compute the Hubble parameter as a function of redshift.

        Parameters:
        - zbins: Array of redshift values for the edges of the bins
        - zBinsFisher: Array of redshift bins for Fisher matrix

        Returns:
        - Hubble parameter as a numpy array
        """
        
        h = self.H0 / 100
        Omh2 = self.Om * h ** 2
        ORhh = 4.180954748273937e-05 
        x = h ** 2 - Omh2 - self.Ok * h ** 2 - ORhh
        
        if x < 0:
            raise ValueError("Error: Bad Input, Om, Ok")
        else:
            return np.array( 100 * np.sqrt( Omh2 * ( 1 + zBinsFisher) ** 3 + x * self.omegade(zbins) + self.Ok * h**2 * (1 + zBinsFisher)**2) + ORhh * (1 + zBinsFisher)**4 )  

    def comov_dist(self, zbins, zBinsFisher):
        """
        Compute the comoving distance based on redshift bins.

        Parameters:
        - zbins: Array of redshift values for the edges of the bins 
        - zBinsFisher: Array of redshift bins for Fisher matrix

        Returns:
        - Comoving distance as a numpy array
        """
        
        h = self.H0 / 100
        Omh2 = self.Om * h ** 2
        ORhh = 4.180954748273937e-05 
        x = h**2 - Omh2 - self.Ok * h**2 - ORhh
        
        if x < 0:
            raise ValueError("Comoving dist: Bad Input, Om, Ok")
        else:
            def f(x1):
                return 1 / (100 * np.sqrt(Omh2 * (1 + x1) ** 3 + x * (1 + x1) + self.Ok * h**2 * (1 + x1) ** 2 + ORhh * (1 + x1)**4))

            less_zmin, _ = integrate.quad(f, 0, zbins[0])
            dz = zbins[1] - zbins[0]

            return self.cmag * less_zmin + self.cmag * np.cumsum(dz / self.hubble(zbins, zBinsFisher))

    def lum_distance(self, zbins, zBinsFisher):
        """
        Compute the luminosity distance based on redshift bins.

        Parameters:
        - zbins: Array of redshift values for the edges of the bins 
        - zBinsFisher: Array of redshift bins for Fisher matrix

        Returns:
        - Luminosity distance as a numpy array
        """
        z_shifted = 1 + zbins[:-1]
        return z_shifted * self.comov_dist(zbins, zBinsFisher)

    def logH0lum_distance(self, zbins, zBinsFisher):
        return np.log10( (self.H0 / self.cmag) * self.lum_distance(zbins , zBinsFisher))


        
