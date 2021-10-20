import matplotlib.pyplot as plt
import numpy as np
from astropy.units.quantity import Quantity


class NFW:
    """
    A class to represent the NFW profile.

    ...

    Attributes
    ----------
    rho_0 : float
    r_s : float
    r_range : np.ndarray

    Methods
    -------
    density():
        returns an array of the mass density
    mass():
        returns an array of the mass
    """

    def __init__(self, rho_s: Quantity, r_s: Quantity):
        """
        Constructs all the necessary attributes for the Isothermal object.

        Parameters
        ----------
        rho_0 : float
        r_s : float
        """
        self.rho_s = rho_s
        self.r_s = r_s

    def __rho(self, r: float):
        return self.rho_s / (r / self.r_s * (1 + r/self.r_s)**2)

    def __M(self, r: float):
        return 4 * np.pi * self.rho_s * self.r_s**3 * (np.log((self.r_s + r) / self.r_s) + self.r_s / (self.r_s + r) - 1)

    def density(self, r: float):
        return self.__rho(r)

    def mass(self, r: float):
        return self.__M(r)


if __name__ == "__main__":

    r_min = Quantity(1e-4, 'kpc')
    r_max = Quantity(1e2, 'kpc')
    rho_s = Quantity(2.55*1e8, 'M_sun/kpc3')
    r_s = Quantity(13.88, 'kpc')
    r = np.geomspace(r_min, r_max, 10000)
    profile = NFW(rho_s, r_s)

    fig, ax = plt.subplots()
    ax.plot(r, profile.density(r), color="blue", label="density")
    ax2 = ax.twinx()
    ax2.plot(r, profile.mass(r), color="orange", label="mass")
    ax.set_xlabel(r'r [kpc]')
    ax2.set_ylabel(r'$M_{iso} [kpc^{-2}]$')
    ax.set_ylabel(r'$\rho_{iso}(r)$')
    ax.legend(loc="upper left")
    ax.set_yscale('log')
    ax2.legend(loc="upper right")
    plt.show()
