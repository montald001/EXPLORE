from astropy.units.quantity import Quantity
import matplotlib.pyplot as plt

from SIDM import SIDM
from imripy.merger_system import solar_mass_to_pc
from imripy.halo import NFW as NFW_niklas


class SIDM_natural_units():
    """
    A class to represent the self-interacting DM profile in natural units.
​
    ...
​
    Attributes
    ----------
​
​
    Methods
    -------
    density(r):
        returns a float of the isothermal or NFW density depending on r in natural units.
    mass(r):
        returns a float of the isothermal or NFW mass depending on r in natural units.
    """

    def __init__(self, rho_s_nat: float, r_s_nat: float, sigma_over_m: Quantity, BH_mass_nat: float = None, r_min_nat: float = 1e-1, r_max_nat: float = 1e5):
        rho_s = Quantity(rho_s_nat / solar_mass_to_pc, 'M_sun/pc3')
        r_s = Quantity(r_s_nat, 'pc')
        sigma_over_m = sigma_over_m
        BH_mass = Quantity(BH_mass_nat / solar_mass_to_pc, 'M_sun')
        r_min = Quantity(r_min_nat, 'pc')
        r_max = Quantity(r_max_nat, 'pc')
        self.profile = SIDM(rho_s, r_s, sigma_over_m, BH_mass, min(
            r_min, Quantity(1e-4, 'kpc')), max(r_max, Quantity(1e2, 'kpc')))
        self.r_range = self.profile.r_range
        self.r_1 = self.profile.r_1
        self.sigma_0 = self.profile.sigma_0
        self.r_spike = self.profile.r_spike

    def density(self, r):
        density = self.profile.density(Quantity(r, 'pc').to('kpc'))
        return density.to('M_sun/pc3').value * solar_mass_to_pc

    def mass(self, r):
        mass = self.profile.mass(Quantity(r, 'pc').to('kpc'))
        return mass.to('M_sun').value * solar_mass_to_pc


if __name__ == "__main__":
    sigma_over_m = Quantity(0.1, 'km2/g')
    bh_mass = 1e3 * solar_mass_to_pc
    dm_mass = 1e6 * solar_mass_to_pc
    rho_s = NFW_niklas.FromHaloMass(dm_mass, 20.).rho_s
    r_s = NFW_niklas.FromHaloMass(dm_mass, 20.).r_s

    profile = SIDM_natural_units(rho_s, r_s, sigma_over_m, bh_mass)
    print(
        f'r_1 = {profile.r_1}, r_spike = {profile.r_spike}, sigma_0 = {profile.sigma_0}')
    plt.loglog(profile.r_range, profile.density(profile.r_range))
    plt.show()
    plt.loglog(profile.r_range, profile.mass(profile.r_range))
    plt.show()
