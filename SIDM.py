from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.optimize import newton, brentq
from astropy.units.quantity import Quantity
from astropy.units import isclose
from modules.constants import G, BH_age
from modules.NFW import NFW
from modules.ISOTHERMAL import Isothermal
from modules.SPIKE import Spike


class SIDM:
    """
    A class to represent the self-interacting DM profile.

    ...

    Attributes
    ----------
    rho_s : float
    r_s : float
    sigma_over_m : float

    Methods
    -------
    density(r):
        returns a float of the isothermal or NFW density depending on r.
    mass(r):
        returns a float of the isothermal or NFW mass depending on r.
    """

    def __init__(self, rho_s: Quantity, r_s: Quantity, sigma_over_m: Quantity, BH_mass: Quantity = None, r_min: Quantity = Quantity(1e-4, 'kpc'), r_max: Quantity = Quantity(1e2, 'kpc')):
        # TODO generalize r_range
        self.r_range = Quantity(np.geomspace(r_min, r_max, 100), 'kpc')
        self.sigma_over_m = sigma_over_m
        self.nfw = NFW(rho_s, r_s)
        self.BH_mass = BH_mass

        self.sigma_0 = Quantity(newton(self.__sigma_0_optimization,
                                       Quantity(14., 'km/s').value), 'km/s')

        self.r_1 = self.__find_r1(self.sigma_0)
        self.isothermal = self.__matching_isothermal(self.r_1)
        self.rho_0 = self.isothermal.rho_0
        self.sigma_0 = self.isothermal.sigma_0

        if BH_mass is not None:
            self.__matching_spike()

        self.__create_final_interpolation()

    def __sigma_0_optimization(self, sigma_0: float):
        r_1 = self.__find_r1(Quantity(sigma_0, 'km/s'))
        isothermal = self.__matching_isothermal(r_1)

        return sigma_0 - isothermal.sigma_0.to('km/s').value

    def __find_r1(self, sigma_0):
        # Get rho_1 from rate equation
        rho_1 = np.sqrt(np.pi) / (self.sigma_over_m *
                                  4 * sigma_0 * BH_age)
        r_min = self.r_range.to('kpc').value[0]
        r_max = self.r_range.to('kpc').value[-1]
        r_1 = Quantity(brentq(lambda x: (self.nfw.density(
            Quantity(x, 'kpc')) - rho_1).value, r_min, r_max), 'kpc')
        return r_1

    def __find_root(self, x_1, x_2, y_1, y_2):
        return (y_1*x_2 - y_2*x_1)/(y_1 - y_2)

    def __calculate_roots(self, data_points_x, data_points_y):
        roots = []
        for i in range(len(data_points_y)-1):
            if (data_points_y[i] < 0 and data_points_y[i+1] > 0) or (data_points_y[i] > 0 and data_points_y[i+1] < 0):
                roots.append(self.__find_root(data_points_x[i],
                                              data_points_x[i+1],
                                              data_points_y[i],
                                              data_points_y[i+1]))

        return roots

    def __matching_isothermal(self, r_1):
        m_1 = self.nfw.mass(r_1)
        rho_1 = self.nfw.density(r_1)
        ratio_nfw = m_1/(4*np.pi*rho_1*r_1**3)

        # the starting parameters of the isothermal profile aren't important here,
        # because we are just interrested in h and j
        sigma_0 = Quantity(2, 'km/s')
        rho_0 = Quantity(2, 'M_sun/kpc3')
        isothermal = Isothermal(sigma_0, rho_0, self.r_range.min()
                                * 0.0000000001, self.r_range.max()*100000)
        h = isothermal.h_0
        j = isothermal.j_0

        def optimization():
            return np.exp(j - h)/3 - ratio_nfw

        def potential(profile):
            r_range = profile.r_range[profile.r_range <= r_1]
            integrand = - G / 2 * (profile.mass(r_range) ** 2 -
                                   self.nfw.mass(r_range) ** 2) / r_range**2

            return trapezoid(integrand, r_range)

        roots = self.__calculate_roots(
            isothermal.x_range, optimization())

        solutions = []
        # see if matching fails for certain roots:
        for x_1 in roots:
            rho_0 = self.nfw.density(r_1)*np.exp(-isothermal.h(x_1))
            if rho_0 == np.inf:
                continue
            r_stern = r_1/x_1
            sigma_0 = np.sqrt(4*np.pi*G*rho_0)*r_stern

            isothermal = Isothermal(
                sigma_0, rho_0, self.r_range.min(), self.r_range.max())

            # check if masses and densities are close enough within 1% tolerance
            if not ((isclose(isothermal.mass(r_1), self.nfw.mass(r_1),
                             rtol=1e-2) == True) and (isclose(isothermal.density(r_1),
                                                              self.nfw.density(r_1), rtol=1e-2) == True)):
                continue

            # check if core-collapse or core growing-solution
            delta_U = potential(isothermal)
            if delta_U > 0.:
                solutions.append(isothermal)

        # handle multiple or no matching solutions
        if len(solutions) > 1:
            raise RuntimeWarning(
                "Multiple solutions found, picking first one!")
        elif len(solutions) == 0:
            raise RuntimeError("No working solution found!")

        isothermal = solutions[0]

        return isothermal

    def __matching_spike(self):
        self.r_spike = G*self.BH_mass/self.sigma_0**2
        if self.r_spike > self.r_1:
            raise RuntimeError("The spike radius is larger than r_1")
        self.spike = Spike(self.isothermal.density(
            self.r_spike), self.r_spike, 7/4)

    def __create_final_interpolation(self):
        density = []
        mass = []
        for r in self.r_range:
            if r < self.r_1:
                if self.r_spike is not None and r < self.r_spike:
                    density.append(self.spike.density(r))
                    mass.append(self.spike.mass(r))
                else:
                    density.append(self.isothermal.density(r))
                    if self.r_spike is not None:
                        mass.append(self.isothermal.mass(r) +
                                    self.spike.mass(
                                        self.r_spike) - self.isothermal.mass(self.r_spike))
                    else:
                        mass.append(self.isothermal.mass(r))
            else:
                density.append(self.nfw.density(r))
                if self.r_spike is not None:
                    mass.append(self.nfw.mass(r) +
                                self.spike.mass(self.r_spike) -
                                self.nfw.mass(self.r_spike))
                else:
                    mass.append(self.nfw.mass(r))

        density = Quantity(density)
        mass = Quantity(mass)
        self.__density_interpolate = lambda x: Quantity(interp1d(
            self.r_range, density, fill_value="extrapolate")(x), density.unit)
        self.__mass_interpolate = lambda x: Quantity(interp1d(
            self.r_range, mass, fill_value="extrapolate")(x), mass.unit)

    def density(self, r: float):
        return self.__density_interpolate(r)

    def mass(self, r: float):
        return self.__mass_interpolate(r)


if __name__ == "__main__":

    # sigma_over_m = Quantity(11., 'km2/g')
    # sigma_over_m = Quantity(15.8, 'km2/g')
    # sigma_over_m = Quantity(1., 'km2/g')
    sigma_over_m = Quantity(14, 'km2/g')
    rho_s = Quantity(0.019e9, 'M_sun/kpc3')
    r_s = Quantity(2.59, 'kpc')
    BH_Mass = Quantity(1e5, 'M_sun')
    profile = SIDM(rho_s, r_s, sigma_over_m, BH_Mass)
    print(
        f'r_1 = {profile.r_1}, r_spike = {profile.r_spike}, sigma_0 = {profile.sigma_0}')
    plt.loglog(profile.r_range, profile.density(profile.r_range))
    plt.axvline(profile.r_1.value)
    plt.axvline(profile.r_spike.value)
    plt.show()
    plt.loglog(profile.r_range, profile.mass(profile.r_range))
    plt.show()
