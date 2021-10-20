import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp as solve
from scipy.interpolate import interp1d
from astropy.units.quantity import Quantity
from modules import constants

G = constants.G


class Isothermal:
    """
    A class to represent the isothermal profile.

    ...

    Attributes
    ----------
    sigma_0 : float
    rho_0 : float
    r_range : np.ndarray

    Methods
    -------
    density_interp(x):
        returns a float of the interpolated density at radius x
    mass(x):
        returns a float of the interpolated mass at radius x
    h_interp(x):
        returns a float of the interpolated h_0 at radius x
    j_interp(x):
        returns a float of the interpolated j_0 at radius x
    r_range:
        returns the r range at which the jeans equations are evaluated
    x_range:
        returns the x range at which the jeans equations are evaluated
    """

    def __init__(self, sigma_0: Quantity, rho_0: Quantity, r_min: Quantity, r_max: Quantity):
        """
        Constructs all the necessary attributes for the Isothermal object.

        Parameters
        ----------
        sigma_0 : float
        rho_0 : float
        r_range : np.ndarray
            0.0 should not be included, because this gives a division by zero error.
        """
        self.r_range = np.geomspace(r_min, r_max, 1000)
        self.rho_0 = rho_0
        self.sigma_0 = Quantity(sigma_0, 'km/s')
        self.__r_stern = sigma_0 / np.sqrt(4 * np.pi * G * rho_0)
        self.r_range = self.r_range
        self.x_range = self.r_range/self.__r_stern
        self.x_min = self.r_range[0] / self.__r_stern
        self.x_max = self.r_range[-1] / self.__r_stern

        self.__solve_jeans_equations()

        self.__density_interpolate = lambda x: Quantity(interp1d(
            self.r_range, self.__density(), fill_value="extrapolate")(x), self.__density().unit)
        self.__mass_interpolate = lambda x: Quantity(interp1d(
            self.r_range, self.__mass(), fill_value="extrapolate")(x), self.__mass().unit)
        self.__h_0_interpolate = lambda x: Quantity(interp1d(
            self.x_range, self.h_0, fill_value="extrapolate")(x), self.h_0.unit)
        self.__j_0_interpolate = lambda x: Quantity(interp1d(
            self.x_range, self.j_0, fill_value="extrapolate")(x), self.j_0.unit)

    def __solve_jeans_equations(self):
        h_0_x_start = Quantity(0, '')
        h_1_x_start = - self.x_min / 3

        result = solve(self.__f, (self.x_min, self.x_max),
                       [h_0_x_start, h_1_x_start], t_eval=self.x_range)

        self.h_0 = Quantity(result.y[0], h_0_x_start.unit)
        h_1 = Quantity(result.y[1], h_1_x_start.unit)
        self.x_range = Quantity(result.t, self.x_range.unit)
        self.r_range = self.x_range * self.__r_stern
        self.j_0 = np.log(-h_1 * 3 / self.x_range)

    def __f(self, x: float, y: float):
        return [y[1], -2/x * y[1] - np.exp(y[0])]

    def __density(self):
        return self.rho_0 * np.exp(self.h_0)

    def __mass(self):
        return 4 * np.pi / 3 * self.rho_0 * self.r_range**3 * np.exp(self.j_0)

    def density(self, x: float):
        return self.__density_interpolate(x)

    def mass(self, x: float):
        return self.__mass_interpolate(x)

    def h(self, x: float):
        return self.__h_0_interpolate(x)

    def j(self, x: float):
        return self.__j_0_interpolate(x)


if __name__ == "__main__":

    r_min = Quantity(1e-4, 'kpc')
    r_max = Quantity(1e2, 'kpc')
    sigma_0 = Quantity(13.88, 'km/s')
    rho_0 = Quantity(2.55*1e8, 'M_sun/kpc3')
    profile = Isothermal(sigma_0, rho_0, r_min, r_max)
    r = profile.r_range

    fig, ax = plt.subplots()
    ax.plot(r, profile.density(r), color="blue", label="density")
    ax2 = ax.twinx()
    ax2.plot(r, profile.mass(r), color="orange", label="mass")
    ax.set_xlabel(r'r [kpc]')
    ax2.set_ylabel(r'$M_{iso} [kpc^{-2}]$')
    ax.set_ylabel(r'$\rho_{iso}(r)$')
    ax.legend(loc="upper left")
    ax.set_yscale('log')
    ax2.set_yscale("log")
    ax.set_xscale('log')
    ax2.set_xscale("log")
    ax2.legend(loc="upper right")
    plt.show()
