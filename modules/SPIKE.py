import numpy as np
import matplotlib.pyplot as plt
from astropy.units.quantity import Quantity


class Spike():

    def __init__(self, rho_spike: Quantity, r_spike: Quantity, gamma: float):
        self.r_spike = r_spike
        self.gamma = gamma
        self.rho_spike = rho_spike

    def density(self, r):
        return self.rho_spike * (self.r_spike / r)**self.gamma

    def mass(self, r):
        return 4*np.pi*self.rho_spike*self.r_spike**self.gamma * r**(3.-self.gamma) / (3.-self.gamma)


if __name__ == '__main__':
    density_spike = Spike(4., 8., 7./3.)
    r_range = np.logspace(-4, 2, 100)
    plt.loglog(r_range, density_spike.density(r_range))
    plt.show()
