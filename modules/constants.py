"""This file contains all constants used
    """

from astropy import constants as const
from astropy.units.quantity import Quantity

G = const.G.to("kpc km2/(M_sun s2)")

BH_age = Quantity(10, 'Gyr')
