"""An implementation of the concentration fitting formula as a function of redshift and mass
from Prada 2012, 1104.5130. This is too big for large halos. Note also that Ludlow 2013 seems to imply
that it breaks at z > 0.
For this reason implement also the Ludlow 2016, 1601.02624 concentration mass relation.
"""

import math
import numpy as np

#Utility functions
class PradaConcentration(object):
    """Class to collect the functions for the Prada 2012, 1104.5130, concentration."""
    def __init__(self, omega_matter):
        self.omega_matter = omega_matter
        self.omega_lambda = 1-omega_matter

    def cmin(self, x):
        """Minimum of halo concentration, eq. 19 & 21."""
        c0 = 3.681
        c1 = 5.033
        alpha = 6.948
        x0 = 0.424
        return c0 + (c1-c0) * (np.arctan(alpha * (x-x0))/math.pi + 0.5)

    def sigmamin(self, x):
        """Minimum of inverse sigma, eq. 20 & 22."""
        sigma0 = 1.047
        sigma1 = 1.646
        localbeta = 7.386
        x1 = 0.526
        return sigma0 + (sigma1 - sigma0) * (np.arctan(localbeta * (x-x1))/math.pi + 0.5)

    def Bzero(self, x):
        """B_0(x): the scaling of the concentration with redshift. Eq. 18.
        Defined to be 1 at z=0"""
        return self.cmin(x)/self.cmin(1.393)

    def Bone(self, x):
        """B_1(x): the scaling of sigma with redshift. Eq. 18.
        Defined to be 1 at z=0"""
        return self.sigmamin(x)/self.sigmamin(1.393)

    def curlyC(self, sigmap):
        """Eq. 16: the unscaled redshift zero concentration."""
        A = 2.881
        b = 1.257
        c = 1.022
        d = 0.06
        return A * ((sigmap / b)**c + 1) * np.exp(d/sigmap**2)

    def xxtime(self,zz):
        """Rescaled time variable as a function of redshift"""
        aa = 1./(1+zz)
        return (self.omega_lambda/ self.omega_matter)**(1./3) * aa

    def concentration(self, nu, zz):
        """Concentration parameter for NFW halo. Eq. 14. Mass in M_sun/h"""
        #Get time.
        x = self.xxtime(zz)
        #Fluctuation amplitude at this redshift
        sp = self.Bone(x) * 1.686/nu
        #Concentration
        conc = self.Bzero(x) * self.curlyC(sp)
        #For large halos at high redshift, the Prada concentration becomes very large.
        #This is clearly unphysical, and only happens because there are no halos
        #to fit to in that box. Impose a maximum so that the lack of halos always wins.
        if np.size(conc) > 1:
            conc[np.where(conc > 1000)] = 1000
        elif conc > 1000:
            conc = 1000
        return conc

class LudlowConcentration(object):
    """Class to compute the concentration from Ludlow 2016, 1601.02624"""
    def __init__(self, Dofz):
        self.Dofz = Dofz

    def conc0(self, zz):
        """Utility function from Ludlow 2015. Eq. C2."""
        return 3.395*(1+zz)**(-0.215)

    def beta(self, zz):
        """Utility function from Ludlow 2015. Eq. C3."""
        return 0.307*(1+zz)**(0.540)

    def gamma1(self, zz):
        """Utility function from Ludlow 2015. Eq. C4."""
        return 0.628*(1+zz)**(-0.047)

    def gamma2(self, zz):
        """Utility function from Ludlow 2015. Eq. C5"""
        return 0.317*(1+zz)**(-0.893)

    def nu0(self,zz):
        """nu0 parameter from Ludlow 2015."""
        a = 1./(1+zz)
        nu0 = (4.135 - 0.564/a - 0.21/a/a + 0.0557/a**3 - 0.00348/a**4)
        #Prevent from going negative at high redshift
        return np.max([nu0, 0.5])

    def concentration(self,nu,zz):
        """Concentration fitting formula for Ludlow 2016, 1601.02624, as a function of peak height. Appendix C"""
        nuu = nu/(self.nu0(zz)/self.Dofz(zz))
        conc = self.conc0(zz) * (nuu)**(-self.gamma1(zz))*(1+(nuu)**(1./self.beta(zz)))**(-self.beta(zz)*(self.gamma2(zz) - self.gamma1(zz)))
        assert np.all(np.isfinite(conc))
        return conc

    def comoving_concentration(self,nu,zz):
        """Concentration fitting formula for Ludlow 2016, 1601.02624, as a function of peak height. Appendix C.
        This assumes that nu is nu * D(z)"""
        nuu = nu/self.nu0(zz)
        conc = self.conc0(zz) * (nuu)**(-self.gamma1(zz))*(1+(nuu)**(1./self.beta(zz)))**(-self.beta(zz)*(self.gamma2(zz) - self.gamma1(zz)))
        return conc


class ConstantConcentration(object):
    """A constant concentration parameter. Appropriate if the halos just nucleated."""
    def __init__(self,conc=1.):
        self.conc = conc

    def concentration(self, nu, zz):
        """Constant concentration."""
        _ = zz
        return self.conc*np.ones_like(nu)
