"""Do a Burkhart profile as well as an NFW one.
This assumes the mass concentration relation is the same.
This should be ok as the profiles only differ in the innermost regions,
and the velocity structure is unchanged."""
import math
import numpy as np
import scipy.special
import pbhmergers

class BurkHalo(pbhmergers.NFWHalo):
    """Burkhert halo. Profile is Rho[r_, r0_] := 1/(r + r0)/(r^2 + r0^2)"""

    def pbhpbhrate(self, mass):
        """The merger rate for primordial black holes (per year) in a halo of mass in Msun/h, computed in the attached pdf."""
        #Virial radius R200 in Mpc/h from the virial mass
        R200 = self.R200(mass)
        conc = self.concentration(mass)
        R0 = R200/conc
        crosssec = self.cross_section(mass)
        rho0 = mass / (4*math.pi) / (-0.5*np.arctan(conc) + 0.25 * np.log((conc**2+1)*(conc+1)**2))/R0**3
        burkd2 = -2*math.pi*0.25 * (-2 + 1/(1+conc) + 1/(1+conc**2) + np.arctan(conc))*rho0**2*R0**3
        rate = 4 * math.pi * crosssec * burkd2
        return rate

    def profile(self, rr, mass):
        R200 = self.R200(mass)
        conc = self.concentration(mass)
        R0 = R200/conc
        rred = rr/R0
        rho0 = mass / (4*math.pi) / (-0.5*np.arctan(conc) + 0.25 * np.log((conc**2+1)*(conc+1)**2))/R0**3
        return rho0 / (1+ rred)/(1+rred**2)

    def vel_disp(self, mass):
        """The 1D velocity dispersion of a halo in m/s, as a function of the virial radius. Equal to v_max/sqrt(2)"""
        v1d = self.virialvel(mass)/math.sqrt(2)
        return v1d

    def concentration(self, mass):
        try:
            return self.conc0
        except AttributeError:
            return 50
