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
        R0 = R200/conc*self.Mpc
        crosssec = self.cross_section(mass)
        rho0 = mass*self.solarmass / (4*math.pi) / (-0.5*np.arctan(conc) + 0.25 * np.log((conc**2+1)*(conc+1)**2))/R0**3
        assert np.all(rho0 > 0)
        burkd2 = -2*math.pi*0.25 * (-2 + 1/(1+conc) + 1/(1+conc**2) + np.arctan(conc))*rho0**2*R0**3
        assert np.all(burkd2 > 0)
        rate = 4 * math.pi * crosssec * burkd2
        return rate*self.secperyr

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


class EinastoHalo(pbhmergers.NFWHalo):
    """Einasto profile with alpha = 0.18"""

    def pbhpbhrate(self, mass):
        """The merger rate for primordial black holes (per year) in a halo of mass in Msun/h, computed in the attached pdf."""
        conc = self.concentration(mass)
        alpha = 0.18
        crosssec = self.cross_section(mass)
        rho0 = self.rho0(mass) * self.solarmass/self.Mpc**3 * self.overden.hubble0**2
        d2 = np.exp(4/alpha) * (self.Rs(mass)*self.Mpc/self.overden.hubble0)**3 /alpha * (alpha/4)**(3/alpha) * scipy.special.gammainc(3/alpha, 4/alpha * conc**alpha) * scipy.special.gamma(3/alpha)
        rate = 2 * math.pi* crosssec * d2 * rho0**2
        return rate*self.secperyr

    def rho0(self, mass):
        """Central density for the Einasto profile in M_sun/Mpc^3 h^2"""
        alpha = 0.18
        R200 = self.R200(mass)
        conc = self.concentration(mass)
        gamma = scipy.special.gammainc(3/alpha, 2/alpha * conc**alpha) * scipy.special.gamma(3/alpha)
        prefac = 4 * math.pi * np.exp(2/alpha)/ alpha *(alpha/2)**(3/alpha)
        return mass / gamma / prefac / (R200 / conc)**3

    def profile(self, rr, mass):
        R200 = self.R200(mass)
        conc = self.concentration(mass)
        Rs = R200/conc
        alpha = 0.18
        rho0 = self.rho0(mass)
        rho = rho0 * np.exp(-2 / alpha * ((rr/Rs)**alpha -1))
        return rho
