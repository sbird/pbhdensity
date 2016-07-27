"""Class to extend the HaloMassFunction class to compute the estimated merger rate of primordial black holes in different sized halos."""
import math
import numpy as np
import scipy.special
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import concentration
import halo_mass_function as hm
from save_figure import save_figure

def ggconc(conc):
    """Utility function that drops out of the NFW profile. Eq. 10 of the attached pdf."""
    return np.log(1+conc)-conc/(1+conc)

class NFWHalo(hm.HaloMassFunction):
    """Class to add the ability to compute concentrations to the halo mass function"""
    def __init__(self,*args,conc_model="ludlow", conc_value=1., **kwargs):
        #in kg
        self.solarmass = 1.98855e30
        #1 Mpc in m
        self.Mpc = 3.086e22
        #speed of light in m/s
        self.light = 2.99e8
        #Newtons constant in units of m^3 kg^-1 s^-2
        self.gravity = 6.67408e-11
        #Number of seconds in a year
        self.secperyr = 60*60*24*365
        #Factor of R_s at which the maximum circular velocity of the halo is reached.
        self.dmax = 2.1626
        super().__init__(*args, **kwargs)
        if conc_model == "ludlow":
            self.conc_model = concentration.LudlowConcentration(self.overden.Dofz)
        elif conc_model == "prada":
            self.conc_model = concentration.PradaConcentration(self.overden.omega_matter0)
        else:
            self.conc_model = concentration.ConstantConcentration(conc_value)

    def concentration(self,mass):
        """Compute the concentration for a halo mass in Msun"""
        nu = 1.686/self.overden.sigmaof_M_z(mass*self.overden.hubble0)
        zz = self.overden.redshift
        return self.conc_model.concentration(nu, zz)

    def rhocrit(self):
        """Critical density at redshift of the snapshot. Units are kg m^-3."""
        #Newtons constant in units of m^3 kg^-1 s^-2
        #scale factor
        aa = 1./(1+self.overden.redshift)
        #Hubble factor (~70km/s/Mpc) at z=0 in s^-1
        hubble = self.overden.hubble0*3.24077929e-18
        hubz2 = (self.overden.omega_matter0/aa**3 + self.overden.omega_lambda0) * hubble**2
        #Critical density at redshift in units of kg m^-3
        rhocrit = 3 * hubz2 / (8*math.pi* self.gravity)
        return rhocrit

    def R200(self, mass):
        """Get the virial radius in Mpc for a given mass in Msun"""
        rhoc = self.rhocrit()
        #Virial radius R200 in Mpc from the virial mass
        R200 = np.cbrt(3 * mass * self.solarmass / (4* math.pi* 200 * rhoc))/self.Mpc
        return R200

    def Rs(self, mass):
        """Scale radius of the halo in Mpc"""
        conc = self.concentration(mass)
        return self.R200(mass)/conc

    def virialvel(self, mass):
        """Get the virial velocity in m/s for mass in Msun"""
        #The factors of h cancel
        return np.sqrt(2*self.gravity*self.solarmass*mass/(self.R200(mass)*self.Mpc))

    def Rmax(self, mass):
        """The radius at which the maximum circular velocity of a halo is reached"""
        return self.dmax*self.Rs(mass)

    def vel_disp(self, mass):
        """The 1D velocity dispersion of a halo in m/s, as a function of the virial radius. Equal to v_max/sqrt(2)"""
        conc = self.concentration(mass)
        v1d = self.virialvel(mass)/math.sqrt(2)*np.sqrt(conc/self.dmax*ggconc(self.dmax)/ggconc(conc))
        return v1d

    def cross_section(self, mass):
        """The PBH merger cross-section for a halo as a function of halo mass. Eq. 11 of PDF.
        Assumes that the relative velocities are distributed like a Maxwell-Boltzmann with a temperature
        of the velocity dispersion and a maximum value of the virial velocity.
        Since MPBH drops out, set it to one here.
        Returns cross-section in m^3/s kg^-2"""
        prefac = (4*math.pi)**2*(85*math.pi/3)**(2./7)*self.gravity**2/self.light**(10/7.)
        sigma = self.vel_disp(mass)
        vvir = self.virialvel(mass)
        #Now we have a mathematica integral in terms of gamma functions.
        #P[v_, sigma_, vvir_] := Exp[-v^2/sigma^2] - Exp[-vvir^2/sigma^2]
        #FunctionExpand[Integrate[v^(3/7)*P[v, sigma, Vvir], {v, 0, Vvir}]]
        #Piece from the constant exponential cutoff
        cutoff = -(7/10)*np.exp(-(vvir**2/sigma**2)) * vvir**(10/7)
        #Piece from the gamma integral: note that mathematica's incomplete gamma function
        #is not quite the same as scipy's: scipy is (Gamma[a] - Gamma[a,z])/Gamma[a]
        gammaint = sigma**(10/7)*scipy.special.gammainc(5/7,vvir**2/sigma**2)* scipy.special.gamma(5/7)/2
        #We also need to normalise the probability function for v:
        #Integrate[4*Pi*v^2*P[v, sigma, Vvir], {v, 0, Vvir}]
        probnorm = math.pi**(3/2)*sigma**3*scipy.special.erf(vvir/sigma) - 2*math.pi/3*np.exp(-(vvir**2/sigma**2))*(3*sigma**2*vvir + 2*vvir**3)
        #Once the normalisation passes through zero, we probably have roundoff
        #and we should try again with a long double.
        if not np.all(probnorm) > 0:
            sigma = sigma.astype(np.float128)
            vvir = vvir.astype(np.float128)
            #Some crazy casting here because erf doesn't have a long double version.
            probnorm = math.pi**(3/2)*sigma**3*scipy.special.erf((vvir/sigma).astype(np.float64)).astype(np.float128) - 2*math.pi/3*np.exp(-(vvir**2/sigma**2))*(3*sigma**2*vvir + 2*vvir**3)
        assert np.all(probnorm > 0)
        return prefac*(gammaint + cutoff)/probnorm

    def profile(self, radius, mass):
        """The NFW profile at a given radius and mass."""
        #scale radius
        R_s = self.Rs(mass)
        #NFW profile
        rho_0 = self.rho0(mass)
        density = rho_0 / (radius/R_s * (1+ radius/R_s)**2)
        return density

    def pbhpbhrate(self, mass):
        """The merger rate for primordial black holes (per year) in a halo of mass in Msun, computed in the attached pdf."""
        conc = self.concentration(mass)
        crosssec = self.cross_section(mass)
        #In m
        Rs = self.Rs(mass)*self.Mpc
        rho0 = self.rho0(mass) * self.solarmass / self.Mpc**3
        rate = crosssec * 2 * math.pi * rho0**2 * Rs**3 /3 * (1 - 1/(1+conc)**3)
        return rate*self.secperyr

    def rho0(self, mass):
        """Central density for the NFW halo in units of M_sun Mpc^-3"""
        conc = self.concentration(mass)
        return mass / ( 4 * math.pi * self.Rs(mass)**3 * ggconc(conc))

    def mergerpervolume(self, lowermass=5e2, uppermass=1e16):
        """The merger rate for primordial black holes in events per Gpc per yr."""
        #See notes for these limits
        #mass has units M_sun/h
        mass = np.logspace(np.log10(lowermass),np.log10(uppermass),1000)
        #pbhrate is 1/s
        pbhrate = self.pbhpbhrate(mass)
        #dndm has units: h^4 M_sun^-1 Mpc^-3
        dndm = self.dndm(mass*self.overden.hubble0)*self.overden.hubble0**4
        assert np.all(dndm >= 0)
        #So result is (h/Mpc)^3 /s
        mergerrate = np.trapz(dndm * pbhrate*mass,np.log(mass))
        GpcperMpch = 1e3**3
        return mergerrate*GpcperMpch

    def mergerfraction(self, vvir, time=6., bhmass = 30):
        """Compute the fraction of black hole binaries which merge within time,
        following O'Leary+2008 Eq. 27. for a halo of virial velocity vvir.
        Time is in Gyr.
        bhmass is in M_sun and is the mass of the merging objects."""
        #Convert time to s
        timesec = time * self.secperyr * 1e9
        c1 = 3* math.sqrt(3)/(170*math.sqrt(85* math.pi))
        #c2 = (340*math.pi/3)**(1/7.)
        prefac = (self.light**3 * timesec / (c1 * self.gravity * 2 * bhmass * self.solarmass))**(2/21.)
        effcs = prefac * (vvir/self.light)**(2/7.)
        #A quick way to set a maximum that works for both single numbers and arrays
#         effcs *= ((effcs > 1)/effcs + (effcs < 1))
        return effcs**2

    def halomergerratepervolume(self, mass):
        """The merger rate per year per unit volume for halos in a mass bin."""
        GpcperMpch = 1e3**3
        return self.overden.hubble0**4*self.dndm(mass*self.overden.hubble0)*self.pbhpbhrate(mass)*mass*GpcperMpch

    def evaptime(self,mass, bhmass=30):
        """The evaporation timescale following Binney and Tremaine."""
        vv = self.vel_disp(mass)
        Rs = self.Rs(mass)
        return 14 * mass/ bhmass / np.log(mass/bhmass) * Rs *self.Mpc/0.7 / vv / self.secperyr

    def threebodyratio(self,mass):
        """The ratio between three body and two body binary formation rate.
        This becomes large in small halos."""
        vel = self.vel_disp(mass)
        return 18*(mass/30.)**(-2)*(vel/self.light)**(-10./7)

    def mergerhalflife(self,mass,threefac=True, bhmass=30.):
        """The timescale for 50% of the mass of the halo to have merged."""
        rate = self.pbhpbhrate(mass)
        if threefac:
            threefac = self.threebodyratio(mass)
            threefac = np.max([threefac, np.ones_like(threefac)],axis=0)
            rate *= threefac
        return 0.5*(mass/bhmass)/rate

    def bias(self,mass):
        """The formula for halo bias in EPS theory (Mo & White 1996), eq. 13"""
        delta_c = 1.686
        nu = delta_c/self.overden.sigmaof_M_z(mass*self.overden.hubble0)
        bhalo = (1 + (nu**2 -1)/ delta_c)
        return bhalo

class EinastoHalo(NFWHalo):
    """Einasto profile with alpha = 0.18"""
    def pbhpbhrate(self, mass):
        """The merger rate for primordial black holes (per year) in a halo of mass in Msun/h, computed in the attached pdf."""
        #Virial radius R200 in Mpc/h from the virial mass
        conc = self.concentration(mass)
        alpha = 0.18
        crosssec = self.cross_section(mass)
        rho0 = self.rho0(mass) * self.solarmass/self.Mpc**3
        d2 = np.exp(4/alpha) * (self.Rs(mass)*self.Mpc)**3 /alpha * (alpha/4)**(3/alpha) * scipy.special.gammainc(3/alpha, 4/alpha * conc**alpha) * scipy.special.gamma(3/alpha)
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

def plot_pbh_halo(redshift):
    """Plot the PBH merger rate as a function of halo mass."""
    mass = np.logspace(2,15)
    hh = NFWHalo(redshift)
    hh.conc_model = concentration.LudlowConcentration(hh.overden.Dofz)
    pbhrate = hh.pbhpbhrate(mass)
    hh.conc_model = concentration.PradaConcentration(hh.overden.omega_matter0)
    pbhrate_prada = hh.pbhpbhrate(mass)
    plt.loglog(mass, pbhrate, ls='-', label="Ludlow")
    plt.loglog(mass, pbhrate_prada, ls='--', label="Prada")
    plt.loglog(mass, hh.pbhpbhrate(1e9)*mass/1e9, ls='-.', color="black")
    plt.xlabel(r"$M_\mathrm{vir}$ ($M_\odot/h$)")
    plt.ylabel(r"Merger rate per halo (yr$^{-1}$)")
    plt.xlim(300,1e15)
    plt.ylim(1e-15,1e-10)
    plt.xticks(np.logspace(3,15,5))
    plt.legend(loc=0)
    save_figure("halomergerrate")
    plt.clf()

def plot_pbh_per_mass(redshift):
    """Plot the PBH merger rate per unit volume as a function of halo mass"""
    mass = np.logspace(2,15)
    hh = NFWHalo(redshift,conc_model="ludlow")
    GpcperMpch = 1e3**3*hh.overden.hubble0**3
    hh.conc_model = concentration.LudlowConcentration(hh.overden.Dofz)
    plt.loglog(mass,  hh.halomergerratepervolume(mass), ls='-', label="Ludlow concentration")
    hh.conc_model = concentration.PradaConcentration(hh.overden.omega_matter0)
    plt.loglog(mass, hh.halomergerratepervolume(mass), ls='--', label="Prada concentration")
    hh.conc_model = concentration.LudlowConcentration(hh.overden.Dofz)
    hh.mass_function = hh.press_schechter
    plt.loglog(mass, hh.halomergerratepervolume(mass), ls=':', label="Press-Schechter m.f.")
    hh.mass_function = hh.jenkins
    plt.loglog(mass, hh.halomergerratepervolume(mass), ls='-.', label="Jenkins m.f.")
    plt.xlim(500,1e15)
    plt.xticks(np.logspace(3,15,5))
    plt.xlabel(r"$M_\mathrm{vir}$ ($M_\odot/h$)")
    plt.ylabel(r"Merger rate (yr$^{-1}$ Gpc$^{-3}$)")
    plt.ylim(1e-8, 1)
    #plt.title("Total mergers is area under this curve")
    plt.legend(loc=0)
    save_figure("volumemergerrate")
    plt.clf()

def plot_concentration_vs_mass(redshift):
    """Plot the concentration as a function of halo mass"""
    mass = np.logspace(2,16)
    hh = NFWHalo(redshift)
    plt.loglog(mass, hh.concentration(mass), ls='-', label="Ludlow concentration")
    hh.conc_model = concentration.PradaConcentration(hh.overden.omega_matter0)
    plt.loglog(mass, hh.concentration(mass), ls='--', label="Prada concentration")
    plt.xlim(500,1e16)
    plt.xticks(np.logspace(3,15,5))
    plt.xlabel(r"$M_\mathrm{vir}$ ($M_\odot/h$)")
    plt.ylabel(r"Concentration")
    plt.ylim(1e-8, 1)
    plt.legend(loc=0)
    save_figure("concentration")
    plt.clf()

def _print_numbers(hh):
    """Print numbers for the latex writeup"""
    print("Mergers in MW:",hh.pbhpbhrate(1e12),"/yr")
    print("Mergers per Gpc:",hh.mergerpervolume(400)," above 10^9: ",hh.mergerpervolume(1e9))
    #z02vol = 2.336
    #print("Mergers in six months in z=0.2, lower limit:",z02vol*hh.mergerpervolume(500*0.7)/2,"above 10^9: ",z02vol*hh.mergerpervolume(1e9)/2)
    #Press-Schechter
    hh.mass_function = hh.press_schechter
    print("Press-Schechter Mergers per Gpc:",hh.mergerpervolume(400)," above 10^9: ",hh.mergerpervolume(1e9))
    hh.mass_function = hh.jenkins
    print("Jenkins Mergers per Gpc:",hh.mergerpervolume(400)," above 10^9: ",hh.mergerpervolume(1e9))

def merger_at_z(z, conc="Ludlow", halo="Einasto"):
    """Compute the merger rate at a given redshift"""
    if halo=="Einasto":
        hh = EinastoHalo(z)
    else:
        hh = NFWHalo(z)
    if conc=="Ludlow":
        hh.conc_model = concentration.LudlowConcentration(hh.overden.Dofz)
    else:
        hh.conc_model = concentration.PradaConcentration(hh.overden.omega_matter0)
    merg = hh.mergerpervolume(400)
    #Note that this is Gpc/yr in
    #the rest frame of the event.
    #For the rest frame of the observer,
    #you need to account for time dilation.
    return merg / (1+z)

def rate_over_redshift(zmin=0., zmax=20., nred = 100,conc="Ludlow", halo="Einasto"):
    """Compute the merger rate over a wide redshift range."""
    zzs = 1/np.linspace(1/(1+zmax), 1/(1+zmin),nred) -1.
    mergers = np.array([merger_at_z(zz,conc=conc, halo=halo) for zz in zzs])
    return zzs, mergers

def redshift_tables():
    """Print tables of the redshift evolution of the mergers"""
    zzs, mergers = rate_over_redshift(conc="Ludlow", halo="Einasto")
    np.savetxt("ludlow_einasto.txt", np.array((zzs,mergers)).T)
    zzs, mergers = rate_over_redshift(conc="Ludlow", halo="NFW")
    np.savetxt("ludlow_nfw.txt", np.array((zzs,mergers)).T)
    zzs, mergers = rate_over_redshift(conc="Prada", halo="Einasto")
    np.savetxt("prada_einasto.txt", np.array((zzs,mergers)).T)
    zzs, mergers = rate_over_redshift(conc="Prada", halo="NFW")
    np.savetxt("prada_nfw.txt", np.array((zzs,mergers)).T)

def print_numbers():
    """Wrapper to print for both halos"""
    print("NFW halo:")
    hh = NFWHalo(0)
    print("Ludlow:")
    _print_numbers(hh)
    print("Prada:")
    hhp = NFWHalo(0,conc_model="prada")
    _print_numbers(hhp)
    print("Einasto halo:")
    hh2 = EinastoHalo(0)
    _print_numbers(hh2)
    print("Einasto halo Prada:")
    hh2p = EinastoHalo(0,conc_model="prada")
    _print_numbers(hh2p)


#zz = np.linspace(0,3,15)
#
#class biasofz(object):
#     def __init__(self, zz):
#         self.objs = [pbhmergers.NFWHalo(z) for z in zz]
#     def bias(self,mass):
#         return [obj.bias(mass) for obj in self.objs]
#
#bbb = biasofz(zz)
#plt.plot(zz, bbb.bias(1e6), label="10^6")

if __name__ == "__main__":
    plot_concentration_vs_mass(0)
    plot_pbh_halo(0)
    plot_pbh_per_mass(0)
    print_numbers()
