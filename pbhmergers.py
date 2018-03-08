"""Class to extend the HaloMassFunction class to compute the estimated merger rate of primordial black holes in different sized halos."""
import math
import numpy as np
import scipy.special
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import pint
import concentration
import halo_mass_function as hm

#This is a global so that the decorator checking works.
ureg_chk=pint.UnitRegistry()

def ggconc(conc):
    """Utility function that drops out of the NFW profile. Eq. 10 of the attached pdf."""
    return np.log(1+conc)-conc/(1+conc)

class NFWHalo(hm.HaloMassFunction):
    """Class to add the ability to compute concentrations to the halo mass function"""
    def __init__(self,*args,conc_model="ludlow", conc_value=1.,hubble=0.67, **kwargs):
        self.ureg=pint.UnitRegistry()
        self.ureg.define("Msolar = 1.98855*10**30 * kilogram")
        #Mpc newton's constant and light speed are already defined.
        #Hubble constant and related objects!
        #The try except is because the method names change between pint 0.6 and pint 0.8
        try:
            self.ureg.define(pint.unit.UnitDefinition('hub', '', (),pint.unit.ScaleConverter(hubble)))
        except AttributeError:
            self.ureg.define(pint.definitions.UnitDefinition('hub', '', (),pint.converters.ScaleConverter(hubble)))
        self.ureg.define("Msolarh = Msolar / hub")
        self.ureg.define("Mpch = Mpc / hub")
        #Factor of R_s at which the maximum circular velocity of the halo is reached.
        self.dmax = 2.1626
        super().__init__(*args, **kwargs)
        if conc_model == "ludlow":
            self.conc_model = concentration.LudlowConcentration(self.overden.Dofz)
        elif conc_model == "prada":
            self.conc_model = concentration.PradaConcentration(self.overden.omega_matter0)
        else:
            self.conc_model = concentration.ConstantConcentration(conc_value)

    def get_nu(self,mass):
        """Get nu, delta_c/sigma"""
        return 1.686/self.overden.sigmaof_M_z(mass.to(self.ureg.Msolarh).magnitude)

    def concentration(self,mass):
        """Compute the concentration for a halo mass in Msun"""
#         assert self.ureg.get_dimensionality('[mass]') == self.ureg.get_dimensionality(mass)
        nu = self.get_nu(mass)
        zz = self.overden.redshift
        return self.conc_model.concentration(nu, zz)

    def rhocrit(self):
        """Critical density at redshift of the snapshot. Units are kg m^-3."""
        #Newtons constant in units of m^3 kg^-1 s^-2
        #scale factor
        aa = 1./(1+self.overden.redshift)
        #Hubble factor (~70km/s/Mpc) at z=0:
        hubble = 100 * self.ureg.hub* self.ureg.km / (1*self.ureg.s) / (1*self.ureg.Mpc)
        hubz2 = (self.overden.omega_matter0/aa**3 + self.overden.omega_lambda0) * hubble**2
        #Critical density at redshift in units of kg m^-3
        rhocrit = 3 * hubz2 / (8*math.pi* self.ureg.newtonian_constant_of_gravitation)
        return rhocrit.to_base_units()

    def R200(self, mass):
        """Get the virial radius in Mpc for a given mass in Msun"""
#         assert self.ureg.get_dimensionality('[mass]') == self.ureg.get_dimensionality(mass)
        rhoc = self.rhocrit()
        #Virial radius R200 in Mpc from the virial mass
        R200 = ((3 * mass / (4* math.pi* 200 * rhoc)).to('Mpc**3'))**(1/3.)
        return R200.to('Mpc')

    def Rs(self, mass):
        """Scale radius of the halo in Mpc"""
        conc = self.concentration(mass)
        return self.R200(mass)/conc

    def virialvel(self, mass):
        """Get the virial velocity in m/s for mass in Msun"""
#         assert self.ureg.get_dimensionality('[mass]') == self.ureg.get_dimensionality(mass)
        return np.sqrt(2*self.ureg.newtonian_constant_of_gravitation*mass/self.R200(mass)).to_base_units()

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
#         assert self.ureg.get_dimensionality('[mass]') == self.ureg.get_dimensionality(mass)
        prefac = ((4*math.pi)**2*(85*math.pi/3)**(2./7)*self.ureg.newtonian_constant_of_gravitation**2/self.ureg.speed_of_light**3).to_base_units()
        sigma = (self.vel_disp(mass)/self.ureg.speed_of_light).to_base_units()
        vvir = (self.virialvel(mass)/self.ureg.speed_of_light).to_base_units()
#         assert self.ureg.get_dimensionality('') == self.ureg.get_dimensionality(sigma)
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
        assert np.all(probnorm.magnitude > 0)
        cross_section = prefac*(gammaint + cutoff)/probnorm
#         assert self.ureg.get_dimensionality('[length]**3 [time]**(-1) [mass]**(-2)') == self.ureg.get_dimensionality(cross_section)
        return cross_section

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
        if self.ureg.get_dimensionality('') == self.ureg.get_dimensionality(mass):
            mass = mass * self.ureg.Msolar
        conc = self.concentration(mass)
        crosssec = self.cross_section(mass)
        #In m
        Rs = self.Rs(mass)
        rho0 = self.rho0(mass)
        rate = crosssec * 2 * math.pi * rho0**2 * Rs**3 /3 * (1 - 1/(1+conc)**3)
        return rate.to('year**(-1)')

    def rho0(self, mass):
        """Central density for the NFW halo in units of M_sun Mpc^-3"""
#         assert self.ureg.get_dimensionality('[mass]') == self.ureg.get_dimensionality(mass)
        conc = self.concentration(mass)
        return mass / ( 4 * math.pi * self.Rs(mass)**3 * ggconc(conc))

    def mergerpervolume(self, lowermass=None, uppermass=None):
        """The merger rate for primordial black holes in events per Gpc per yr."""
        #See notes for these limits
        if lowermass is None:
            lowermass = 400*self.ureg.Msolar
        if uppermass is None:
            uppermass = 1e16*self.ureg.Msolar
        if self.ureg.get_dimensionality('') == self.ureg.get_dimensionality(uppermass):
            uppermass = uppermass * self.ureg.Msolar
        if self.ureg.get_dimensionality('') == self.ureg.get_dimensionality(lowermass):
            lowermass = lowermass * self.ureg.Msolar
        #mass has units M_sun
        mass = np.logspace(np.log10(lowermass/self.ureg.Msolar),np.log10(uppermass/self.ureg.Msolar),1000)*self.ureg.Msolar
        integrand = self.halomergerratepervolume(mass)
        #trapz needs a wrapper: because we are integrating d log M the units do not change.
        int_units = self.ureg.Gpc**(-3)/self.ureg.year
        trapz = self.ureg.wraps(int_units, int_units)(np.trapz)
        mergerrate = trapz(integrand,np.log(mass/self.ureg.Msolar))
        return mergerrate.to('Gpc**(-3) year**(-1)')

    def mergerfraction(self, vvir, time=None, bhmass = None):
        """Compute the fraction of black hole binaries which merge within time,
        following O'Leary+2008 Eq. 27. for a halo of virial velocity vvir.
        Time is in Gyr.
        bhmass is in M_sun and is the mass of the merging objects."""
        if bhmass is None:
            bhmass = 30 * self.ureg.Msolar
        if time is None:
            time = 6 * self.ureg.Gyear
#         assert self.ureg.get_dimensionality(self.ureg.speed_of_light) == self.ureg.get_dimensionality(vvir)
        #Convert time to s
        c1 = 3* math.sqrt(3)/(170*math.sqrt(85* math.pi))
        #c2 = (340*math.pi/3)**(1/7.)
        prefac = (self.ureg.speed_of_light**3 * time / (c1 * self.ureg.newtonian_constant_of_gravitation * 2 * bhmass ))**(2/21.)
        effcs = (prefac * (vvir/self.ureg.speed_of_light)**(2/7.)).to_base_units()
        return effcs**2

    def halomergerratepervolume(self, mass):
        """The merger rate per year per unit volume for halos in a mass bin."""
        if self.ureg.get_dimensionality('') == self.ureg.get_dimensionality(mass):
            mass = mass * self.ureg.Msolar
        #pbhrate is 1/s
        pbhrate = self.pbhpbhrate(mass)
        dndm_func = self.ureg.wraps('Mpch**(-3) Msolarh**(-1)', self.ureg.Msolarh)(self.dndm)
        #dndm has units: h^4 M_sun^-1 Mpc^-3
        dndm = dndm_func(mass).to('Mpc**(-3) Msolar**(-1)')
        assert np.all(dndm.magnitude >= 0)
        #So result is (Mpc)^-3 s^-1
        return (dndm * pbhrate * mass).to('Gpc**(-3) year**(-1)')

    def evaptime(self,mass, bhmass=None):
        """The evaporation timescale following Binney and Tremaine."""
        if bhmass is None:
            bhmass = 30 * self.ureg.Msolar
        vv = self.vel_disp(mass)
        Rs = self.Rs(mass)
        return (14 * mass/ bhmass / np.log(mass/bhmass) * Rs / vv).to('Gyear')

    def threebodyratio(self,mass, bhmass=None):
        """The ratio between three body and two body binary formation rate.
        This becomes large in small halos."""
        if bhmass is None:
            bhmass = 30 * self.ureg.Msolar
        vel = self.vel_disp(mass)
        return 18*(mass/bhmass)**(-2)*(vel/self.ureg.speed_of_light)**(-10./7)

    def mergerhalflife(self,mass,threefac=True, bhmass=None):
        """The timescale for 50% of the mass of the halo to have merged."""
        rate = self.pbhpbhrate(mass)
        if bhmass is None:
            bhmass = 30 * self.ureg.Msolar
        if threefac:
            threefac = self.threebodyratio(mass)
            threefac = np.max([threefac, np.ones_like(threefac)],axis=0)
            rate *= threefac
        return 0.5*(mass/bhmass)/rate

    def bias(self,mass):
        """The formula for halo bias in EPS theory (Mo & White 1996), eq. 13"""
        nu = self.get_nu(mass)
        bhalo = (1 + (nu**2 -1)/ 1.686)
        return bhalo

class EinastoHalo(NFWHalo):
    """Einasto profile with alpha = 0.18"""
    def pbhpbhrate(self, mass):
        """The merger rate for primordial black holes (per year) in a halo of mass in Msun/h, computed in the attached pdf."""
        #Virial radius R200 in Mpc/h from the virial mass
        if self.ureg.get_dimensionality('') == self.ureg.get_dimensionality(mass):
            mass = mass * self.ureg.Msolar
        conc = self.concentration(mass)
        alpha = 0.18
        crosssec = self.cross_section(mass)
        rho0 = self.rho0(mass)
        d2 = np.exp(4/alpha) * self.Rs(mass)**3 /alpha * (alpha/4)**(3/alpha) * scipy.special.gammainc(3/alpha, 4/alpha * conc**alpha) * scipy.special.gamma(3/alpha)
        rate = 2 * math.pi* crosssec * d2 * rho0**2
        return rate.to('year**(-1)')

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
    plt.savefig("halomergerrate.pdf")
    plt.clf()

def plot_pbh_per_mass(redshift):
    """Plot the PBH merger rate per unit volume as a function of halo mass"""
    mass = np.logspace(2,15)
    hh = NFWHalo(redshift,conc_model="ludlow")
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
    plt.savefig("volumemergerrate.pdf")
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
    plt.savefig("concentration.pdf")
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
    return merg.to('Gpc**(-3) year**(-1)') / (1+z)

def rate_over_redshift(zmin=0., zmax=20., nred = 100,conc="Ludlow", halo="Einasto"):
    """Compute the merger rate over a wide redshift range."""
    zzs = 1/np.linspace(1/(1+zmax), 1/(1+zmin),nred) -1.
    mergers = np.array([merger_at_z(zz,conc=conc, halo=halo).magnitude for zz in zzs])
    return zzs, mergers

def redshift_tables(nred=100):
    """Print tables of the redshift evolution of the mergers"""
    zzs, mergers = rate_over_redshift(conc="Ludlow", halo="Einasto", nred=nred)
    np.savetxt("ludlow_einasto.txt", np.array((zzs,mergers)).T)
    zzs, mergers = rate_over_redshift(conc="Ludlow", halo="NFW", nred=nred)
    np.savetxt("ludlow_nfw.txt", np.array((zzs,mergers)).T)
    zzs, mergers = rate_over_redshift(conc="Prada", halo="Einasto", nred=nred)
    np.savetxt("prada_einasto.txt", np.array((zzs,mergers)).T)
    zzs, mergers = rate_over_redshift(conc="Prada", halo="NFW", nred=nred)
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
