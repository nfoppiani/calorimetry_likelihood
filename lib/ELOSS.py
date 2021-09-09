import numpy as np

mass_dict = {'electron': 0.511,
             'muon': 105.658,
             'pion': 139.570,
             'proton': 938.272}

class Eloss:

    def __init__(self, setDefaultArgon=False):
        self.mass_dict = mass_dict
        if setDefaultArgon:
            self.setDefaultArgonParameters()
        self.dEdx_from_Range_v = np.vectorize(self.dEdx_from_Range, excluded=['self', 'mass', 'pitch', 'stepmax', 'thisstep'])
        self.dQdx_from_Range_v = np.vectorize(self.dQdx_from_Range, excluded=['self', 'mass', 'pitch', 'stepmax', 'thisstep']) 
        
    def setParameters(self, I, rho, Z, A, K, Wion, E_field, ModBoxA, ModBoxB):
        self.I   = I   # mean excitation energy [eV]
        self.rho = rho # argon density [g/cm3]
        self.Z = Z
        self.A = A
        self.I = I
        self.K = K
        self.Wion = Wion
        self.E_field = E_field
        self.ModBoxA = ModBoxA
        self.ModBoxB = ModBoxB
        
    def setDefaultArgonParameters(self):
        self.rho = 1.396 # argon density [g/cm3]
        self.Z = 18
        self.A = 39.948   # g / mol
        self.I = 188.0*(10**(-6)) # MeV
        self.K = 0.307 # MeV * cm^2 / mol
        
        self.Wion = 23.6/1e6; # 23.6 eV = 1e, Wion in MeV/e
        self.E_field = 0.274 # kV / cm
        self.ModBoxA = 0.930
        self.ModBoxB = 0.212
        self.Alpha = self.ModBoxA
        self.Beta = self.ModBoxB / (self.rho * self.E_field)
        
        self.Me = self.mass_dict['electron']
        
    # set density [input in g/cm3]
#     def setDensity(rho):
#         self.rho = rho

#     # set mean excitation energy I [input in eV]
#     def setI(I):
#         self.I = I

#     # print argon properties currently loaded
#     def PrintInfo():
#         print
#         print 'Density .............. : %.03f g/cm3'%self.rho
#         print '<ean excitation energy : %.03f eV'%self.I
#         print
    
    @staticmethod
    def beta(gamma):
        return np.sqrt(1-(1./(gamma**2)))
    @staticmethod
    def gamma(KE, mass):
        return (KE/mass)+1
    
    # density correction for LAr
    @staticmethod
    def density_correction(bg):
        # constants and variable names obtained from :
        # PDG elos muons table [http://pdg.lbl.gov/2016/AtomicNuclearProperties/MUE/muE_liquid_argon.pdf]

        C  = -5.2146
        X0 = 0.2
        X1 = 3.0
        a  = 0.19559
        m  = 3.0
        N    = 2 * np.log(10)

        x = np.log10(bg)

        if (x < X0):
            return 0.
        if (x > X1):
            return N * x + C
        addition = a*((X1-x)**m)
        return N * x + C + addition
    
    @staticmethod
    def Wmax(KE, mass, Me):
        g = Eloss.gamma(KE, mass)
        b = Eloss.beta(g)
        num = 2*Me*((b*g)**2)
        den = 1 + 2*g*Me/mass + (Me/mass)**2
        return num/den
    
    @staticmethod
    def Tfunc(g, Me):
        return (g-1.)*Me
    
    @staticmethod
    def tfunc(g):
        return (g-1.)
    
    @staticmethod
    def Fminus(b,t):
        f = (1-b*b) * ( 1 + t*t/8. - (2*t+1)*np.log(2) )
        return f
    
    @staticmethod
    def Fplus(b,t):
        f = 2*np.log(2) - (b*b/12.) * ( 23. + 14./(t+2.) + 10./(t+2.)**2 + 4./(t+2.)**3 )
        return f
    
    # KE in MeV
    # x in cm
    # mass in MeV
    def dpdx(self, KE, pitch, mass):
        if isinstance(mass, str):
            mass = self.mass_dict[mass]
        g = self.gamma(KE, mass)
        b = self.beta(g)
        epsilon = (self.K/2.)*(self.Z/self.A)*(pitch*self.rho/(b*b))
        A0 = (2*self.Me*(b*g)**2)/self.I
        A1 = epsilon/self.I
        return (1./pitch) * epsilon * (np.log(A0) + np.log(A1) + 0.2 - (b*b) - self.density_correction(b*g))

    # in MeV/cm
    def dedx(self, KE, mass, dens=True):
        if isinstance(mass, str):
            mass = self.mass_dict[mass]
        g = self.gamma(KE, mass)
        b = self.beta(g)
        F = self.K * (self.Z/self.A)*(1/b)**2
        wmax = self.Wmax(KE, mass, self.Me)
        a0 = 0.5*np.log( 2*self.Me*(b*g)**2 * wmax / (self.I**2) )
        ret = a0 - b*b
        if dens:
            ret -= self.density_correction(b*g)/2.
        return F * ret

    def dedxelectrons(self, beta, dens=True):
        g = 1./np.sqrt(1-beta*beta)
#         T = Tfunc(g)
        t = self.tfunc(g)
        T = t * self.Me
        F = (self.Z/self.A) * 0.153536 * ((1./beta)**2)
        ret =  (np.log((T/self.I)**2)) + np.log(1.+t/2.) + self.Fminus(beta, t)
        if dens:
            ret -= self.density_correction(beta*g)
        return ret * F

    def dedxpositrons(self, beta, dens=True):
        g = 1./np.sqrt(1-beta*beta)
#         T = Tfunc(g)
        t = self.tfunc(g)
        T = t * self.Me
        F = (self.Z/self.A) * 0.153536 * ((1./beta)**2)
        ret = (np.log((T/self.I)**2)) + np.log(1+t/2.) + self.Fplus(beta, t)
        if dens:
            ret -= self.density_correction(beta*g)
        return ret * F

    def Range(self, KE, mass, dx):
        dist = 0.
        while KE > 0.1:
            #print 'star KE : %.02f'%KE
            eloss = rho * self.dedx(KE, mass, dens=True)
            KE -= (eloss * dx)
            dist += dx
            #print 'KE : %.02f dEdx : %.02f distance : %.02f'%(KE,eloss,dist)
        return dist

    def KE_from_Range(self, R, mass, stepmax=0.1, thisstep=1e-3):
        KE = 0.3 # MeV
        dist = 0.
        thisstep = 1e-3        
        while ((dist+thisstep) < R):
            eloss = self.rho * self.dedx(KE, mass, dens=True)
            #print 'step: %.03f dEdx at KE %.02f is %.02f. Total dist : %.03f'%(thisstep,KE,eloss,dist)
            KE += (eloss * thisstep)
            dist += thisstep
            # update step size in an efficient way
            if ((thisstep < dist/10.) and (dist/10. < stepmax)):
                thisstep = dist/10.
        return KE
    
    def dEdx_from_Range(self, R, mass, pitch, stepmax=0.1, thisstep=1e-3):
        KE = 0.3 # MeV
        dist = 0.
        thisstep = 1e-3        
        while ((dist+thisstep) < R):
            eloss = self.rho * self.dedx(KE, mass, dens=True)
            #print 'step: %.03f dEdx at KE %.02f is %.02f. Total dist : %.03f'%(thisstep,KE,eloss,dist)
            KE += (eloss * thisstep)
            dist += thisstep
            # update step size in an efficient way
            if ((thisstep < dist/10.) and (dist/10. < stepmax)):
                thisstep = dist/10.
        return self.dpdx(KE, pitch, mass)
    
    #bisogna passare da 45000 a 500
    def dQdx_from_dEdx(self, dedx):
        return np.log(self.Beta * dedx + self.Alpha)/(self.Beta*self.Wion)
    
    def dQdx_from_Range(self, R, mass, pitch, stepmax=0.1, thisstep=1e-3):
        return self.dQdx_from_dEdx(self.dEdx_from_Range(R, mass, pitch, stepmax, thisstep))
    
    # vectorised version of the functions       