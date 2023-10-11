from math import pi, isnan, inf
from astropy import constants
from scipy.integrate import odeint
from scipy.optimize import bisect
import matplotlib.pyplot as plt
import numpy as np

#Constants

### This is what we change
lam = 0


sigma   = 5.6e-8# constants.sigma_sb.value
c       = 3e8 #constants.c.value
a       = 4.*sigma/c
G       = 6.67e-11 #constants.G.value
m_p     = 1.67e-27 #constants.m_p.value
m_e     = 9.1e-31 #constants.m_e.value
hbar    = 1.054e-34 #constants.hbar.value
k       = 1/38e-23# constants.k_B.value     
X       = 0.70                          # Fraction of Hydrogen
Z       = 0.02                          # Fraction of Metals
Y       = 1 - X - Z                     # Fraction of Helium
mu      = (2*X+0.75*Y+0.5*Z)**(-1)      # mean molecular mass
gamma   = 5/3.
kappa_es = 0.02*(1+X) 
R_sun   = 7e7 #constants.R_sun.value
M_sun   = 2e30 #constants.M_sun.value
L_sun   = 3.8e26 #constants.L_sun.value


class Star:
def __init__(self):
self.r0          = 1e-15 # initial r, 0 would be nice if you could devide by it
self.hmax        = 1e8   #0.1% sun res, time step for RK
self.hmin        = 5e2   #1km min res, time step for RK
self.dtau_list      = [] #start list for dtau used in try_star
self.rho_c_list     = [] #start list for rho_c used in MS and try_star
self.optimal_target = [] #start list for optimal_target used in MS and try_star

def MS(self,central_T,save,file_name):
self.save       = save
self.file_name  = file_name
self.central_T  = central_T
bisect(self.try_star, 0.3e3,500e3,xtol=1e-30)
print(bisect(self.try_star, 0.3e3,500e3,xtol=1e-30))
correct_central_rho = self.rho_c_list[np.argmin(np.array(self.optimal_target))] 
L,M,T,R = self.point_star(correct_central_rho)
print('Luminosity:', L)
print('Mass:', M)
print('Temperature:',T)
output = [L,M,T,R]
return output

## trys star        
def try_star(self,central_rho):
self.index = 1
self.htol  = 1e-8       
rho  = central_rho  
T    = self.central_T
M    = 4*pi*self.r0**3.*rho/3.
L    = 4*pi*self.r0**3.*rho*self.epsilon(rho,T)/3.
RK_init = np.array([rho,T,M,L,0])                
RK_output = np.array(self.odeint(RK_init))       
rho  = RK_output[:,0]
T    = RK_output[:,1]
M    = RK_output[:,2]
L    = RK_output[:,3]
tau  = RK_output[:,4]
index = self.R_s(tau)                                                 
target_optimization = self.f(self.r[index], T[index], L[index])    
self.optimal_target.append(abs(target_optimization))
self.dtau_list.append(self.dtau)
self.rho_c_list.append(central_rho)
print('NOPE')
return target_optimization

def point_star(self,central_rho):       
self.index = 1
self.htol  = 1e-8       
rho  = central_rho
T    = self.central_T
M    = 4*pi*self.r0**3.*rho/3.
L    = 4*pi*self.r0**3.*rho*self.epsilon(rho,T)/3.           
RK_init = np.array([rho, T, M, L, 0])  
RK_output = np.array(self.odeint(RK_init))
rho  = RK_output[:,0]
T    = RK_output[:,1]
M    = RK_output[:,2]
L    = RK_output[:,3]
tau  = RK_output[:,4]
P    = self.pressure(rho,T)           
P_IGL = rho*k*T/(mu*m_p)                                                                            #Ideal Gas Law
P_DEG = (3*pi**2.)**(2/3.)*hbar**2.*(rho)**(5/3.)/(5*m_e*m_p**(5/3.))                               #Degenerate
P_RAD = 1/3*a*T**4.                                                                                 #Radiative
dL     = self.dL_dr(np.array(self.r),rho,T)
dL_pp  = self.dL_pp(np.array(self.r),rho,T)
dL_cno = self.dL_cno(np.array(self.r),rho,T)
kaps = self.kappa(rho,T)
kaps_es = np.zeros(len(self.r))+kappa_es
kaps_H  = self.kappa_H(rho,T)
kaps_ff = self.kappa_ff(rho,T)
index = self.R_s(tau) 
dlogT = np.diff(np.log10(T))
dlogP = np.diff(np.log10(P))
std = np.where(dlogP/dlogT - 2.5 < 0.01)
lst = []
for i in range(len(std[0]) - 1):
    if std[0][i + 1] - std[0][i] > 1:
        lst.append(std[0][i])
        lst.append(std[0][i+1])
if len(lst) == 0:
    lst.append(std[0][0])
else:
    lst.insert(0,std[0][0])
lst.append(std[0][-1])
if self.save:
    np.savetxt(self.file_name,np.concatenate((np.reshape(self.r,(len(self.r),1)),RK_output),axis=1), delimiter=',', header='radius, rho, T, M, L, tau, index = {}'.format(index))
return L[index],M[index],T[index],self.r[index]

def odeint(self,vals):
current_r = self.r0
self.step = self.hmin           
self.condition = True        
force = False       
RK_output = [vals]      
self.r = [self.r0]      
M   = vals[2]                  
while self.condition and M <= M_sun*3e2 and self.r[-1] <= 20*R_sun:           
    vals_2 = self.RK45(vals, current_r, force=False)           
    rho = vals_2[0]
    T   = vals_2[1]
    M   = vals_2[2]
    L   = vals_2[3]          
    vals = vals_2                
    self.condition = self.opacity_limit(rho,T) # T/F is Opacity limited               
    if not self.condition:
        print('Opacity limited')               
    current_r += self.step
    self.r.append(current_r)
    RK_output.append(vals)
return RK_output

## opacity_limit check        
def opacity_limit(self,rho,T):
drho_dr = self.drho 
kappa = self.kappa(rho,T)
delta_tau = kappa*rho**2./abs(drho_dr)
if delta_tau <= 1e-2: 
    return False
else:
    return True

## Tau surface boundry condition              
def R_s(self,tau): 
i = -1
first = True
self.dtau = 0
while self.dtau <= 2/3.:
    if len(tau) == abs(i):
        i = 0
        break
    elif not isnan(tau[i]) and first:
        tau_inf = tau[i]
        first = False
    elif not first:
        self.dtau = tau_inf - tau[i]   
    i -= 1
return i

## def DEs into array for RK45   
def DE(self,init,r):
rho = init[0] 
T   = init[1]
M   = init[2]
L   = init[3]
P     = self.pressure(rho,T)
kappa = self.kappa(rho,T)        
self.drho   = self.drho_dr(r,rho,T,L,M,P,kappa)  #self. because subject to change
dT          = self.dT_dr(r,rho,T,L,M,P,kappa)
dM          = self.dM_dr(r,rho)
dL          = self.dL_dr(r,rho,T)
dtau        = self.dtau_dr(rho,T)        
return np.array([self.drho,dT,dM,dL,dtau])

## Runge Kutta    
def RK45(self,inits,r,force):
f = self.DE
i = 0                       
while i <= 1:
    k1 = self.step*f(inits,r)
    k2 = self.step*f(inits + 1/4.*k1                                                    , r+1/4.*self.step  )
    k3 = self.step*f(inits + 3/32.*k1 + 9/32.*k2                                        , r+3/8.*self.step  )
    k4 = self.step*f(inits + 1932/2197.*k1 - 7200/2197.*k2 + 7296/2197.*k3              , r+12/13.*self.step)
    k5 = self.step*f(inits + 439/216.*k1 - 8*k2 + 3680/513.*k3 - 845/4104.*k4           , r+self.step       )
    k6 = self.step*f(inits - 8/27.*k1 + 2*k2 - 3544/2565.*k3 + 1859/4104.*k4 - 11/40.*k5, r+1/2.*self.step  )    
    y1 = inits + 25/216.*k1 + 1408/2565.*k3 + 2197/4104.*k4 - 1/5.*k5
    z1 = inits + 16/135.*k1 + 6656/12825.*k3 + 28561/56430.*k4 - 9/50.*k5 + 2/55.*k6
    if i == 0:
        s = (self.htol/(2*abs(z1[self.index]-y1[self.index])))**(1/4.)                              
        if abs(s) == inf:
            s = 1
        elif isnan(s):
            s = 0.5               
        self.step *= s                              
        if r == self.r0:
            self.step = self.hmin
        elif self.step > self.hmax:
            self.step = self.hmax
        elif self.step < self.hmin and not force:
            self.step = self.hmin           
    i += 1            
return y1

## Luminosity boundary    
def f(self,R_s,T_s,L_s):
return (L_s-4*pi*sigma*R_s**2.*T_s**4.)/np.sqrt(4*pi*sigma*R_s**2.*T_s**4.*L_s)  

## Pressure
def pressure(self,rho,T):                                                                               #Pressure
P_IGL = rho*k*T/(mu*m_p)                                                                            #Ideal Gas Law
P_DEG = (3*pi**2.)**(2/3.)*hbar**2.*(rho)**(5/3.)/(5*m_e*m_p**(5/3.))                               #Degenerate
P_RAD = 1/3*a*T**4.                                                                                 #Radiative

return P_IGL + P_DEG + P_RAD                                            

## Opacities
def kappa_ff(self,rho,T):                                                                               # Free-free opacity
return 1.0e24*(Z+0.0001)*(rho*1e-3)**0.7*T**(-3.5)

def kappa_H(self,rho,T):                                                                                # H- opacity
return 2.5e-32*(Z/0.02)*(rho*1e-3)**0.5*(T**9.)   

def kappa(self,rho,T):
return ( 1./self.kappa_H(rho,T) + 1./np.maximum(kappa_es,self.kappa_ff(rho,T)) )**(-1) 

## dT/dr
def dT_dr_rad(self,r,rho,T,L,kappa):                                                                    # radiative dT
return 3.*self.kappa(rho,T)*rho*L / ( 16. * pi * a * c * (T**3.) * (r**2.) )         

def dT_dr_conv(self,r,rho,T,M,P):                                                                       # convective dT
return (1-1./gamma)*T*G*M*rho**(1+lam/r)/(self.pressure(rho,T)*r**2.)                                              

def dT_dr(self,r,rho,T,L,M,P,kappa):                                                                    # dT
return -min(self.dT_dr_rad(r,rho,T,L,kappa), self.dT_dr_conv(r,rho,T,M,P))

## Reaction Rates 
def epsilon_pp(self,rho,T):                                                                             # reaction rate PP chain
return 1.07e-7*(rho*1e-5)*X**2.*(T*1e-6)**4                                       

def epsilon_cno(self,rho,T):                                                                            # reaction rate CNO cycle
return 8.24e-26*(rho*1e-5)*X**2.*0.03*(T*1e-6)**19.9                                

def epsilon(self,rho,T):                                                                                # total reaction rate
return self.epsilon_pp(rho,T) + self.epsilon_cno(rho,T)                             

## drho/dr
def dP_drho(self,rho,T):                                                                                # dP_drho needed for drho_dr
return (3*pi**2)**(2/3.)*hbar**2.*rho**(2/3.)/(3*m_e*m_p*m_p**(2/3.)) + k*T/(mu*m_p)                

def dP_dT(self,rho,T):                                                                                  # dP_dT needed for drho_dr
return rho*k/(mu*m_p) + 4/3.*a*(T**3.)

def drho_dr(self,r,rho,T,L,M,P,kappa):                                                                  # drho_dr
return -(G*M*rho*(1+lam/r)/(r**2.) + self.dP_dT(rho,T)*self.dT_dr(r,rho,T,L,M,P,kappa))/self.dP_drho(rho,T)

## Luminosity
def dL_dr(self,r,rho,T):                                                                                # dL
return 4*pi*r**2.*rho*self.epsilon(rho,T)                                                           

def dL_pp(self,r,rho,T):                                                                                # dL_pp
return 4*pi*r**2.*rho*self.epsilon_pp(rho,T)    

def dL_cno(self,r,rho,T):                                                                               # dL_cno
return 4*pi*r**2.*rho*self.epsilon_cno(rho,T)    

## Mass
def dM_dr(self,r,rho):                                                                                  # mass continuity
return 4*pi*r**2.*rho                                                                                

## Optical depth
def dtau_dr(self,rho,T):
return self.kappa(rho,T)*rho



## Run and plot
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

R     = [] #will be filled bellow
T     = [] #will be filled bellow 
M     = [] #will be filled bellow
L     = [] #will be filled bellow

temp_range = np.linspace(0.3e7,3.e7,10) #Range of temps for main plot

count = 0
for temp in temp_range: #runs through all temps each temp = each point on main
output = Star().MS(temp,save=True,file_name='./temp_points/{}_{}E6_K.csv'.format(count,int(temp*1e-6)))
L.append(output[0])
M.append(output[1])
T.append(output[2])     
R.append(output[3])
count += 1

R = np.array(R)
T = np.array(T)
L = np.array(L)
M = np.array(M)

print('Radius;',R)
print('Mass;',M)
print('Luminosity;',L)

plt.figure('Main Sequence') 
plt.loglog((L/(4*pi*R*R*sigma))**(1/4.),L,'yo')
plt.gca().invert_xaxis()
plt.xlabel('Temperature [K]')
plt.ylabel('Luminosity [W]')

plt.figure('M/$M_\odot$ vs L/$L_\odot$')
plt.loglog(M/M_sun,L/L_sun,'ro')
plt.xlabel('M/$M_\odot$')
plt.ylabel('L/$L_\odot$')

plt.figure('M/$M_\odot$ vs R/$R_\odot$')
plt.loglog(M/M_sun,R/R_sun,'go')
plt.xlabel('M/$M_\odot$')
plt.ylabel('R/$R_\odot$')

plt.show()

output_list = np.concatenate((R.reshape([-1,1]),L.reshape([-1,1]),M.reshape([-1,1]),T.reshape([-1,1])),axis=1)
#np.savetxt('Main_Sequence_HIRES.txt', output_list, delimiter=',', header='R, L, M, T')