# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:00:06 2021

@author: To+
"""

import numpy as np
import math
#import numpy.random as rand
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import scipy.interpolate as interp
#from ass_3_NO import *
from numba import njit
from iminuit import Minuit 

def polynomial(xi,p1=0.55, p2=0.9):
    nu=0.09585*p1+0.809613*p2+2.13
    return 1/nu*(1+p1*xi+p2*xi**2)
def poisson(xi,p=3.8):
    return p**(xi)*np.e**(-p)/math.factorial(xi)

def derivative(f, h=1E-5):
    def aux(x):
        d= (f(x+h)-f(x-h))/(2*h)
        return np.array([d])
    return aux

def likelihood(f, x,mode):
    if mode=='1':
        def Lik(p1,p2):
            L=0
            for i in range(len(x)):
                    L+=np.log(f(x[i],p1,p2))
            return L
    elif mode=='2':
        def Lik(p):
            L=0
            for i in range(len(x)):
                    L+=np.log(f(x[i],p))
            return L
    return(Lik)      

def gaussian(x, p):
    return 1/(np.sqrt(2*np.pi)*p[1])*np.exp(-np.power(x - p[0], 2.) / (2 * np.power(p[1], 2.)))

x=[]
samples=0
while samples<513:
   xi=int(1E2*np.random.rand())
   r=poisson(xi)
   s=np.random.rand()
   if(s<r):
       x.append(xi)
       samples+=1
       
plt.figure()
plt.hist(x, bins=10)


np.savetxt("bouvier_poisson.txt",x)


x=[]
samples=0
while samples<807:
   xi=2.13*np.random.rand()-1.02 
   r=polynomial(xi)
   s=np.random.rand()
   if(s<r):
       x.append(xi)
       samples+=1
       
plt.figure()
plt.hist(x, bins=30)
plt.figure()

np.savetxt("bouvier_polynomial.txt",x)

#y,x=np.histogram(x,bins=30); y=y/float(sum(y))

paux=Minuit(likelihood(polynomial, x, '1'), p1=0.55, p2=0.9)

p1=paux.values[0];p2=paux.values[1]
a=np.linspace(min(x), max(x), 1000)
#plt.plot(a, polynomial(a, p1,p2), 'r--')
y,x=np.histogram(x,bins=30); y=y/float(sum(y))
plt.plot(a,polynomial(a, p1,p2)/sum(polynomial(a, p1,p2)), 'r--')
plt.plot(x[1:],y,'b.')


#%%

#Exercise 2

df= pd.read_csv("velocity_vs_radius.csv")
df.fillna(0)


MDmtot=[0,2e42, 1.4e43, 4.941E42] #kg
def MT(MDmtot, r):
    h= 15 #kpc
    MNr=np.array([5,10,50,100,200])
    MN=np.array([2.85e40,3.838e40, 4.0e40, 4.0e40, 4.0e40])
    f=interp.interp1d(MNr,MN)
    return(MDmtot*(np.log((h+r)/h)+h/(h+r)-1),f(r))

MD,MN=MT(MDmtot[0], 5)

def V(r,Mt):
    G=4.3E-6/(1.98847E30)
    return np.sqrt(G*(Mt)/r)

x=np.linspace(5,200, 1000)

v= np.array(list(df['measured velocity [km/s]']))
r= np.array(list(df['r to center [kpc]']))
plt.figure()
for i in range(len(MDmtot)):
    plt.plot(x, V(x, sum(MT(MDmtot[i],x))), label=MDmtot[i])
plt.plot(r,v, 'b.', label="experimental")
plt.xlabel("r (kpc)")
plt.ylabel("v (km/s)")
plt.legend(loc="best", fontsize=9, title="$M_{DMtotal} (kg) $ ",ncol=3)

r=df['r to center [kpc]'].to_numpy(); v= df['measured velocity [km/s]'].to_numpy()

def Likelihood(x,y, f):
    def Lik(p):
        aux=0
        for i in range(len(y)):
            p1=sum(MT(p,x[i]))
            aux+=np.log(gaussian(y[i],[f(x[i],p1),0.1*f(x[i], p1)]))
        return aux
    return Lik

def Likelihood2(x,y, f):
        def aux(p):
            return Likelihood(x,y, f)(10**p)
        return aux

#best_p= opt.minimize(Likelihood(r,v,V), x0=42, method='BFGS',tol=1E-1,)
plt.figure()
x=np.linspace(3E42, 3E43, 1000)
plt.plot(x,np.log(10**Likelihood(r,v,V)(x)))
plt.xlabel("$M_{DMtotal}$ $ (kg) $")
plt.ylabel(" LLH ")

#V(r[0], 1E39)= 20.79647384; 
#maximo a ojo en mu=4.941E42   sigma=5.006E42-4.941E42


"""
def gaussian2(x, p0,p1,p2):
    return p0*np.exp(-np.power(x - p2, 2.) / (2 * np.power(p1, 2.)))

y=-Likelihood2(r,v,V)(x)
p,pcov=opt.curve_fit(gaussian2, x,y, p0=[3,42,4])
"""
#%%
#exercise 3

perc=np.array([35,15,5,20,25],dtype=float); perc*=1/100
defect=np.array([2,4,10, 3.5,3.1]); defect*=1/100

ratio=0
for i in range(len(perc)):
    ratio+= perc[i]*defect[i]
for j in range(len(perc)):
    print("prob that defect comes from A",j,": " , perc[j]*defect[j]/ratio )

#MOST LIKELY TO COME FROM A4



print(perc*defect)

@njit()
def sigma(defect, perc):
    pob=perc*defect; #pob*=1/np.sum(pob)
    return(np.sum(pob)+30*np.std(pob)) #feature scaling para caso1 30

#@njit()
def MCMC(defect,perc,itmax=1E7, beta=5E-4):
    sigma1=1E25
    monitor=[]
    acc=0
    it=0
    defect0=defect.copy()
    
    model=[]
    while(it<itmax):
        it+=1
        i=np.random.randint(0, len(defect))
        defect_new=defect.copy()
        defect_new[i]=(1-defect0[i])*np.random.rand()+defect0[i]
        #defect_new=(np.ones(len(defect))-defect0)*np.random.rand(len(defect))+defect0
        sigma2=sigma(defect_new, perc)
        r=sigma2-sigma1
        if r<0:
            defect=defect_new.copy()
            sigma1=sigma2
            acc+=1
        
        elif(np.random.rand()<np.e**(-r/beta)):
                defect=defect_new.copy()
                sigma1=sigma2
                acc+=1
    
        monitor.append(sigma1)
        
        if(it>0.9*itmax):
            model.append(defect)
        
    defect=np.mean(model,axis=0) 
    plt.plot(monitor)
    print("cost: ", sigma(defect,perc)-np.mean(perc*defect0))
    print("perc*defect: ",perc*defect)
    print("defect: ",defect)
    print("u_defect: ",np.std(model,axis=0) )
    print("defect0: ",defect0)
    print("perc*defect0: ",perc*defect0)
    return(defect)



perc2=np.array([0.27,0.1,0.05,0.08,0.25,0.033,0.019, 0.085, 0.033, 0.02, 0.015,0.022,0.015,0.008],dtype=float); 
defect2=np.array([0.02,0.04,0.1,0.035, 0.022,0.092,0.12,0.07,0.11,0.02, 0.07,0.06,0.099,0.082], dtype=float); 


defect=MCMC(defect2, perc2, itmax=1E6, beta=2E-4)