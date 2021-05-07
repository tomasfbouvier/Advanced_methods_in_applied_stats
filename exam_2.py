# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:11:59 2021

@author: To+
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.stats as stats

az, ze=np.loadtxt("Exam_2021_Problem2.txt", unpack=True)

def chi2(O, E): 
    chi2=0;
    for i in range(len(O)):
        chi2+=((O[i]-E[i])/E[i])**2
    return chi2

def correlation(phi, phis):
    N_tot=len(phis)
    
    def angular_distance(phi1, phi2):
        n1=np.array([np.cos(phi1), np.sin(phi1)]); n2=np.array([np.cos(phi2), np.sin(phi2)])
        return n1@n2
        
    
    C=0
    for i in range(N_tot):
        for j in range(i):
            C+= np.heaviside( np.cos(angular_distance(phis[i],phis[j]))-np.cos(phi),1)
    C*=2/(N_tot*(N_tot-1))
    
    return C

def sphericalrandom():
       theta= 2*np.pi*np.random.rand()
       phi= np.arccos(1-2*np.random.rand())
       return(theta, phi)
   
def histo(c, bins):
    y,x= np.histogram(c, bins)
    y= y/sum(y*(x[1]-x[0]))
    #print(x), print(y)
    #plt.figure();plt.plot(y[:-1], x)
    return x, y       

def KS(sample1, sample2, x):
    cdf_1=[correlation(x[0], sample1)]
    for i in range(1,len(x)):
        cdf_1.append(cdf_1[i-1]+correlation(x[i], sample1))
    
    cdf_2=[correlation(x[0], sample2)]
    for i in range(1,len(x)):
        cdf_2.append(cdf_2[i-1]+correlation(x[i], sample2))
        
    cdf_1=np.array(cdf_1); cdf_1/=max(cdf_1)    
    cdf_2=np.array(cdf_2); cdf_2/=max(cdf_2)  
    
        
    return max(abs(cdf_1 -cdf_2))
        

thetas=[]; phis= []
for i in range(100):
    theta, phi=sphericalrandom()
    thetas.append(theta); phis.append(phi)

thetas=np.array(thetas); phis=np.array(phis)


x=np.linspace(0., 1., 100)

result_isotropic_ze=KS(x, ze, phis)

print("result_isotropic_ze:", result_isotropic_ze)

result_isotropic_az=KS(x, az, thetas)

print("result_isotropic_az:", result_isotropic_az)

#f1=stats.gaussian_kde(cdf_i)






#%%

def hypo_A(size):
    thetas=[]; phis= []
    for i in range(size):
        theta, phi=sphericalrandom()
        thetas.append(theta); phis.append(phi)
    for i in range(int(0.25*size)):
        thetas.append((0.725-0.225)*np.pi*np.random.rand()+0.225*np.pi)
        phis.append((1.-0.30)*np.pi*np.random.rand()+0.30*np.pi)


    return thetas, phis

    
thetas, phis=hypo_A(100)
#plt.hist(phis, 100)


x=np.linspace(0., 1., 100)

result_isotropic_ze=KS(x, ze, phis)

print("result_hypo_A_ze:", result_isotropic_ze)

result_isotropic_az=KS(x, az, thetas)

print("result_hypo_A_az:", result_isotropic_az)

#%%


def hypo_B(size):
    thetas=[]; phis= []
    for i in range(size):
        theta, phi=sphericalrandom()
        thetas.append(theta); phis.append(phi)
    for i in range(int(0.25*size)):
        thetas.append((1.-0.)*np.pi*np.random.rand()+0.225*np.pi)
        phis.append((1.-0.5)*np.pi*np.random.rand()+0.30*np.pi)


    return thetas, phis


thetas, phis=hypo_B(100)

result_isotropic_ze=KS(x, ze, phis)

print("result_hyo_B_ze:", result_isotropic_ze)

result_isotropic_az=KS(x, az, thetas)

print("result_hypo_B_az:", result_isotropic_az)


