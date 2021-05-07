# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:23:24 2021

@author: To+
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})



c1,c2,c3,c4,c5=np.loadtxt("samples.txt", unpack=True)

bnds = ((-10., 10.), (-10., 10.), (4000., 8000.), (0.,20.))

def f1(x,p):
    return p[1]*(1/(x+5)*np.sin(p[0]*x))

def f2(x,p):
    return p[1]*((1+np.sin(p[0]*x)))
    
def f3(x,p):
    return p[1]*np.sin(p[0]*x*x)

def f4(x,p):
    return p[1]*x*np.tan(x)

def f5(x,p):
    return p[2]*(1+p[0]*x+p[1]*x*x)

def f6(x, p):
    return(p[0]+p[1]*x)

def f7(x,p):
    return(p[3]*(np.sin(p[0]*x)+p[1]*np.e**(p[2]*x)+1))

def f8(x, p):
    return p[2]*np.e**(-((x-p[0])/(2*p[1]))**2)


def find_best_value(xs, f, bnds):

    def log_likelihood(xs, f):
        def LLH_aux(p):
            LLH=0
            for x in xs:
                LLH+=np.log(f(x, p))
            
            return -LLH
        return(LLH_aux)


    #xs-=np.mean(xs); xs/=np.std(xs)
    
    """
    a=np.linspace(bnds[0][0], bnds[0][1], 100) 
    b=np.linspace(bnds[1][0], bnds[1][1], 100)
   
    
    a, b = np.meshgrid(a, b)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Z=log_likelihood(xs,f)([a,b])
    surf = ax.plot_surface(a, b, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    """

    p=opt.minimize(log_likelihood(xs, f), x0=[0.,0., 5000., 10.], bounds=bnds)
    return(p)


p=find_best_value(c1, f7,bnds)

print(p)

x=np.linspace(min(c1), max(c1), 1000)

plt.figure()
plt.hist(c1)
plt.plot(x, f8(x, p.x))