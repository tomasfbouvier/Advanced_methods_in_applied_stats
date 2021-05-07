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
from scipy.stats import poisson, norm

def histo(c, bins):
    y,x= np.histogram(c, bins)
    #y= y/sum(y*(x[1]-x[0]))
    #print(x), print(y)
    #plt.figure();plt.plot(y[:-1], x)
    return x, y     

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})



c1,c2,c3,c4,c5=np.loadtxt("samples.txt", unpack=True)

bnds = {'a':(-10., 10.), 'b':(-10., 10.), 'c':(4000., 8000.)}
p0= {'a':0., 'b':0., 'c':6000.}


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
    return((np.sin(p[0]*x)+p[2]*np.e**(p[1]*x)+1))

def f8(x, p):
    return p[2]*np.e**(-((x-p[0])/(2*p[1]))**2)


def find_best_value(xs,ys, f, bnds, p0):

    def log_likelihood(xs,ys,f):
        def LLH_aux(p):
            LLH=0
            for i in range(len(xs)-1):
                y=f(xs[i], p)
                #print(y-ys[i])
                LLH+=np.log(norm(0., 10.).pdf(y-ys[i]))
            print(-LLH)
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
    
    

    p=opt.minimize(log_likelihood(xs,ys,f), x0=p0, bounds=bnds)
    return(p)


x,y=histo(c1,100)

plt.plot(x[:-1],y)


bnds2=(bnds['a'], bnds['b'], bnds['c'])


p0s=(p0['a'], p0['b'], p0['c'])
p0s=(0.64,-0.2,4100.)

print(find_best_value(x, y, f7, bnds2, p0s))

x2=np.linspace(min(x),max(x),200)
plt.plot(x2,f7(x2,p0s))




"""
p=find_best_value(c1, f7,bnds)

print(p)

x=np.linspace(min(c1), max(c1), 1000)

plt.figure()
plt.hist(c1)
plt.plot(x, f8(x, p.x))
"""