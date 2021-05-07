# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 01:49:56 2021

@author: To+
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
from scipy.optimize import minimize 
from scipy.special import sici, erf

#%%


def histo(c, bins):
    y,x= np.histogram(c, bins)
    y= y/sum(y)
    #print(x), print(y)
    #plt.figure();plt.plot(y[:-1], x)
    return x, y     

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})



c1,c2,c3,c4,c5=np.loadtxt("samples.txt", unpack=True)

bnds = {'a':(-10., 10.), 'b':(-10., 10.), 'c':(4000., 8000.)}
p0= {'a':0., 'b':0., 'c':6000.}

bnds2=(bnds['a'], bnds['b'], bnds['c'])
def f7(x,a,b,c):
    norm= (np.cos(20*a)-np.cos(27*a))/(a)+(np.e**(20*b)*(np.e**(7*b)-1.)*c)/b+7.

    #we will transform this into a PDF right away...
    pdf= (1+np.sin(a*x)+c*np.e**(b*x))
    #...remembering the PDF should be normalized! this is the definite integral over range -1..1
    #return the evaluated PDF only inside -1..1 range, and 0 outside
    """
    if(x.any()<20 or x.any()>27):
        pdf= np.zeros(np.shape(pdf))
    
    """
    return np.where(x>= 20  , pdf/norm, 0.)


def likelihood(x,a,b,c,f):
    pdf=f(x,a,b,c)
    return pdf

def log_likelihood(x,a,b,c):
    #The log-likelihood to get all of the x-points (together) given our custom PDF with parameters alpha and beta.
    return np.sum(np.log(likelihood(x,a,b,c,f7)),axis=1)


x,y=histo(c1,100)

plt.hist(c1,bins=np.linspace(20.,27.,20),density=True,label = 'samples',  color='b', edgecolor='black')

lambda_neg_LLH = lambda params: -log_likelihood(c1,
                                np.expand_dims([params[0]],axis=1),np.expand_dims([params[1]],axis=1),np.expand_dims([params[2]],axis=1))
res=minimize(lambda_neg_LLH,x0=[-10.,-1., 4000],bounds=bnds2,method='L-BFGS-B')

print(res)

p=res.x
errors = np.sqrt(np.diag(res.hess_inv.todense()))

print("p=", p)
print("errors= ", errors)
x2=np.linspace(min(c1),max(c1), 100)

plt.plot(x2, likelihood(x2, p[0], p[1], p[2],f7), label="fit")

plt.xlabel("x")
plt.ylabel("# entries")
plt.legend(loc='best')


#%%

def f1(x,a):
    
    
    
    norm= sici(25*a)[1]*np.sin(5*a)-sici(32*a)[1]*np.sin(5*a)+ (sici(32*a)[0]-sici(25*a)[1])*np.cos(5*a)

    #we will transform this into a PDF right away...
    pdf= 1/(x+5)*np.sin(a*x)
    #...remembering the PDF should be normalized! this is the definite integral over range -1..1
    #return the evaluated PDF only inside -1..1 range, and 0 outside
    """
    if(x.any()<20 or x.any()>27):
        pdf= np.zeros(np.shape(pdf))
    
    """
    return np.where(x>= 20  , pdf/norm, 0.)


def likelihood(x,a,f):
    pdf=f(x,a)
    return pdf

def log_likelihood(x,a):
    #The log-likelihood to get all of the x-points (together) given our custom PDF with parameters alpha and beta.
    return np.sum(np.log(likelihood(x,a,f1)),axis=1)


x,y=histo(c1,100)

#plt.hist(c1,bins=np.linspace(20.,27.,20),density=True,label = 'MC samples')

lambda_neg_LLH = lambda params: -log_likelihood(c1,
                                np.expand_dims([params[0]],axis=1))


bnds2 = (bnds['a']); 
res=minimize(lambda_neg_LLH,x0=[-1.],bounds=((-10.,10.),),method='L-BFGS-B')

print(res)

p=res.x
errors = np.sqrt(np.diag(res.hess_inv.todense()))

print("p=", p)
print("errors= ", errors)
x2=np.linspace(min(c1),max(c1), 100)

#plt.plot(x2, likelihood(x2, p[0],f1))

#%%


#c1,c2,c3,c4,c5=np.loadtxt("samples.txt", unpack=True)


def likelihood(x,alpha,beta):
    #we will transform this into a PDF right away...
    pdf = 1+alpha*x+beta*x**2
    #...remembering the PDF should be normalized! this is the definite integral over range -1..1
    norm = 7./6.*(141*alpha+3338*beta+6.)
    #return the evaluated PDF only inside -1..1 range, and 0 outside
    return np.where(x >= 20, pdf/norm, 0.)

def log_likelihood(x,alpha,beta):
    #The log-likelihood to get all of the x-points (together) given our custom PDF with parameters alpha and beta.
    return np.sum(np.log(likelihood(x,alpha,beta)),axis=1)


lambda_neg_LLH = lambda params: -log_likelihood(c1,
                                np.expand_dims([params[0]],axis=1),np.expand_dims([params[1]],axis=1))


res = minimize(lambda_neg_LLH,x0=[ -10., -1.],bounds=(bnds['a'],bnds['b']),method='L-BFGS-B')

print(res)

p=res.x
errors = np.sqrt(np.diag(res.hess_inv.todense()))

print("p=", p)
print("errors= ", errors)
x2=np.linspace(min(c1),max(c1), 100)

#plt.plot(x2, likelihood(x2, p[0],p[1]))


#plt.hist(c1,bins=np.linspace(20.,27.,20),density=True,label = 'MC samples')

#%% c2


def likelihood(x,alpha,beta):
    #we will transform this into a PDF right away...
    pdf = 1+alpha*x+beta*x**2
    #...remembering the PDF should be normalized! this is the definite integral over range -1..1
    norm = 2+(2*beta/3)
    #return the evaluated PDF only inside -1..1 range, and 0 outside
    return np.where(np.abs(x) <= 1, pdf/norm, 0.)

def log_likelihood(x,alpha,beta):
    #The log-likelihood to get all of the x-points (together) given our custom PDF with parameters alpha and beta.
    return np.sum(np.log(likelihood(x,alpha,beta)),axis=1)
lambda_neg_LLH = lambda params: -log_likelihood(c2,
                                np.expand_dims([params[0]],axis=1),np.expand_dims([params[1]],axis=1))
#notice some broadcasting tricks above, which let us operate on vector-like x-samples as well as vector-like
#parameters alpha and beta. This is no different from what we've done in Exercise 1.

res = minimize(lambda_neg_LLH,x0=[0.3,1.],bounds=((-10.,6.),(-10.,10.)),method='L-BFGS-B')

print(res)

p=res.x
errors=np.sqrt(np.diag(res.hess_inv.todense()))

print("asdjjqwdjjw", p, errors)

x2=np.linspace(min(c2),max(c2), 100)

plt.figure()
plt.plot(x2, likelihood(x2, p[0],p[1]), label="fit")


plt.hist(c2,bins=np.linspace(-1.,1.,20),density=True,label = 'samples',  color='b', edgecolor='black')
plt.xlabel("x")
plt.ylabel("# entries")
plt.legend(loc='best')


#%% c3
"""
def likelihood(x,alpha,beta):
    #we will transform this into a PDF right away...
    pdf = np.e**(-((x-alpha)/(2*beta))**2)
    #...remembering the PDF should be normalized! this is the definite integral over range -1..1
    norm = 1.25331*beta*erf(0.707107*alpha/beta)-1.25331*beta*erf(0.707107*(alpha-2.5)/beta)
    #return the evaluated PDF only inside -1..1 range, and 0 outside
    return np.where(x >=0, pdf/norm, 0.)

def log_likelihood(x,alpha,beta):
    #The log-likelihood to get all of the x-points (together) given our custom PDF with parameters alpha and beta.
    return np.sum(np.log(likelihood(x,alpha,beta)),axis=1)
lambda_neg_LLH = lambda params: -log_likelihood(c4,
                                np.expand_dims([params[0]],axis=1),np.expand_dims([params[1]],axis=1))
#notice some broadcasting tricks above, which let us operate on vector-like x-samples as well as vector-like
#parameters alpha and beta. This is no different from what we've done in Exercise 1.

res = minimize(lambda_neg_LLH,x0=[1.13,0.1],bounds=((-10,10),(-10,10)),method='L-BFGS-B')

print(res)

p=res.x
errors=np.sqrt(np.diag(res.hess_inv.todense()))

x2=np.linspace(min(c4),max(c4), 100)


plt.figure()
plt.plot(x2, likelihood(x2, p[0], p[1]),label="fit")




plt.hist(c4,bins=np.linspace(min(c4),max(c4),20),density=True,label = 'samples',  color='b', edgecolor='black')

"""
#%%

def likelihood(x,alpha, beta):
    #we will transform this into a PDF right away...
    pdf = np.e**(-((x-alpha)/(2*beta))**2)
    #...remembering the PDF should be normalized! this is the definite integral over range -1..1
    norm = 1.25331*beta*erf(0.707107*alpha/beta)-1.25331*beta*erf(0.707107*(alpha-2.5)/beta)
    #norm=np.sqrt(2*np.pi*beta*beta)
    
    #return the evaluated PDF only inside -1..1 range, and 0 outside
    return np.where(x >=0, pdf/norm, 0.)

def log_likelihood(x,alpha,beta):
    #The log-likelihood to get all of the x-points (together) given our custom PDF with parameters alpha and beta.
    return np.sum(np.log(likelihood(x,alpha,beta)),axis=1)
lambda_neg_LLH = lambda params: -log_likelihood(c4,
                                np.expand_dims([params[0]],axis=1),np.expand_dims([params[1]],axis=1))
#notice some broadcasting tricks above, which let us operate on vector-like x-samples as well as vector-like
#parameters alpha and beta. This is no different from what we've done in Exercise 1.

res = minimize(lambda_neg_LLH,x0=[1.13,0.1],bounds=((-10,10),(-10,10)),method='L-BFGS-B')

print(res)

p=res.x
errors=np.sqrt(np.diag(res.hess_inv.todense()))



print(p,errors)
x2=np.linspace(min(c4),max(c4), 100)


plt.figure()
plt.plot(x2, likelihood(x2, p[0], p[1]), label="fit")




plt.hist(c4,bins=np.linspace(min(c4),max(c4),20),density=True,label = 'samples',  color='b', edgecolor='black')
plt.xlabel("x")
plt.ylabel("# entries")
plt.legend(loc='best')
