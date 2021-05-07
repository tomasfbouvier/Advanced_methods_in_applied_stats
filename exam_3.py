# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:51:12 2021

@author: To+
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.interpolate as interp
from scipy import stats

#%%
class island:
    r= 5 #Km
    def __init__(self,x0,y0):
        self.M_T=0.
        self.day=0
        
        self.crabs=[]
        n_crabs=len(x0)
        for i in range(n_crabs):
            one_crab=crab(x0[i], y0[i])

            self.crabs.append(one_crab)
            self.M_T+=one_crab.M
        return None
    
    def move_crabs(self):
        for each_crab in self.crabs:
            each_crab.move()
        return
    
    def carnage(self):
        for each_crab in self.crabs:
            for other_crab in self.crabs[self.crabs.index(each_crab)+1:len(self.crabs)]:
                #other_crab=self.crabs[j]
                d2= (each_crab.x-other_crab.x)*(each_crab.x-other_crab.x)+(each_crab.y-other_crab.y)*(each_crab.y-other_crab.y)
                if(d2<=0.175**2 and each_crab.alive==True and other_crab.alive==True and each_crab.M>=other_crab.M and each_crab!=other_crab):
                    each_crab.battle(other_crab)
        return 
    
    def remove_crabs_and_total_mass(self):
        self.M_T=0
        for each_crab in self.crabs:
            if(each_crab.alive==False):
                self.crabs.remove(each_crab)
            else:
                self.M_T+=each_crab.M
        return

    def plot_crabs(self):
        x=[crab.x for crab in self.crabs]
        y=[crab.y for crab in self.crabs]
        
        plt.plot(x,y, 'b.')
        
    def move_1_day(self, plotter=False, verbosity=False):
        self.day+=1
        self.move_crabs()
        self.carnage()
        self.remove_crabs_and_total_mass()
        if(plotter==True):
            self.plot_crabs()
        if(verbosity==True):
            print("day: ", self.day)
            print("crabs alive: ", len(self.crabs))
            print("total mass:", self.M_T)
        return
        
        
class crab():

    def __init__(self, x0,y0):
        self.x=x0
        self.y=y0
        self.r=np.sqrt(x*x+y*y)
        self.l=0.
        self.M=1. #Kg
        self.alive=True
        self.out=False #Only for experiment 2
        return None
    
    def move(self):
        dl=0.2
        theta=2*np.pi*np.random.rand()
        x_aux=self.x+dl*np.cos(theta)
        y_aux=self.y+dl*np.sin(theta)
        #self.r= np.sqrt(self.x*self.x+self.y*self.y)
        if(x_aux*x_aux+y_aux*y_aux<=25.):
            self.x=x_aux
            self.y=y_aux
            self.l+=np.linalg.norm(dl)
            self.r= np.sqrt(self.x*self.x+self.y*self.y)
            self.out=False
        else:
            self.out=True
        return()
    
    def battle(self, other_crab):
        M_large2=self.M*self.M; M_small2=other_crab.M*other_crab.M
        p= M_small2/(M_small2+M_large2)
        if(np.random.rand()>p):
            other_crab.alive=False
            self.M+=other_crab.M
        return()
            

x,y= np.loadtxt("crab_init.txt", unpack=True)

#plt.figure()

#circle1 = plt.Circle((0, 0), 5., color='r', alpha=0.5)
#plt.gca().add_patch(circle1)
#plt.plot(x, y, 'b.')



#this_island.plot_crabs()

#%%
masses=[]; alives=[]

for i in range(500):
    print(i)
    this_island=island(x, y)

    for days in range(200):
        this_island.move_1_day(verbosity=False, plotter=False)
    
    alives.append(len(this_island.crabs))
    masses.append(max([crab.M for crab in this_island.crabs]))
    del(this_island)

print(np.mean(masses))
print(np.std(masses))


print(np.mean(alives))
print(np.std(alives))



#%%
days=[]
for i in range(500):
    print(i)

    this_island=island(x,y)
    while(len(this_island.crabs)>=10):
        this_island.move_1_day(verbosity=False, plotter=False)
    
    days.append(this_island.day)
    del(this_island)



plt.hist(days,30)









#%%

y,x= np.histogram(days, 20)
plt.hist(days,20, density=True, color='b', edgecolor='black', label="samples")
#plt.plot(x[:-1],y/sum(y*(x[1]-x[0])))

#f=interp.CubicSpline(x[:-1],y)

f= stats.gaussian_kde(days)

x_r= np.arange(min(x), max(x[:-1])-1., 1.)

plt.plot(x_r, f(x_r), label="gaussian KDE")

for i in range(0, len(x_r), 1):
    p=sum(f(x_r[0:i]))/sum(f(x_r))
    if p>((1.-0.6827)/2.):
        sigma_low=x_r[i-1]
        break
print("sigma low", sigma_low)

for i in range(len(x_r),0, -1):
    p=sum(f(x_r[i:len(x_r)]))/sum(f(x_r))
    if p>((1.-0.6827)/2.):
        sigma_high=x_r[i+1]
        break
print("sigma high", sigma_high)


plt.xlabel("days")
plt.ylabel("P(days)")
plt.legend(loc="best")

#%%

"""    
plt.figure()
plt.hist(alives)
plt.figure()
plt.hist(masses)
"""




this_crab=crab(3.6,-2.)

def experiment(this_crab):
    i=0
    xs=[this_crab.x]; ys=[this_crab.y]
    while(i<200 ):
        i+=1
        this_crab.move()
        xs.append(this_crab.x)
        ys.append(this_crab.y)
    return (xs, ys)
plt.figure()

circle1 = plt.Circle((0, 0), 5., color='r', alpha=0.5)
plt.gca().add_patch(circle1)
xs,ys=experiment(this_crab)
plt.plot(xs, ys, 'b', label='crab random path')
plt.plot(3.6,-2., 'g.', markersize=9, label='initial position')

plt.xlim(-5.1,5.1)
plt.ylim(-5.1,5.1)
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.legend(loc="best")
#%%
def experiment2(this_crab):
    #this_crab.l=0
    while(this_crab.out!=True):
        this_crab.move()
    return(this_crab.l)


ls=[]
for i in range(500):
    #print(i)
    this_crab=crab(3.6,-2.)
    ls.append(experiment2(this_crab))

plt.figure()
plt.hist(ls,20, density=True,  color='b', edgecolor='black')
plt.xlabel("l before reaching the borders ")
plt.ylabel("P(l)")

