# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:46:18 2021

@author: To+
"""


import numpy as np
from scipy.stats import poisson, norm
from statistics import mean
from scipy import stats

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class Alexa():
    
    def __init__(self):
        
        self.type_A=412
        self.type_B=0
        
        self.time= 8
        self.money=0    #Dolars
        
        self.traded=False
        self.trading_time=None
        
        return None
    
    def lose_fish_A(self):
        
        dfish=poisson.rvs(23)
        self.type_A-=dfish
        return None
    
    def lose_fish_B(self):
        dfish=poisson.rvs(23)
        self.type_B-=dfish
        return None
    
    def sell_fish_A(self):
        self.type_A-=1
        self.money+=6 
        
    def sell_fish_B(self):
        self.type_B-=1
        self.money+=10
    
    def wanna_trade_fish(self):
        p= norm(240, 10).pdf(self.type_A)
        #print(p)
        if(np.random.rand()<p):
            self.traded=True
            self.trading_time=self.time
        return None
    
    def trade_fish(self):
        self.type_A=0
        self.type_B=120
        
        return None
    
    def day(self):
        max_time=18
        dt= 0.01;c=0
        
        while self.time<= max_time:
            #print(self.time, self.type_A)
            self.time+=dt
            c+=1
            if(c%50==0):
                self.lose_fish_A()
                for i in range(5):
                    self.sell_fish_A()
            self.wanna_trade_fish()
            if(self.traded==True):
                self.trade_fish()
                break
        dt=1
        while(self.time<= max_time and self.type_B>0):
            #print(self.type_B)
            self.time+=dt
            self.lose_fish_B()
            for i in range(3):
                self.sell_fish_B()
            
        self.money+=self.type_B*4

        return None
            
times=[]; moneys=[]
for i in range(50):
    print(i)
    this_alexa=Alexa()

    this_alexa.day()
    times.append(this_alexa.trading_time)   
    moneys.append(this_alexa.money)




times=[d for d in times if d is not None]

f= stats.gaussian_kde(times)

x_r= np.arange(min(times), max(times), .01)

plt.plot(x_r, f(x_r), label="gaussian KDE")
plt.hist(times,20, density=True  ,color='b', edgecolor='black', label='samples')

mean_time=np.mean(times)
print("mean", mean_time)
plt.xlabel("hour of the exchange")
plt.ylabel("P(exchange)")
plt.legend(loc='best')

plt.figure()


plt.hist(moneys,30, density=True  ,color='b', edgecolor='black')
plt.xlabel("money ($)")
plt.ylabel("P(money)")
np.savetxt("exam_4.txt", moneys)

