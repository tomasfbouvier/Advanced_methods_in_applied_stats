# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 20:56:35 2021

@author: To+
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

data=np.array([[ 203.41 , -89.37 ],[ 203.435, -94.88 ],
               [ 203.46 , -101.25 ],[ 203.484, -106.52 ],
               [ 203.509, -108.66 ],[ 203.534, -114.25 ],
               [ 203.558, -114.3 ],[ 203.583, -117.66 ],[ 203.608, -122.45 ]])

def find_lower_nearest(array, value):
    if(value>min(array) and value<max(array)):
        array = np.asarray(array[array<value])
        idx = (np.abs(array - value)).argmin()
        return idx
    else:
        print("value out of interpolation range")
        return None
    
def linear_spline(array):
    
    def f_aux(x):
        i=find_lower_nearest(array[:,0],x)
        I_x= array[i,1]+ (array[i+1,1]-array[i,1])*(x-array[i,0])/(array[i+1,0]-array[i,0])
        return I_x
    return f_aux

"""

def cubic_spline(data):
    def f_aux(x):
        
        
        i=find_lower_nearest(data[:,0],x);i+=2
        def A_row(x,y, i):
            return (np.array([x[i-1]-x[i], 2*(x[i-1]-x[i+1]),x[i]-x[i+1]]), 6*((y[i-1]-y[i])/(x[i-1]-x[i])-(y[i]-y[i+1])/(x[i]-x[i+1])))
        
        def f(x, xs,y, k, i):
            f=0
            f+=k[i]/6*(1/(x[i]-x[i+1])*(xs-x[i+1])**3-(xs-x[i+1])*(x[i]-x[i+1]))
            f+=-k[i+1]/6*(1/(x[i]-x[i+1])*(xs-x[i])**3-(xs-x[i])*(x[i]-x[i+1]))
            f+= (y[i]*(xs-x[i+1])-y[i+1]*(xs-x[i]))/(x[i]-x[i+1])
            
            return f
        
        n=np.shape(data)[0]
        k=np.zeros(n); 
        A=np.zeros((np.shape(data)[0]-2,np.shape(data)[0]-2))
        for i in range(1, n-1):
            
            a,b=A_row(data[:,0], data[:,1], i)
            #print("asd",a[i: np.shape(A)[0]])
            for j in range(i-1,n-3):  

                A[i-1, j]= a[j+1]

        
        #I tried for hours to implement a cubic spline but algebra drives me crazy
        
        A=np.zeros([3,3]); b=np.zeros(3)
        print(i)
        print(data[i-2,0])
        
        A[0,0]=2*(data[i-2,0]-data[i,0]); A[0,1]=(data[i-1,0]-data[i,0]); A[0,2]=0; 
        #A[1,0]=data[i-1]
        A[1,0]=(data[i-1,0]-data[i,0]); A[1,1]=2*(data[i-1,0]-data[i+1,0]); A[1,2]=(data[i,0]-data[i+1,0])
        A[2,0]=0;  A[2,1]=(data[i+1,0]-data[i+2,0]); A[2,2]=2*(data[i,0]-data[i+2,0])
    
        b[0]= 6*((data[i-2,1]-data[i-1,1])/(data[i-2,0]-data[i-1,0])-(data[i-1,1]-data[i,1])/(data[i-1,0]-data[i,0]))
        b[1]= 6*((data[i-1,1]-data[i,1])/(data[i-1,0]-data[i,0])-(data[i,1]-data[i+1,1])/(data[i,0]-data[i+1,0]))
        b[2]= 6*((data[i,1]-data[i+1,1])/(data[i,0]-data[i+1,0])-(data[i+1,1]-data[i+2,1])/(data[i+1,0]-data[i+2,0]))
        
        print(A)
        k=np.linalg.solve(A,b)
        print(b)
        print(k)
        
        
        
        
        
        I_x= k[0]/6*(1/(data[i,0]-data[i+1,0])*(x-data[i+1,0])**3-(x-data[i+1,0])*(data[i,0]-data[i+1,0]))
        I_x+= -k[1]/6*(1/(data[i,0]-data[i+1,0])*(x-data[i, 0])**3-(x-data[i,0])*(data[i,0]-data[i+1,0]))
        I_x+= 1/(data[i,0]-data[i+1,0])*(data[i, 1]*(x-data[i+1,0])-data[i+1,1]*(x-data[i,0]))
        return I_x


    return f_aux
#data=np.array([[1.,0.],[2.,1.], [3.,0.],[4.,1.], [5.,0.]])

#print(cubic_spline(data)(1.5))

    """

x=203.570

print("linear estimation:", linear_spline(data)(x))
print("cubic estimation:", sci.interpolate.CubicSpline(data[:,0],data[:,1])(x))



value = 203.5
xs=np.linspace(min(data[:,0])+0.0001, max(data[:,0])-0.00001, 10000)

ys=sci.interpolate.CubicSpline(data[:,0],data[:,1])(xs)

ys_lin=[]
for x in xs:
    ys_lin.append(linear_spline(data)(x))

plt.figure()

plt.xlabel("time (sol)")
plt.ylabel("T (ºC)")


plt.plot(xs,ys_lin, 'y', label='linear Spline')



plt.plot(xs, ys, 'b.',markersize=0.5, label="Cubic Spline")

plt.plot(data[:,0],data[:, 1],'r.', markersize=10)
plt.legend(loc='best')

slopes=[]
for i in range(len(xs)-1):
    slopes.append((ys[i+1]-ys[i-1])/(xs[i+1]-xs[i-1]))





plt.figure()
plt.hist(slopes, 100, color='b', edgecolor='black', density=True)
plt.xlabel("dT/dt (ºC/sol)")
plt.ylabel("P(dT/dt)")

plt.figure()
plt.xlabel("time (sol)")
plt.ylabel("dT/dt (ºC/sol)")

plt.plot(xs[1:-1],slopes[1:])

slopes=slopes
p_big_T_fluct= len([a for a in slopes if abs(a)>=0.09/0.0004])/len(slopes)

print("probability of high fluctuations= ", p_big_T_fluct*100, " %") 




