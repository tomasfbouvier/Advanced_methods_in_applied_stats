#!/usr/local/bin/python3

import healpy as H
import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":

	my_cmap = cm.Greys
	my_cmap.set_under("w")
	
	rc('text', usetex=True)
	rc('font',**{'family':'serif','serif':['Palatino']})
	
	# resolution of output maps
	nside = 32
	npix = H.nside2npix(nside)
		
	map = np.zeros(npix,dtype=np.int)
	
	RUNS = 10000
	Ntot = 100
	
	KS = []
	
	Nbins = 10000
	
	cosx = np.arange(0,Nbins)/(1.0*Nbins)*2.0-1.0
	sumyiso = 1./2.*(1.0-cosx)
	
	for run in range(0,RUNS) :
	
		print(run)
		
		theta = []
		phi = []
		nx = []
		ny = []
		nz = []
	
		for i in range(0,Ntot) :
			phitemp = np.random.rand()*2*np.pi
			x = np.random.rand()
		
			# dipole anisotropy :
			ani = 0.9
			costhetatemp = (1.-np.sqrt(1.0+2.*ani+ani**2-4.*ani*x))/ani
		
			#isotropic :
			#costhetatemp = 2.0*x-1.0
		
			thetatemp = np.arccos(costhetatemp)
		
			nxtemp = np.cos(phitemp)*np.sin(thetatemp)
			nytemp = np.sin(phitemp)*np.sin(thetatemp)
			nztemp = np.cos(thetatemp)
		
			phi.append(phitemp)
			theta.append(thetatemp)
			nx.append(nxtemp)
			ny.append(nytemp)
			nz.append(nztemp)
	
		cosphi = []
		for i in range(0,Ntot) :
			for j in range(0,i) :
				cosphitemp = nx[i]*nx[j] + ny[i]*ny[j] + nz[i]*nz[j]
				cosphi.append(cosphitemp)
	
		totpairs = len(cosphi)
		
		y = np.zeros(Nbins,dtype=np.int)
		
		for i in range(0,totpairs) :
			bin = np.int((cosphi[i]+1.)/2.*Nbins)
			y[bin] += 1
	
		sumy = np.zeros(Nbins,dtype=np.int)
	
		sumy[Nbins-1] = y[0]
		for i in range(1,Nbins) :
			sumy[Nbins-i-1] = sumy[Nbins-i] + y[i]
			
		sumy = sumy/(1.*totpairs)
	
		KS.append(max(np.abs(sumy-sumyiso)))
		
	pickle.dump(KS,open('KSdipole_Ntot100_Nsample10000.dat', "bw" ))	
	
	show()
	