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
	
	Ntot = 10
	
	H.mollview(map,title=r'simulation with dipole anisotropy (' + str(Ntot) + ' events)',min=0,max=1,cbar=False,cmap=my_cmap)
	H.graticule()
	
	theta = []
	phi = []
	nx = []
	ny = []
	nz = []
	
	for i in range(0,Ntot) :
		phitemp = np.random.rand()*2*np.pi
		x = np.random.rand()
		
		# dipole anisotropy :
		ani = 0.9 # parameter between -1.0 and 1.0
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
		
		H.projscatter(thetatemp,phitemp,marker='*',s=50,color='black',linewidth=0)
	
	savefig("map10dipole.pdf",bbox_inches = 'tight',cmap=my_cmap);
	#show()
	close()
	
	cosphi = []
	for i in range(0,Ntot) :
		for j in range(0,i) :
			cosphitemp = nx[i]*nx[j] + ny[i]*ny[j] + nz[i]*nz[j]
			cosphi.append(cosphitemp)
	
	totpairs = len(cosphi)
	
	Nbins = 1000
	
	y = np.zeros(Nbins,dtype=np.int)
	cosx = np.arange(0,Nbins)/(1.0*Nbins)*2.0-1.0
	x = np.arccos(cosx)
	
	for i in range(0,totpairs) :
		bin = np.int((cosphi[i]+1.)/2.*Nbins)
		y[bin] += 1
	
	sumy = np.zeros(Nbins,dtype=np.int)
	
	
	for i in range(0,Nbins) :
		for j in range(i+1,Nbins) :
			sumy[i] += y[j]
	
	sumy = sumy/(1.*totpairs)
	sumyiso = 1./2.*(1.0-cosx)
	
	KS = np.max(np.abs(sumy-sumyiso))
	KSindex = np.argmax(np.abs(sumy-sumyiso))
	
	fig = figure(figsize=(6, 6))
	ax = fig.add_subplot(1,1,1)
	
	title(r'simulation (' + str(Ntot) + ' events)',fontsize=16,y=1.05)
	
	xlabel(r'$\cos\varphi$',fontsize=16)
	ylabel(r'cumulative auto-correlation $\mathcal{C}(\varphi)$',fontsize=16)
	
	ax.tick_params(axis='both',which='both',direction='in')
	
	for tick in ax.xaxis.get_major_ticks() :
		tick.label.set_fontsize(16) 
   	
	for tick in ax.yaxis.get_major_ticks() :
		tick.label.set_fontsize(16)  
	
	xlim([-1,1])
	ylim([0,1])
	
	plot(cosx,sumy,drawstyle='steps-mid',label=r'dipole')
	plot(cosx,sumyiso,drawstyle='steps-mid',label=r'isotropic')
	plot(np.array([cosx[KSindex],cosx[KSindex]]),np.array([sumy[KSindex],sumyiso[KSindex]]),color='red',linewidth=2.0,label='$KS = ' + '{:.2f}'.format(KS) + '$')
	
	leg = plt.legend(bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0.,fancybox=False,framealpha=0.0,frameon=True,numpoints=1, scatterpoints = 1,handlelength=1)
	for t in leg.get_texts() :
		t.set_fontsize(14)
	
	#show()
	savefig("C10dipole.pdf",bbox_inches = 'tight');
	close()
	