#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:56:56 2025

@author: ogurcan
"""

import numpy as np
import numpy as xp
from prrl3D import cvpgrid,partlist,conv,conv_singlefft
import matplotlib.pylab as plt

xp.random.seed(42)
Nrx,Nlx=64,4
Nry,Nly=64,4
Nrz,Nlz=64,4

gr=cvpgrid(Nrx,Nry,Nrz,Nlx,Nly,Nlz)
kx,ky,kz=xp.ix_(gr.knx,gr.kny,gr.knz)

def hermsym(phik,gr):
    phik[0,0,0]=phik[0,0,0].real
    phik[-1:-gr.Nrhx-gr.Nlx:-1,0,0]=phik[1:gr.Nrhx+gr.Nlx,0,0].conj()
    phik[0,-1:-gr.Nrhy-gr.Nly:-1,0]=phik[0,1:gr.Nrhy+gr.Nly,0].conj()
    phik[-1:-gr.Nrhx-gr.Nlx:-1,1:,0]=phik[1:gr.Nrhx+gr.Nlx,:0:-1,0].conj()

gk=xp.exp(1j*2*xp.pi*xp.random.rand(*gr.shp))*xp.random.rand(*gr.shp)
hk=xp.exp(1j*2*xp.pi*xp.random.rand(*gr.shp))*xp.random.rand(*gr.shp)
hermsym(gk, gr)
hermsym(hk, gr)

pl=partlist(gr,fft_type='numpy')

fk=conv(gk,hk,pl)
fk2=conv_singlefft(gk,hk,gr)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(2,2,1,projection="3d")
x,y,z=np.mgrid[0:gr.knx.size,0:gr.kny.size,0:gr.knz.size]
sc=ax.scatter(x,y,z,c=np.log10(np.abs(gk)),alpha=0.05,cmap='seismic',vmin=-3,vmax=2,marker='s',s=2)
cb=plt.colorbar(sc,shrink=0.5)
cb.solids.set(alpha=1)
plt.title("$\log|g_k|$")
ax = fig.add_subplot(2,2,2,projection="3d")
sc=ax.scatter(x,y,z,c=np.log10(np.abs(hk)),alpha=0.05,cmap='seismic',vmin=-3,vmax=2,marker='s',s=2)
cb=plt.colorbar(sc,shrink=0.5)
cb.solids.set(alpha=1)
plt.title("$\log|h_k|$")
ax = fig.add_subplot(2,2,3,projection="3d")
sc=ax.scatter(x,y,z,c=np.log10(np.abs(fk)),alpha=0.05,cmap='seismic',vmin=-2,vmax=4,marker='s',s=2)
cb=plt.colorbar(sc,shrink=0.5)
cb.solids.set(alpha=1)
plt.title("$\log|f_k|$")
ax = fig.add_subplot(2,2,4,projection="3d")
sc=ax.scatter(x,y,z,c=np.log10(np.abs(fk-fk2)),alpha=0.05,cmap='seismic',marker='s',s=2)
cb=plt.colorbar(sc,shrink=0.5)
cb.solids.set(alpha=1)
plt.title("$\log|f_k-f_k^{fft}|$")
plt.tight_layout()
