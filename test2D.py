#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:56:56 2025

@author: ogurcan
"""

import numpy as np
import numpy as xp
from prrl import cvpgrid,partlist,conv,conv_singlefft
import matplotlib.pylab as plt

xp.random.seed(42)
Nrx,Nlx=64,8
Nry,Nly=64,8

gr=cvpgrid(Nrx,Nry,Nlx,Nly)
kx,ky=xp.ix_(gr.knx,gr.kny)

def hermsym(phik,gr):
    phik[0,0]=phik[0,0].real
    phik[-1:-gr.Nrhx-gr.Nlx:-1,0]=phik[1:gr.Nrhx+gr.Nlx,0].conj()

gk=xp.exp(1j*2*xp.pi*xp.random.rand(*gr.shp))*xp.random.rand(*gr.shp)
hk=xp.exp(1j*2*xp.pi*xp.random.rand(*gr.shp))*xp.random.rand(*gr.shp)
hermsym(gk, gr)
hermsym(hk, gr)

pl=partlist(gr,fft_type='numpy')

fk=conv(gk,hk,pl)
fk2=conv_singlefft(gk,hk,gr)

plt.figure(figsize=(5,4),dpi=200)
plt.subplot2grid((2,2), (0,0))
plt.pcolormesh(np.log10(np.abs(gk)),rasterized=True,cmap='seismic',vmin=-3,vmax=2)
plt.title("$\log|g_k|$")
plt.colorbar()
plt.subplot2grid((2,2), (0,1))
plt.pcolormesh(np.log10(np.abs(hk)),rasterized=True,cmap='seismic',vmin=-3,vmax=2)
plt.gca().set_yticklabels([])
plt.colorbar()
plt.title("$\log|h_k|$")
plt.subplot2grid((2,2), (1,0))
plt.pcolormesh(np.log10(np.abs(fk)),rasterized=True,cmap='seismic',vmin=-2,vmax=4)
plt.colorbar()
plt.title("$\log|f_k|$")
plt.subplot2grid((2,2), (1,1))
plt.pcolormesh(np.log10(np.abs(fk-fk2)),rasterized=True,cmap='seismic')
plt.gca().set_yticklabels([])
plt.colorbar()
plt.title("$\log|f_k-f_k^{fft}|$")

plt.tight_layout()