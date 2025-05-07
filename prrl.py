#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:04:33 2025

@author: ogurcan
"""

import numpy as np
import numpy as xp
from concurrent.futures import ThreadPoolExecutor
#import cupy as xp
PRRLArray=np.ndarray

ILS=np.array([1,3,4,7,11])
class cvpgrid:
    def __init__(self,Nrx,Nry,Nlx,Nly):
        Nrhx,Nrhy=int(Nrx/2),int(Nry/2)
        N=2*(Nrhx+Nlx-1)*(Nrhy+Nly)
        knx=np.arange(0,Nrhx)
        kny=np.arange(0,Nrhy)
        for l in range(Nlx):
            knx=np.append(knx,knx[-2]+knx[-1])
        for l in range(Nly):
            kny=np.append(kny,kny[-2]+kny[-1])
        knx=np.hstack([knx,-knx[-1:0:-1]])
        self.Nrhx,self.Nrhy,self.Nlx,self.Nly=Nrhx,Nrhy,Nlx,Nly
        self.N,self.knx,self.kny=N,knx,kny
        self.shp=(knx.size,kny.size)

class partlist:
    def __init__(self,gr,fft_type='numpy'):
        Nrhx,Nrhy,Nlx,Nly=gr.Nrhx,gr.Nrhy,gr.Nlx,gr.Nly
        N=(Nlx+4)*(Nly+4)
        parts=[]
        for n in range(-1,Nlx+3):
            for l in range(-1,Nly+3):
                parts.append(part(n,l,Nrhx,Nrhy,Nlx,Nly,fft_type))
        self.N=N
        self.parts=parts

class part:
    def __init__(self,n,l,Nrhx,Nrhy,Nlx,Nly,fft_type):
        self.partid=(n,l)
        Npx,Npy,self.input,self.mapping,self.valid,self.output = partget(n,l,Nrhx,Nrhy,Nlx,Nly)
        shp=Npx,Npy
        self.gkp=xp.zeros((shp[0],int(shp[1]/2)+1),complex)
        self.hkp=xp.zeros((shp[0],int(shp[1]/2)+1),complex)
        self.fkp=xp.zeros((shp[0],int(shp[1]/2)+1),complex)
        if fft_type=='numpy':
            self.rft = np.fft.rfft2
            self.irft = lambda x : np.fft.irfft2(x,norm="forward")
            self.fftconv = fftconv_numpy
        elif fft_type=='scipy':
            from scipy import fft
            self.rft = fft.rfft2
            self.irft = lambda x : fft.irfft2(x,norm="forward")
            self.fftconv = fftconv_numpy
        if fft_type=='cupy':
            self.rft = xp.fft.rfft2
            self.irft = lambda x : xp.fft.irfft2(x,norm="forward")
            self.fftconv = fftconv_numpy
        elif fft_type=="pyfftw":
            import pyfftw
            self.gkp=pyfftw.empty_aligned((shp[0],int(shp[1]/2)+1),complex)
            self.hkp=pyfftw.empty_aligned((shp[0],int(shp[1]/2)+1),complex)
            self.fp=pyfftw.empty_aligned(shp)
            self.rftfp=pyfftw.builders.rfft2(self.fp,norm='backward')
            self.irftgk=pyfftw.builders.irfft2(self.gkp,norm='forward')
            self.irfthk=pyfftw.builders.irfft2(self.hkp,norm='forward')
            self.fftconv = fftconv_pyfftw
        self.shp=shp

def partget(n,l,Nrhx,Nrhy,Nlx,Nly):
    if(n==-1):
        Npx = Nrhx*3
        input_x=np.r_[0:Nrhx,-Nrhx+1:0]
        mapping_x=input_x
        valid_x=input_x
        output_x=input_x
    elif(n==0):
        Npx=int(np.ceil((Nlx)/2)*6)
        input_x=np.r_[Nrhx:Nrhx+Nlx,-Nrhx-Nlx+1:-Nrhx+1]
        mapping_x=np.r_[1:Nlx+1,-1-Nlx+1:0]
        valid_x=[0]
        output_x=[0]
    else:
        nmn=np.maximum(1,n-2)
        nmx=np.minimum(n+3,Nlx+3)
        numn=nmx-nmn
        n0=ILS[np.minimum(n-1,2)]
        Npx=int(np.ceil((ILS[numn-1]+1)/2))*6
        input_x=np.r_[0,Nrhx-3+np.r_[nmn:nmx],-Nrhx+3-np.r_[nmx-1:nmn-1:-1]]
        mapping_x=np.r_[0,ILS[:numn],-ILS[numn-1::-1]]
        valid_x=[n0,-n0]
        output_x=[Nrhx-3+n,-Nrhx+3-n]
        if(n<3):
            input_x=input_x[1:]
            mapping_x=mapping_x[1:]
    if(l==-1):
        Npy = Nrhy*3
        input_y=np.r_[:Nrhy]
        mapping_y=input_y
        valid_y=input_y
        output_y=input_y
    elif(l==0):
        Npy=int(np.ceil((Nly)/2)*6)
        input_y=np.r_[Nrhy:Nrhy+Nly]
        mapping_y=np.r_[1:Nly+1]
        valid_y=[0]
        output_y=[0]
    else:
        lmn=np.maximum(1,l-2)
        lmx=np.minimum(l+3,Nly+3)
        numl=lmx-lmn
        l0=ILS[np.minimum(l-1,2)]
        Npy=int(np.ceil((ILS[numl-1]+1)/2))*6
        input_y=np.r_[0,Nrhy-3+np.r_[lmn:lmx]]
        mapping_y=np.r_[0,ILS[:numl]]
        valid_y=[l0]
        output_y=[Nrhy-3+l]
        if(l<3):
            input_y=input_y[1:]
            mapping_y=mapping_y[1:]
    input_xy=xp.ix_(input_x,input_y)
    mapping_xy=xp.ix_(mapping_x,mapping_y)
    valid_xy=xp.ix_(valid_x,valid_y)
    output_xy=xp.ix_(output_x,output_y)
    return Npx,Npy,input_xy,mapping_xy,valid_xy,output_xy

def fftconv_numpy(gk,hk,part):
    part.gkp[part.mapping]=gk[part.input]
    part.hkp[part.mapping]=hk[part.input]
    part.fkp=part.rft(part.irft(part.gkp)*part.irft(part.hkp))
    return part.output,part.fkp[part.valid]/part.shp[0]/part.shp[1]

def fftconv_pyfftw(gk,hk,part):
    part.gkp.fill(0)
    part.hkp.fill(0)
    part.gkp[part.mapping]=gk[part.input]
    part.hkp[part.mapping]=hk[part.input]
    gp=part.irftgk()
    hp=part.irfthk()
    part.fp[:]=gp*hp
    part.fkp=part.rftfp()
    return part.output,part.fkp[part.valid]/part.shp[0]/part.shp[1]

def conv(gk,hk,pl):
    fk=np.zeros_like(gk)
    ff = lambda p : p.fftconv(gk,hk,p)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(ff, pl.parts))
    for output,f in results:
        fk[output]+=f
    return fk

def conv_serial(gk,hk,pl):
    fk=np.zeros_like(gk)
    ff = lambda p : p.fftconv(gk,hk,p)
    results = list(map(ff,pl.parts))
    for output,f in results:
        fk[output]+=f
    return fk

def conv_singlefft(gk,hk,gr):
    Nx,Ny=int(np.ceil((gr.knx[int(gr.knx.size/2)]+1)/2))*4,int(np.ceil((gr.kny[-1]+1)/2))*4
    Npx,Npy=int(Nx/2)*3,int(Ny/2)*3
    gkp=np.zeros((Npx,int(Npy/2)+1),dtype=complex)
    hkp=np.zeros_like(gkp)
    ij=np.ix_(gr.knx,gr.kny)
    gkp[ij]=gk
    hkp[ij]=hk
    fkp=np.fft.rfft2(np.fft.irfft2(gkp,norm='forward')*np.fft.irfft2(hkp,norm='forward'),norm='backward')/Npx/Npy
    return fkp[ij]
