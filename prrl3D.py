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
    def __init__(self,Nrx,Nry,Nrz,Nlx,Nly,Nlz):
        Nrhx,Nrhy,Nrhz=int(Nrx/2),int(Nry/2),int(Nrz/2)
        knx=np.arange(0,Nrhx)
        kny=np.arange(0,Nrhy)
        knz=np.arange(0,Nrhz)
        for l in range(Nlx):
            knx=np.append(knx,knx[-2]+knx[-1])
        for l in range(Nly):
            kny=np.append(kny,kny[-2]+kny[-1])
        for l in range(Nlz):
            knz=np.append(knz,knz[-2]+knz[-1])
        knx=np.hstack([knx,-knx[-1:0:-1]])
        kny=np.hstack([kny,-kny[-1:0:-1]])
        self.Nrhx,self.Nrhy,self.Nrhz,self.Nlx,self.Nly,self.Nlz=Nrhx,Nrhy,Nrhz,Nlx,Nly,Nlz
        self.knx,self.kny,self.knz=knx,kny,knz
        self.shp=(knx.size,kny.size,knz.size)

class partlist:
    def __init__(self,gr,fft_type='numpy'):
        Nrhx,Nrhy,Nrhz,Nlx,Nly,Nlz=gr.Nrhx,gr.Nrhy,gr.Nrhz,gr.Nlx,gr.Nly,gr.Nlz
        N=(Nlx+4)*(Nly+4)*(Nlz+4)
        parts=[]
        
        for i in range(-1,Nlx+3):
            for j in range(-1,Nly+3):
                for k in range(-1,Nlz+3):
                    parts.append(part(i,j,k,Nrhx,Nrhy,Nrhz,Nlx,Nly,Nlz,fft_type))
        self.N=N
        self.parts=parts

class part:
    def __init__(self,i,j,k,Nrhx,Nrhy,Nrhz,Nlx,Nly,Nlz,fft_type):
        self.partid=(i,j,k)
        Npx,Npy,Npz,self.input,self.mapping,self.valid,self.output = partget(i,j,k,Nrhx,Nrhy,Nrhz,Nlx,Nly,Nlz)
        shp=Npx,Npy,Npz
        self.gkp=xp.zeros((shp[0],shp[1],int(shp[2]/2)+1),complex)
        self.hkp=xp.zeros_like(self.gkp)
        self.fkp=xp.zeros_like(self.gkp)
        if fft_type=='numpy':
            self.rft = np.fft.rfftn
            self.irft = lambda x : np.fft.irfftn(x,norm="forward")
            self.fftconv = fftconv_numpy
        elif fft_type=='scipy':
            from scipy import fft
            self.rft = fft.rfftn
            self.irft = lambda x : fft.irfftn(x,norm="forward")
            self.fftconv = fftconv_numpy
        if fft_type=='cupy':
            self.rft = xp.fft.rfftn
            self.irft = lambda x : xp.fft.irfftn(x,norm="forward")
            self.fftconv = fftconv_numpy
        elif fft_type=="pyfftw":
            import pyfftw
            self.gkp=pyfftw.empty_aligned((shp[0],shp[1],int(shp[2]/2)+1),complex)
            self.hkp=pyfftw.empty_aligned((shp[0],shp[1],int(shp[2]/2)+1),complex)
            self.fp=pyfftw.empty_aligned(shp)
            self.rftfp=pyfftw.builders.rfftn(self.fp,norm='backward')
            self.irftgk=pyfftw.builders.irfftn(self.gkp,norm='forward')
            self.irfthk=pyfftw.builders.irfftn(self.hkp,norm='forward')
            self.fftconv = fftconv_pyfftw
        self.shp=shp
        
def partget(i,j,k,Nrhx,Nrhy,Nrhz,Nlx,Nly,Nlz):
    if(i==-1):
        Npx = Nrhx*3
        input_x=np.r_[0:Nrhx,-Nrhx+1:0]
        mapping_x=input_x
        valid_x=input_x
        output_x=input_x
    elif(i==0):
        Npx=int(np.ceil((Nlx)/2)*6)
        input_x=np.r_[Nrhx:Nrhx+Nlx,-Nrhx-Nlx+1:-Nrhx+1]
        mapping_x=np.r_[1:Nlx+1,-1-Nlx+1:0]
        valid_x=[0]
        output_x=[0]
    else:
        imn=np.maximum(1,i-2)
        imx=np.minimum(i+3,Nlx+3)
        numi=imx-imn
        i0=ILS[np.minimum(i-1,2)]
        Npx=int(np.ceil((ILS[numi-1]+1)/2))*6
        input_x=np.r_[0,Nrhx-3+np.r_[imn:imx],-Nrhx+3-np.r_[imx-1:imn-1:-1]]
        mapping_x=np.r_[0,ILS[:numi],-ILS[numi-1::-1]]
        valid_x=[i0,-i0]
        output_x=[Nrhx-3+i,-Nrhx+3-i]
        if(i<3):
            input_x=input_x[1:]
            mapping_x=mapping_x[1:]
    if(j==-1):
        Npy = Nrhy*3
        input_y=np.r_[0:Nrhy,-Nrhy+1:0]
        mapping_y=input_y
        valid_y=input_y
        output_y=input_y
    elif(j==0):
        Npy=int(np.ceil((Nly)/2)*6)
        input_y=np.r_[Nrhy:Nrhy+Nly,-Nrhy-Nly+1:-Nrhy+1]
        mapping_y=np.r_[1:Nly+1,-1-Nly+1:0]
        valid_y=[0]
        output_y=[0]
    else:
        jmn=np.maximum(1,j-2)
        jmx=np.minimum(j+3,Nly+3)
        numj=jmx-jmn
        j0=ILS[np.minimum(j-1,2)]
        Npy=int(np.ceil((ILS[numj-1]+1)/2))*6
        input_y=np.r_[0,Nrhy-3+np.r_[jmn:jmx],-Nrhy+3-np.r_[jmx-1:jmn-1:-1]]
        mapping_y=np.r_[0,ILS[:numj],-ILS[numj-1::-1]]
        valid_y=[j0,-j0]
        output_y=[Nrhy-3+j,-Nrhy+3-j]
        if(j<3):
            input_y=input_y[1:]
            mapping_y=mapping_y[1:]
    if(k==-1):
        Npz = Nrhz*3
        input_z=np.r_[:Nrhz]
        mapping_z=input_z
        valid_z=input_z
        output_z=input_z
    elif(k==0):
        Npz=int(np.ceil((Nlz)/2)*6)
        input_z=np.r_[Nrhz:Nrhz+Nlz]
        mapping_z=np.r_[1:Nlz+1]
        valid_z=[0]
        output_z=[0]
    else:
        kmn=np.maximum(1,k-2)
        kmx=np.minimum(k+3,Nlz+3)
        numk=kmx-kmn
        k0=ILS[np.minimum(k-1,2)]
        Npz=int(np.ceil((ILS[numk-1]+1)/2))*6
        input_z=np.r_[0,Nrhx-3+np.r_[kmn:kmx]]
        mapping_z=np.r_[0,ILS[:numk]]
        valid_z=[k0]
        output_z=[Nrhz-3+k]
        if(k<3):
            input_z=input_z[1:]
            mapping_z=mapping_z[1:]
    input_xyz=xp.ix_(input_x,input_y,input_z)
    mapping_xyz=xp.ix_(mapping_x,mapping_y,mapping_z)
    valid_xyz=xp.ix_(valid_x,valid_y,valid_z)
    output_xyz=xp.ix_(output_x,output_y,output_z)
    return Npx,Npy,Npz,input_xyz,mapping_xyz,valid_xyz,output_xyz

def fftconv_numpy(gk,hk,part):
    part.gkp[part.mapping]=gk[part.input]
    part.hkp[part.mapping]=hk[part.input]
    part.fkp=part.rft(part.irft(part.gkp)*part.irft(part.hkp))
    return part.output,part.fkp[part.valid]/np.prod(part.shp)

def fftconv_pyfftw(gk,hk,part):
    part.gkp.fill(0)
    part.hkp.fill(0)
    part.gkp[part.mapping]=gk[part.input]
    part.hkp[part.mapping]=hk[part.input]
    gp=part.irftgk()
    hp=part.irfthk()
    part.fp[:]=gp*hp
    part.fkp=part.rftfp()
    return part.output,part.fkp[part.valid]/np.prod(part.shp)

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
    Nx,Ny,Nz=int(np.ceil((gr.knx[int(gr.knx.size/2)]+1)/2))*4,int(np.ceil((gr.kny[int(gr.kny.size/2)]+1)/2))*4,int(np.ceil((gr.knz[-1]+1)/2))*4
    Npx,Npy,Npz=int(Nx/2)*3,int(Ny/2)*3,int(Nz/2)*3
    gkp=np.zeros((Npx,Npy,int(Npz/2)+1),dtype=complex)
    hkp=np.zeros_like(gkp)
    ij=np.ix_(gr.knx,gr.kny,gr.knz)
    gkp[ij]=gk
    hkp[ij]=hk
    fkp=np.fft.rfftn(np.fft.irfftn(gkp,norm='forward')*np.fft.irfftn(hkp,norm='forward'),norm='backward')/Npx/Npy/Npz
    fk=np.zeros_like(gk)
    fk[:]=fkp[ij]
    return fk