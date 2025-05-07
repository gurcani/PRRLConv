# PRRLConv

## Partially Regular Recurrent Lattice Efficient Convolution Algorithm

A simple implementation of the aglorithm for computing convolutions efficiently on Partially Regular Recurrent Lattices (PRRLs),
which are lattices that are constructed by extending a regular grid in the wave-number domain through a recurrence relation. A PRRL
is parametrized by two integers in each directions, `Nr`, representing the number of elements in the regular part of the grid,
and `Nl` representing the number of iterations of the recurrence relation.

In order to see how to use them, see the files `test2D.py` and `test3D.py`, where a comparison with a dense fft algorithm is computed.

Nonetheless, here is a basic usage example:

## Basic usage:

```python

import numpy as np
from prrl import cvpgrid,partlist,conv

Nrx,Nlx=64,8
Nry,Nly=64,8

gr=cvpgrid(Nrx,Nry,Nlx,Nly)
kx,ky=xp.ix_(gr.knx,gr.kny)

def hermsym(phik,gr):
    phik[0,0]=phik[0,0].real
    phik[-1:-gr.Nrhx-gr.Nlx:-1,0]=phik[1:gr.Nrhx+gr.Nlx,0].conj()

gk=np.exp(1j*2*np.pi*np.random.rand(*gr.shp))*np.random.rand(*gr.shp)
hk=np.exp(1j*2*np.pi*np.random.rand(*gr.shp))*np.random.rand(*gr.shp)
hermsym(gk, gr)
hermsym(hk, gr)

pl=partlist(gr,fft_type='numpy')

fk=conv(gk,hk,pl)

```

The options for the `fft_type` argument are `'numpy'`, `'scipy'` and `'pyfftw'` as ordered from the slowest to the fastest.