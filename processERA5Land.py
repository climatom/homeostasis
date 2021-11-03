#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Td/Tw/MDI from ERA5 land data. 

Specifically: 
    
    - [1] Compute Tw from Ta and Td
    - [2] Compute MDI from Ta and tw
    - [3] Write out: 
        (i)   max MDI
        (ii)  Ta at max MDI
        (iii) Tw at max MDI
        (iv)  year, month, day of max MDI

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORTS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys, pickle
import numpy as np
from netCDF4 import Dataset
# import scipy.optimize as optimize
import numba as nb
import warnings
warnings.filterwarnings("ignore")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
eps=0.05 # Tw toleance (C)
eps=eps*1013./(2.501*10**6)
itermax=10 # max iterations for Tw convergence

def _satVP(t,p):
    """Saturation specific humidity from temp (k) and pressure (Pa)"""
    
    # t=np.atleast_1d(t)
    # esat=np.zeros(t.shape)
    t0=273.16
    a1_w=611.21; a3_w=17.502; a4_w=32.19
    a1_i=611.21; a3_i=22.587; a4_i=-0.7
    
    # Sat Vp according to Teten's formula
    if t>273.15:
        esat=a1_w*np.exp(a3_w*(t-t0)/(t-a4_w))
    else:
        esat=a1_i*np.exp(a3_i*(t-t0)/(t-a4_i))    
    return esat


@nb.njit
def _satQ(t,p):
    """Saturation specific humidity from temp (k) and pressure (Pa)"""
    
    # t=np.atleast_1d(t)
    # esat=np.zeros(t.shape)
    t0=273.16
    a1_w=611.21; a3_w=17.502; a4_w=32.19
    a1_i=611.21; a3_i=22.587; a4_i=-0.7
    rat=0.622;
    rat_rec=1-rat
    # widx=t>273.1
    
    # Sat Vp according to Teten's formula
    if t>273.15:
        esat=a1_w*np.exp(a3_w*(t-t0)/(t-a4_w))
    else:
        esat=a1_i*np.exp(a3_i*(t-t0)/(t-a4_i))    
    # Now sat q according to Eq. 7.5 in ECMWF thermodynamics guide
    satQ=rat*esat/(p-rat_rec*esat)
    return satQ


@nb.njit
def _LvCp(t,q):
    """ Latent heat of evaporation (J) and specific heat capacity of moist 
    air (J) from temp (K) and specific humidity (g/g)"""
    r=q/(1-q)
    Cp=1005.7*(1-r)+1850* r*(1-r)
    Lv=1.918*1000000*np.power(t/(t-33.91),2)

    return Lv,Cp

@nb.njit
def _IMP(Tw,t,td,p):
    q=_satQ(td,p)
    qs=_satQ(Tw,p)
    Lv,Cp=_LvCp(t,q)
    diff=Cp/Lv*(t-Tw)-(qs-q)
    
    return diff

@nb.njit
def _ME(nr,nc,t,td,p,mask):
    me=np.zeros(td.shape)*np.nan
    for i in range(nr):
        for j in range(nc):   
            if ~mask[i,j]: continue
            q=_satQ(td[i,j],p[i,j])
            Lv,Cp=_LvCp(t[i,j],q)
            me[i,j]=Cp*t[i,j]+q*Lv
            
    return me
    
@nb.njit
def _TW(nt,nr,nc,ta,td,p,mask):
    
    """ 
    Minimizes the implicit equation:
        
        c p T + Lq = c p TW + εLe sat (TW)/p 
        
    Using the Newton-Rhapson method
        
    Note that t and td are in K; Tw is returned in K
    
    Note that only evaluates where mask is TRUE
        
    """
    tw=np.zeros((nt,nr,nc),dtype=np.float32)*np.nan
    # nilog=np.zeros((nt,nr,nc),dtype=np.float32)*np.nan
    itermax=10 # max iterations
    for _r in range(nr):
        
        for _c in range(nc):
            
            if ~mask[_r,_c]: continue
        
            for _t in range(nt):
                            
                ni=0 # start iteration value
                
                # Initial guess. Assume saturation. 
                x0=ta[_t,_r,_c]-0. 
                f0=_IMP(x0,ta[_t,_r,_c],td[_t,_r,_c],p[_t,_r,_c])
                if np.abs(f0)<=eps: tw[_t,_r,_c]=x0; continue # Got it first go!
                # Second guess, assume Tw=Ta-2    
                xi=x0-2.;
                fi=_IMP(xi,ta[_t,_r,_c],td[_t,_r,_c],p[_t,_r,_c])
                dx=(xi-x0)            
                dfdx=(fi-f0)/dx # first gradient
                if np.abs(fi)<=eps: tw[_t,_r,_c]=xi; continue # Got it 2nd go
                while np.abs(fi)>eps and ni<itermax:
                    xi=x0-f0/dfdx # new guess at Tw
                    fi=_IMP\
                        (xi,ta[_t,_r,_c],td[_t,_r,_c],p[_t,_r,_c]) # error from this guess
                    if np.abs(fi)<=eps: tw[_t,_r,_c]=xi; break # Exit if small error
                    dx=(xi-x0)
                    # if np.abs(dx) <=eps: tw[_t,_r,_c]=xi; break # Exit if only need small change
                    dfdx=(fi-f0)/dx # gradient at x0
                    x0=xi*1. # Store old Tw
                    f0=fi*1. # Store old error                    
                    ni+=1  # Increment counter
                # Catch odd behaviour of iteration 
                if xi >ta[_t,_r,_c] or ni==itermax: xi=np.nan
                tw[_t,_r,_c]=xi # Store the last value of tw. 
                # nilog[_t,_r,_c]=ni
    return tw

@nb.njit
def call_TW(nr,nc,nt,land,ta,td,sp):
    tw=np.zeros((nt,nr,nc))*np.nan
    for i in range(nr):
        for j in range(nc):
            if ~land[i,j]: continue
            else:
                for t in range(nt):
                    tw[t,i,j]=_TW(ta[t,i,j],td[t,i,j],sp[t,i,j])
            

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

fname=sys.argv[1]
# fname="/home/lunet/gytm3/Homeostasis/ERA5Land/19810101.nc"
data=Dataset(fname)
mask=~data.variables["d2m"][0,:,:].mask
td=data.variables["d2m"][:,:,:].data
ta=data.variables["t2m"][:,:,:].data
sp=data.variables["sp"][:,:,:].data
nt,nr,nc=td.shape

# Compute tw
tw=_TW(nt,nr,nc,ta,td,sp,mask)
# For fun, compute max Tw
print("\n\nMax Tw = %.1fC\n\n"%(np.nanmax(tw)-273.15))
# Compute mdi
mdi=0.75*(tw-273.15)+0.3*(ta-273.15)
mdi[np.isnan(mdi)]=-999.
idx=np.nanargmax(mdi,axis=0)
print("\n\nMax MDI = %.1fC\n\n"%np.nanmax(mdi))
mdi[mdi<-998]=np.nan

# Sub-select mdi, ta, tw, td and sp based on max mdi
mdi=np.squeeze(np.take_along_axis(mdi,idx[None,:,:],axis=0))
ta_mdi=np.squeeze(np.take_along_axis(ta,idx[None,:,:],axis=0))
td_mdi=np.squeeze(np.take_along_axis(td,idx[None,:,:],axis=0))
sp_mdi=np.squeeze(np.take_along_axis(sp,idx[None,:,:],axis=0))

# Compute me for this max
# me=_ME(nr,nc,ta_mdi,td,sp,mask)

# Find the maximum Tw and write that out. It may *not* be at the same time as 
# the max mdi
tw[np.isnan(tw)]=-999
idx=np.nanargmax(tw,axis=0)
tw[tw<-998]=np.nan
tw=np.squeeze(np.take_along_axis(tw,idx[None,:,:],axis=0))
# also take ta, td and sp at these points
ta_tw=np.squeeze(np.take_along_axis(ta,idx[None,:,:],axis=0))
td_tw=np.squeeze(np.take_along_axis(td,idx[None,:,:],axis=0))
sp_tw=np.squeeze(np.take_along_axis(sp,idx[None,:,:],axis=0))


# Pickle
pickle.dump(mdi,open( fname.replace(".nc","_mdi.p"), "wb" ) )
pickle.dump(tw,open( fname.replace(".nc","_tw.p"), "wb" ) )
pickle.dump(ta_mdi,open( fname.replace(".nc","_ta.p"), "wb" ) )
pickle.dump(td_mdi,open( fname.replace(".nc","_td.p"), "wb" ) )
# pickle.dump(me,open( fname.replace(".nc","_me.p"), "wb" ) )
pickle.dump(sp_mdi,open( fname.replace(".nc","_sp.p"), "wb" ) )
# Repeat for TW specials
pickle.dump(ta_tw,open( fname.replace(".nc","_ta_tw.p"), "wb" ) )
pickle.dump(td_tw,open( fname.replace(".nc","_td_tw.p"), "wb" ) )
pickle.dump(sp_tw,open( fname.replace(".nc","_sp_tw.p"), "wb" ) )


# pickle.dump(wbgt,open( fname.replace(".nc","_wbgt.p"), "wb" ) )