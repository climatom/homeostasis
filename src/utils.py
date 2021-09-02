#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boiler-plate functions for fast humid heat computations

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORTS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import numba as nb
from numba import prange
from netCDF4 import Dataset
import sys 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

eps=0.05 # Tw toleance (C)
itermax=10 # max iterations for Tw convergence
rh_thresh=1. # %. Below this, we assume RH is this. 
    
@nb.njit(fastmath=True)
def _satVP(t):
    """Saturation specific humidity from temp (k)"""
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

@nb.njit(fastmath={"nnan":False},parallel=True)
def _satVP_3D(nt,nr,nc,t):    
    out=np.zeros((nt,nr,nc))*np.nan
    for _t in prange(nt):
        for _r in prange(nr):
            for _c in prange(nc):
                if np.isnan(t[_t,_r,_c]): continue
                out[_t,_r,_c]=_satVP(t[_t,_r,_c])
    return out

@nb.njit(fastmath=True)
def _rhDew(rh,t):
   """ Dewpoint from t (K) and rh (%)"""
   a1=611.21; a3=17.502; a4=32.19
   t0=273.16
   vp=_satVP(t)*rh/100.
   dew=(a3*t0 - a4*np.log(vp/a1))/(a3 - np.log(vp/a1))
   return dew

@nb.njit(fastmath=True)
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

@nb.njit(fastmath={"nnan":False},parallel=True)
def _satQ3d(t,p,nt,nr,nc):
    out=np.zeros((nt,nr,nc))*np.nan
    for _t in nb.prange(nt):
        for _r in nb.prange(nr):
            for _c in nb.prange(nc):
                if np.isnan(t[_t,_r,_c]) or np.isnan(p[_t,_r,_c]): continue
                out[_t,_r,_c]=_satQ(t[_t,_r,_c],p[_t,_r,_c])
    return out

@nb.njit(fastmath=True)
def _LvCp(t,q):
    """ Latent heat of evaporation (J) and specific heat capacity of moist 
    air (J) from temp (K) and specific humidity (g/g)"""
    r=q/(1-q)
    Cp=1005.7*(1-r)+1850.* r*(1-r)
    Lv=1.918*1000000*np.power(t/(t-33.91),2)

    return Lv,Cp

@nb.njit(fastmath=True)
def _IMP(Tw,t,q,p):
    qs=_satQ(Tw,p)
    Lv,Cp=_LvCp(t,q)
    diff=(t-Tw)-Lv/Cp*(qs-q)
    
    return diff

@nb.njit(fastmath=True)
def _ME(nr,nc,t,td,p,mask):
    me=np.zeros(td.shape)*np.nan
    for i in range(nr):
        for j in range(nc):   
            if ~mask[i,j]: continue
            q=_satQ(td[i,j],p[i,j])
            Lv,Cp=_LvCp(t[i,j],q)
            me[i,j]=Cp*t[i,j]+q*Lv
            
    return me


@nb.njit(fastmath={"nnan":False},parallel=True)
def _TW(nt,ta,rh,p):
    
    """ 
    Minimizes the implicit equation:
        
        c p T + Lq = c p TW + εLe sat (TW)/p 
        
    Using the Newton-Rhapson method
        
    Note that t and td are in K; Tw is returned in K
    
    Note that this is the one-dimensional version of
    the below
    
    RH is in %
        
    """
    tw=np.zeros(nt,dtype=np.float32)*np.nan
    for _t in range(nt):
                ni=0 # start iteration value
                # Protect against "zero" rh
                if rh[_t]<rh_thresh: td=_rhDew(rh_thresh,ta[_t])    
                else:td=_rhDew(rh[_t],ta[_t])
                # Initial guess. Assume saturation. 
                x0=ta[_t]-0. 
                f0=_IMP(x0,ta[_t],td,p[_t])
                if np.abs(f0)<=eps: tw[_t]=x0; continue # Got it first go!
                # Second guess, assume Tw=Ta-2    
                xi=x0-2.;
                fi=_IMP(xi,ta[_t],td,p[_t])
                dx=(xi-x0)            
                dfdx=(fi-f0)/dx # first gradient
                if np.abs(fi)<=eps: tw[_t]=xi; continue # Got it 2nd go
                while np.abs(fi)>eps and ni<itermax:
                    xi=x0-f0/dfdx # new guess at Tw
                    fi=_IMP\
                        (xi,ta[_t],td,p[_t]) # error from this guess
                    if np.abs(fi)<=eps: tw[_t]=xi; break # Exit if small error
                    dx=(xi-x0)
                    dfdx=(fi-f0)/dx # gradient at x0
                    x0=xi*1. # Store old Tw
                    f0=fi*1. # Store old error                    
                    ni+=1  # Increment counter

                # Catch odd behaviour of iteration 
                # If it didn't converge, set to nan    
                if ni == itermax: 
                    xi=np.nan
            
                # If it did converge, but reached a value >ta, set to ta 
                # [can happen when close to saturation and precision accepts
                # solution >ta] 
                elif xi >ta[_t,]:    
                    xi=ta[_t]*1. # V. close to saturation, so set to ta
    return tw


@nb.njit(fastmath={"nnan":False},parallel=True)  
def _TW3d(nt,nr,nc,ta,q,p):
    
    """ 
    Minimizes the implicit equation:
        
        cp*T + Lq = cp*TW + Le*sat(TW)/p 
        
    Using the Newton-Rhapson method
        
    Note that t is in K and q is g/g
    
    Tw is returned in K
        
    """
    
    tw=np.zeros((nt,nr,nc),dtype=np.float32)*np.nan

    for _r in range(nr):
        
        for _c in range(nc):           
        
            for _t in range(nt):
                
                if np.isnan(ta[_t,_r,_c]) or np.isnan(q[_t,_r,_c]) or np.isnan(q[_t,_r,_c]): 
                    continue
                            
                ni=0 # start iteration value
                
                # Initial guess. Assume saturation. 
                x0=ta[_t,_r,_c]-0. 
                f0=_IMP(x0,ta[_t,_r,_c],q[_t,_r,_c],p[_t,_r,_c])
                if np.abs(f0)<=eps: tw[_t,_r,_c]=x0; continue # Got it first go!
        
                # Second guess, assume Tw=Ta-1    
                xi=x0-1.;
                fi=_IMP(xi,ta[_t,_r,_c],q[_t,_r,_c],p[_t,_r,_c])
                if np.abs(fi)<=eps: tw[_t,_r,_c]=xi; continue # Got it 2nd go!
        
            	# Compute first gradient
                dx=(xi-x0)            
                dfdx=(fi-f0)/dx 
        
               # Iterate while error is too big, and while iteration is in <itermax
                while np.abs(fi)>eps and ni<itermax:
        
                    xi=x0-f0/dfdx # new guess at Tw
                    fi=_IMP\
                        (xi,ta[_t,_r,_c],q[_t,_r,_c],p[_t,_r,_c]) # error from this guess
                    if np.abs(fi)<=eps: tw[_t,_r,_c]=xi; break # Exit if small error
                    dx=(xi-x0)
   
                    dfdx=(fi-f0)/dx # gradient at x0
                    x0=xi*1. # Store old Tw
                    f0=fi*1. # Store old error                    
                    ni+=1  # Increment counter
        
                # If it didn't converge, set to nan    
                if ni == itermax: 
                    xi=np.nan
            
                # If it did converge, but reached a value >ta, set to ta 
                # [can happen when close to saturation and precision accepts
                # solution >ta] 
                elif xi >ta[_t,_r,_c]:    
                    xi=ta[_t,_r,_c]*1. # V. close to saturation, so set to ta

    return tw

@nb.njit(fastmath={"nnan":False},parallel=True)
def _MDI(nt,nr,nc,tw,ta):
	"""
	Tw and Ta should be input in K
	MDI is output in C
	"""
	mdi=np.zeros((nt,nr,nc))*np.nan
	for _t in range(nt):
		for _r in range(nr):
			for _c in range(nc):
				mdi[_t,_r,_c]=(tw[_t,_r,_c]-273.15)*0.75+(ta[_t,_r,_c]-273.15)*0.3
	return mdi  

@nb.njit(fastmath={"nnan":False})
def _IMP_TwTa(tw,t,p,rh):
    
    # _satQ(t,p)
    qtw=_satQ(tw,p)
    qt=_satQ(t,p)
    qt=qt*rh/100.
    Lv,Cp=_LvCp(t,qt)
    diff=(Cp*t+qt*Lv)-(Cp*tw+qtw*Lv)
    
    return diff

@nb.njit(fastmath={"nnan":False},parallel=True)
def _TW_TD_2d(nr,nc,tw,rh,p):
    
    """ 
    Minimizes the implicit equation:
        
        cp*Tw + ε*L*esat(Tw)/p = cp*Ta + ε*L*esat(Ta)/p*RH
       
    Using the Newton-Rhapson method
        
    Note that Tw is in K and rh is in %
        
    """
    ta=np.zeros((nr,nc),dtype=np.float32)*np.nan
    itermax=10 # max iterations
    for _r in prange(nr):
        
        for _c in prange(nc):
            
                if np.isnan(tw[_r,_c]) or np.isnan(rh[_r,_c]) or np.isnan(p[_r,_c]): 
                   continue
                
                ni=0 # start iteration value       

                # Initial guess. Assume ta = tw+1 
                x0=tw[_r,_c]+1
                
                # _IMP_TwTa(tw,t,p,rh)
                f0=_IMP_TwTa(tw[_r,_c],x0,p[_r,_c],rh[_r,_c]) # Initial feval
                
                if np.abs(f0)<=eps: ta[_r,_c]=x0; continue # Got it first go!
                # Second guess, assume Ta=Tw+2    
                xi=x0+1.;
                fi=_IMP_TwTa(tw[_r,_c],xi,p[_r,_c],rh[_r,_c])
                dx=(xi-x0)            
                dfdx=(fi-f0)/dx # first gradient
                if np.abs(fi)<=eps: ta[_r,_c]=xi; continue # Got it 2nd go
                while np.abs(fi)>eps and ni<itermax:
                    xi=x0-f0/dfdx # new guess at Ta
                    fi=_IMP_TwTa(tw[_r,_c],xi,p[_r,_c],rh[_r,_c]) # error from this guess                    
                    if np.abs(fi)<=eps: ta[_r,_c]=xi; break # Exit if small error
                    dx=(xi-x0)
                    dfdx=(fi-f0)/dx # gradient at x0
                    x0=xi*1. # Store old Tw
                    f0=fi*1. # Store old error                    
                    ni+=1  # Increment counter
                        
                # If it didn't converge, set to nan    
                if ni == itermax: 
                    tw[_r,_c]=np.nan
            
                # If it did converge, but reached a value >ta, set to ta 
                # [can happen when close to saturation and precision accepts
                # solution >ta] 
                elif xi <tw[_r,_c]:    
                    tw[_r,_c]=np.nan # V. close to saturation, so set to tai
    return ta



