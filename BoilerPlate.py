#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boiler-plate functions for humid heat computations

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORTS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
# import scipy.optimize as optimize
import numba as nb
import pythran as pt
import sympy
esat, a1,a3,a4,t,t0=sympy.symbols('esat a1 a3 a4 t t0')
eq=sympy.Eq(a1*sympy.exp(a3*(t-t0)/(t-a4)),esat)
res=sympy.solve(eq,t)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
eps=0.05 # Tw toleance (C)
itermax=10 # max iterations for Tw convergence
rh_thresh=0.1 # If less than this, take rh=this. 


#pythran export _satVp2d(float[:,:],int,int)
def _satVp2d(t,nr,nc):
    """Saturation specific humidity from 2d array of temp (k)"""
    t0=273.16
    a1_w=611.21; a3_w=17.502; a4_w=32.19
    a1_i=611.21; a3_i=22.587; a4_i=-0.7
    esat=np.zeros((nr,nc))
    for i in range(nr):
        for j in range(nc):
            if t[i,j]>273.15:
                esat[i,j]=a1_w*np.exp(a3_w*(t[i,j]-t0)/(t[i,j]-a4_w))
            else:
                esat[i,j]=a1_i*np.exp(a3_i*(t[i,j]-t0)/(t[i,j]-a4_i))
    return esat      
    
@nb.njit
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

@nb.njit
def _rhDew(rh,t):
   """ Dewpoint from t (K) and rh (%)"""
   a1=611.21; a3=17.502; a4=32.19
   t0=273.16
   vp=_satVP(t)*rh/100.
   dew=(a3*t0 - a4*np.log(vp/a1))/(a3 - np.log(vp/a1))
   return dew

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
    Cp=1005.7*(1-r)+1850.* r*(1-r)
    Lv=1.918*1000000*np.power(t/(t-33.91),2)

    return Lv,Cp

@nb.njit
def _IMP(Tw,t,td,p):
    q=_satQ(td,p)
    qs=_satQ(Tw,p)
    Lv,Cp=_LvCp(t,q)
    diff=(t-Tw)-Lv/Cp*(qs-q)
    
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
def _TW(nt,ta,rh,p):
    
    """ 
    Minimizes the implicit equation:
        
        c p T + Lq = c p TW + εLe sat (TW)/p 
        
    Using the Newton-Rhapson method
        
    Note that t and td are in K; Tw is returned in K
    
        
    """
    tw=np.zeros(nt,dtype=np.float32)*np.nan
    itermax=10 # max iterations
    for _t in range(nt):
                ni=0 # start iteration value
                # Protext against "zero" rh
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
                if xi >ta[_t] or ni==itermax: xi=np.nan
                tw[_t]=xi # Store the last value of tw. 
                # nilog[_t,_r,_c]=ni
    return tw

 
def _TW_TD(tw,rh,p):
    
    """ 
    Minimizes the implicit equation:
        
        cp*Tw + ε*L*esat(Tw)/p = cp*Ta + ε*L*esat(Ta)/p*RH
       
    Using the Newton-Rhapson method
        
    Note that Tw is in K and rh is in %
        
    """
    itermax=10 # max iterations
    ni=0 # start iteration value       

    # Initial guess. Assume ta = tw+1 
    x0=tw+1
    
    # _IMP_TwTa(tw,t,p,rh)
    f0=_IMP_TwTa(tw,x0,p,rh) # Initial feval
    
    if np.abs(f0)<=eps: ta=x0; return ta
    # Second guess, assume Ta=Tw-2    
    xi=x0+1.;
    fi=_IMP_TwTa(tw,xi,p,rh)
    dx=(xi-x0)            
    dfdx=(fi-f0)/dx # first gradient
    if np.abs(fi)<=eps: ta=xi; return ta # Got it 2nd go
    while np.abs(fi)>eps and ni<itermax:
        xi=x0-f0/dfdx # new guess at Ta
        fi=_IMP_TwTa(t,xi,p,rh) # error from this guess                    
        if np.abs(fi)<=eps: ta=xi; return ta
        dx=(xi-x0)
        dfdx=(fi-f0)/dx # gradient at x0
        x0=xi*1. # Store old Tw
        f0=fi*1. # Store old error                    
        ni+=1  # Increment counter
    # Catch odd behaviour of iteration 
    if xi <tw or ni==itermax: xi=np.nan
    ta=xi # Store the last value of tw. 
    # nilog[_t,_r,_c]=ni
    return ta
    
    

@nb.njit
def _TW2d(nr,nc,ta,rh,p):
    
    """ 
    Minimizes the implicit equation:
        
        c p T + Lq = c p TW + εLe sat (TW)/p 
        
    Using the Newton-Rhapson method
        
    Note that t and td are in K; Tw is returned in K
    
        
    """
    tw=np.zeros((nr,nc),dtype=np.float32)*np.nan
    itermax=10 # max iterations
    for _r in range(nr):
        for _c in range(nc):
                ni=0 # start iteration value       
                td=_rhDew(rh[_r,_c],ta[_r,_c])
                # Initial guess. Assume saturation. 
                x0=ta[_r,_c]-0. 
                f0=_IMP(x0,ta[_r,_c],td,p[_r,_c])
                if np.abs(f0)<=eps: tw[_r,_c]=x0; continue # Got it first go!
                # Second guess, assume Tw=Ta-2    
                xi=x0-2.;
                fi=_IMP(xi,ta[_r,_c],td,p[_r,_c])
                dx=(xi-x0)            
                dfdx=(fi-f0)/dx # first gradient
                if np.abs(fi)<=eps: tw[_r,_c]=xi; continue # Got it 2nd go
                while np.abs(fi)>eps and ni<itermax:
                    xi=x0-f0/dfdx # new guess at Tw
                    fi=_IMP\
                        (xi,ta[_r,_c],td,p[_r,_c]) # error from this guess
                    if np.abs(fi)<=eps: tw[_r,_c]=xi; break # Exit if small error
                    dx=(xi-x0)
                    # if np.abs(dx) <=eps: tw[_t,_r,_c]=xi; break # Exit if only need small change
                    dfdx=(fi-f0)/dx # gradient at x0
                    x0=xi*1. # Store old Tw
                    f0=fi*1. # Store old error                    
                    ni+=1  # Increment counter
                # Catch odd behaviour of iteration 
                if xi >ta[_r,_c] or ni==itermax: xi=np.nan
                tw[_r,_c]=xi # Store the last value of tw. 
                # nilog[_t,_r,_c]=ni
    return tw

@nb.njit
def _TW3d(nt,nr,nc,ta,td,p):
    
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
def _IMP_TwTa(tw,t,p,rh):
    
    # _satQ(t,p)
    qtw=_satQ(tw,p)
    qt=_satQ(t,p)
    qt=qt*rh/100.
    Lv,Cp=_LvCp(t,qt)
    diff=(Cp*t+qt*Lv)-(Cp*tw+qtw*Lv)
    
    return diff

@nb.njit
def _TW_TD_2d(nr,nc,tw,rh,p):
    
    """ 
    Minimizes the implicit equation:
        
        cp*Tw + ε*L*esat(Tw)/p = cp*Ta + ε*L*esat(Ta)/p*RH
       
    Using the Newton-Rhapson method
        
    Note that Tw is in K and rh is in %
        
    """
    ta=np.zeros((nr,nc),dtype=np.float32)*np.nan
    itermax=10 # max iterations
    for _r in range(nr):
        for _c in range(nc):
                if np.isnan(tw[_r,_c]): continue
                ni=0 # start iteration value       

                # Initial guess. Assume ta = tw+1 
                x0=tw[_r,_c]+1
                
                # _IMP_TwTa(tw,t,p,rh)
                f0=_IMP_TwTa(tw[_r,_c],x0,p[_r,_c],rh[_r,_c]) # Initial feval
                
                if np.abs(f0)<=eps: ta[_r,_c]=x0; continue # Got it first go!
                # Second guess, assume Ta=Tw-2    
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
                # Catch odd behaviour of iteration 
                if xi <tw[_r,_c] or ni==itermax: xi=np.nan
                ta[_r,_c]=xi # Store the last value of tw. 
                # nilog[_t,_r,_c]=ni
    return ta

@nb.njit
def call_TW(nr,nc,nt,ta,td,sp):
    tw=np.zeros((nt,nr,nc))*np.nan
    for i in range(nr):
        for j in range(nc):
                for t in range(nt):
                    tw[t,i,j]=_TW(ta[t,i,j],td[t,i,j],sp[t,i,j])

@nb.jit
def haversine_fast(lat1,lng1,lat2,lng2,miles=False):
    R = 6378.1370 # equatorial radius of Earth (km)
    """ 
    Calculate the great-circle distance bewteen two points on the Earth surface.

    :input: two scalars (lat1,lng1) and two arrays (lat2,lng2)

    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))

    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.

    """
    # convert all latitudes/longitudes from decimal degrees to radians
    lat1=np.radians(lat1); lat2=np.radians(lat2); lng1=np.radians(lng1); 
    lng2=np.radians(lng2)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * R * np.arcsin(np.sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers