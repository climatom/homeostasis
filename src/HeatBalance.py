#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human heat balance model according to J. Foster, ported to python by T. 
Matthews

Detailed description to follow

"""
import numpy as np, numba as nb, pandas as pd, pythran as pt, FastHeat as fh
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
import warnings, utils
warnings.filterwarnings('ignore')
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
# Constants/settings
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
tre=37.0 # Rectal temperature
AD=1.92 # surface area of average male (m**2): Foster et al. (2021)
AR_AD=0.77 # Ratio of effective radiative area to body surface area 
# (ISO7933 2004)
emiss=0.95 # Thermal emissivity
boltz=5.67*10**-8 # Boltzmann constant
latent_vap=2430.0 # latent heat of vaporization (2430 J/g)
fix_skin=True # Fix the skin surface temperature? [Bool]
_tsk=35. # Skin surface temperature, if fixed [deg C])
const_hr=False
_hr=4.70
isoeff=1989# 1989 or 2004
wcomp="Cramer"# Josh or Cramer
# Range of sweat  - Comes from mean +/- one stdv of sweat rate in athletes --
# reference is Barnes et al. (2019)
sweat_lower=(1240-620)/1000. # ml/hour => kg/hr
sweat_upper=(1240+620)/1000. # ml/hour => kg/hr
# Range of metabolic heat production
met_lower=55.
met_upper=70.
# Range of air velocity
# Range comes from Morris et al. (2021)
va_lower=0.2
va_upper=3.5
nsim=1000 # Monte Carlo simulations
crit_sed=28/0.74 # From Kenney et al. (2004)
figdir="/home/lunet/gytm3/Homeostasis/Figures/"
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
# Functions
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

#*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
# Dry heat
#*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
       
@nb.jit('float64(float64,float64,float64)')
def DRYHEAT(tsk,to,ia):
    # Notes:
    # tskin in c
    # to is operative temperature in deg C
    c=(tsk-to)/ia # [W/m**2]
    return c


@nb.jit('float64(float64,float64,float64,float64,float64)')
def TSKIN(ta,mrt,pa,tre,va):
    # Notes: 
    # pa is ambient water vapour pressure in kPa
    # va (global) is air velocity in m/s
    # # tre is rectal temperature (assumed 37C) -- constant
    # if fix_skin: tskin=35.
    # else:
    if fix_skin: tskin=_tsk
    else:
        tskin=7.2+0.064*ta+0.061*mrt+0.198*pa-0.348*va+0.616*tre 
    return tskin# [deg C]
        
@nb.jit('float64(float64,float64)')
def VP_PARSONS(ta,rh):
    # Notes:
    # ta is in deg C
    # rh is in percent
    # pa is returned in kPa
    pa=np.exp(18.956-4030.18/(ta+235))/rh*0.1
    return pa

@nb.jit('float64(float64,float64)')
def VP_TETENS(ta,rh):
    # Notes:
    # ta is in degC
    # rh is in percent
    # pa is returned in kPa
    ta+=273.15
    rh/=100.
    t0=273.16
    a1_w=611.21; a3_w=17.502; a4_w=32.19
    a1_i=611.21; a3_i=22.587; a4_i=-0.7
    
    # Sat Vp according to Teten's formula
    if ta>273.15:
        pa=a1_w*np.exp(a3_w*(ta-t0)/(ta-a4_w))
    else:
        pa=a1_i*np.exp(a3_i*(ta-t0)/(ta-a4_i))    
    pa=pa*rh*0.001
    return pa

@nb.jit('float64(float64,float64,float64,float64)')
def TO(hr,tr,hc,ta):
    # Notes:
    # hr and hc are radiative and convective transfer coefficients, resp.
    # to is 
    to=(hr*tr+hc*ta)/(hr+hc)
    return to

@nb.jit('float64(float64,float64)')
def HR(tsk,tre):
    # Notes
    # tsk is skin temperature in deg C
    # tr is rectal temperature in deg C (constant)
    # Other terms (emiss, boltz, AR_AD) are constants
    hr=4.*emiss*boltz*AR_AD*np.power((273.2+(tsk+tre)/2.),3)
    if const_hr: hr =_hr
    return hr

@nb.jit('float64(float64)')    
def HC(va):
    # Notes:
    # va (global) is in m/s
    # HC is returned in W/m**2/K
    hc=8.3*np.power(va,0.6)
    return hc

@nb.jit('float64(float64,float64)')  
def IA(hc,hr):
    # Notes
    # computes the thermal insulation provided by the air layer surrounding
    # the skin (m**2 K/W)
    ia=1/(hc+hr)
    return ia

#*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
# Evaporative heat 
#*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
@nb.jit('float64(float64,float64)') 
def ESWEAT(swmax,eff):
    # Notes
    # swmax is in W/m**2
    # eff is dimensionless
    # esweat is returned in W/m**2
    esweat=swmax*eff
    return esweat

@nb.jit('float64(float64,float64)')
def SWMAX(swprod,ta):
    # Notes:
    # Includes the dependnecy of latent heat of vaporization on temperature, 
    # after Henderson-Sellers (1984). Units are J/kg
    # swprod is a global parameter
    # swmax is returned as W/m**2
    # ta must be in K
    if ta < 100: ta+=273.15 # risky autocorrect if in C
    swlatent=1.918*1000000*np.power(ta/(ta-33.91),2)
    swmax=swprod*swlatent/3600.
    return swmax
    
@nb.jit('float64(float64,float64,float64)')
def EFF(swmax,emax,hreq=0.0):
    # Notes:
    # returns dimensionless skin wettedness (proportion of AD covered by sweat)
    if wcomp=="Josh":
        w=swmax/emax
    else: w=hreq/emax
    
    # ISO 1989
    if isoeff == 1989:
        eff=1-np.power(w,2)/2.
        # if eff<0.5: eff=0.5
        
   # ISO 2004     
    elif isoeff == 2004:        
        if w<=1: 
            eff=1-np.power(w,2)/2.
        elif  w>1 and w <=1.7:
            eff=np.power((2-w),2)/2.
        else:
            eff = 0.05
    else: raise ValueError
    
    return eff

@nb.jit('float64(float64,float64,float64)')
def EMAX(pask,pa,rea):
    # Notes:
    # pask is the saturation vapour pressure at the skin temperature (kPa)
    # pa is the ambient vapour pressure (kPa)
    # rea is the evaporative resistance of the air around the skin
    emax=(pask-pa)/rea    
    return emax
  
@nb.jit('float64(float64)')   
def REA(hc):
    # Returns the evaportaive resistance given the convective exchange
    # coefficient. Units are 1/(W x m**2 x kPa)
    rea=1/(16.5*hc)
    return rea

# *NOTE* The actual value used for evaporative sweat loss is the minimum of
# emax and esweat
    
#*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
# Respiratory heat loss
#*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *

@nb.jit('float64(float64,float64)')
def CRES(ta,met):
    # returns dry respiratory loss in W/m**2 using ta (deg C) and 
    # met -- global parameter (metabolic heat in W/m**2)
    cres=0.0014*met*(34.-ta)
    return cres

@nb.jit('float64(float64,float64)')  
def ERES(pa,met):
    # returns evaporative respiratory loss in W/m**2 using pa (kPa) and 
    # met -- global parameter (metabolic heat in W/m**2)
    eres=0.0173*met*(5.87-pa)
    return eres

#*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *
# Coordinating functions
#*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *

# Compute the (actual) heat balance 
@nb.njit
def HB(ta,rh,met,va,swprod):
    # Note: tre is global constant (rectal temperature)
    pa=VP_TETENS(ta,rh)
    tsk=TSKIN(ta,ta,pa,tre,va) ; # Note - second ta is mrt
    hc=HC(va)
    hr=HR(tsk,tre); 
    to=TO(hr,ta,hc,ta) # Note - second ta is mrt
    ia=IA(hc,hr)
    c=DRYHEAT(tsk,to,ia)
    # Moist heat exchange 
    swmax=SWMAX(swprod,ta)
    pask=VP_TETENS(tsk,100.)
    cres=CRES(ta,va)
    eres=ERES(pa,va)   
    # Compute 'required' evaporative cooling (Cramer and Jay. 2019 )
    hreq=met-(cres+eres+c)
    
    if pa != pask: # If equal, flux must be zero; otherwise evaluate the below. 
        rea=REA(hc)
        emax=EMAX(pask,pa,rea)
        eff=EFF(swmax,emax,hreq)
        esweat=ESWEAT(swmax,eff)
        # *NOTE* The actual value used for evaporative sweat loss is the minimum of
        # emax and esweat
        e=np.min(np.array([emax,esweat]))
    else:e=0.
    
    # * * *   
    # Heat balance for this time step; positive is heat gain; negative is 
    # loss
    # * * * 
    s=met-(c+e+cres+eres)
    # assert ~np.isnan(s),"NaN Detected!"
    #--------------------------------
    return s

# Compute the (absolute) heat balance 
@nb.njit
def INFL(ta,rh,met,va,swprod):
    # Note: tre is global constant (rectal temperature)
    pa=VP_TETENS(ta[0],rh)
    tsk=TSKIN(ta[0],ta[0],pa,tre,va) ; # Note - second ta is mrt
    hc=HC(va)
    hr=HR(tsk,tre); 
    to=TO(hr,ta[0],hc,ta[0]) # Note - second ta is mrt
    ia=IA(hc,hr)
    c=DRYHEAT(tsk,to,ia)
    # Moist heat exchange 
    swmax=SWMAX(swprod,ta[0])
    pask=VP_TETENS(tsk,100.)
    cres=CRES(ta[0],va)
    eres=ERES(pa,va)   
    # Compute 'required' evaporative cooling (Cramer and Jay. 2019 )
    hreq=met-(cres+eres+c)
    
    if pa != pask: # If equal, flux must be zero; otherwise evaluate the below. 
        rea=REA(hc)
        emax=EMAX(pask,pa,rea)
        eff=EFF(swmax,emax,hreq)
        esweat=ESWEAT(swmax,eff)
        # *NOTE* The actual value used for evaporative sweat loss is the minimum of
        # emax and esweat
        e=np.min(np.array([emax,esweat]))
    else:e=0.
    
    # * * *   
    # Heat balance for this time step
    # * * * 
    s=met-(c+e+cres+eres)
    #--------------------------------
    return np.abs(s)

@nb.njit
def INFL_SW(swrate,ta,rh,met,va):
    swprod=swrate[0]/1.92
    # Note: tre is global constant (rectal temperature)
    pa=VP_TETENS(ta,rh)
    tsk=TSKIN(ta,ta,pa,tre,va) ; # Note - second ta is mrt
    hc=HC(va)
    hr=HR(tsk,tre); 
    to=TO(hr,ta,hc,ta) # Note - second ta is mrt
    ia=IA(hc,hr)
    c=DRYHEAT(tsk,to,ia)
    # Moist heat exchange 
    swmax=SWMAX(swprod,ta)
    pask=VP_TETENS(tsk,100.)
    cres=CRES(ta,va)
    eres=ERES(pa,va)   
    # Compute 'required' evaporative cooling (Cramer and Jay. 2019 )
    hreq=met-(cres+eres+c)
    
    if pa != pask: # If equal, flux must be zero; otherwise evaluate the below. 
        rea=REA(hc)
        emax=EMAX(pask,pa,rea)
        eff=EFF(swmax,emax,hreq)
        esweat=ESWEAT(swmax,eff)
        # *NOTE* The actual value used for evaporative sweat loss is the minimum of
        # emax and esweat
        e=np.min(np.array([emax,esweat]))
    else:e=0.
    
    # * * *   
    # Heat balance for this time step
    # * * * 
    s=met-(c+e+cres+eres)
    #--------------------------------
    return np.abs(s)


@nb.njit
def INFL_RH(rh,ta,met,va,swprod):
    # swprod=swrate/AD
    # Note: tre is global constant (rectal temperature)
    pa=VP_TETENS(ta,rh[0])
    tsk=TSKIN(ta,ta,pa,tre,va) ; # Note - second ta is mrt
    hc=HC(va)
    hr=HR(tsk,tre); 
    to=TO(hr,ta,hc,ta) # Note - second ta is mrt
    ia=IA(hc,hr)
    c=DRYHEAT(tsk,to,ia)
    # Moist heat exchange 
    swmax=SWMAX(swprod,ta)
    pask=VP_TETENS(tsk,100.)
    cres=CRES(ta,va)
    eres=ERES(pa,va)   
    # Compute 'required' evaporative cooling (Cramer and Jay. 2019 )
    hreq=met-(cres+eres+c)
    
    if pa != pask: # If equal, flux must be zero; otherwise evaluate the below. 
        rea=REA(hc)
        emax=EMAX(pask,pa,rea)
        eff=EFF(swmax,emax,hreq)
        esweat=ESWEAT(swmax,eff)
        # *NOTE* The actual value used for evaporative sweat loss is the minimum of
        # emax and esweat
        e=np.min(np.array([emax,esweat]))
    else:e=0.
    
    # * * *   
    # Heat balance for this time step
    # * * * 
    s=met-(c+e+cres+eres)
    #--------------------------------
    return np.abs(s)

# Find the inflection point -- zero in the above. 
def CRIT_T(rh,met,va,swprod,n,out):
    for i in range(n):
        out[i]=minimize(fun=INFL,
                          x0=25.,
                          args=(rh[i],met,va,swprod),
                          method="Nelder-Mead").x
    return out

def CRIT_SWrate(ta,rh,met,va):
    # (swrate,ta,rh,met,va)
    out=minimize(fun=INFL_SW,
                          x0=3.,
                          args=(ta,rh,met,va),
                          method="Nelder-Mead",
                          bounds=(0,100),
                          options={'disp':True})
    

    return out

def CRIT_RH(ta,met,va,swrate):
    # (swrate,ta,rh,met,va)
    out=minimize(fun=INFL_RH,
                          x0=10.,
                          args=(ta,met,va,swrate),
                          method="Nelder-Mead",
                          options={'disp':True,'tol':1e-6})
    return out
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
# MAIN 
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
if __name__=="__main__":
    
    # Find swprod at inflection point for given T/RH
    ta_ill=54.4
    met_ill=65.
    va_ill=0.2
    s_ill=3./AD
    crit_rh=CRIT_RH(ta_ill,met_ill,va_ill,s_ill).x
    tw_ill=utils._TW(1,np.atleast_1d(ta_ill)+273.15,np.atleast_1d(crit_rh),
                   np.array([101300.,]))-273.15
    assert 1==2
    
    
    rhs=np.linspace(5,100,100)
    nt=len(rhs)
    store=np.zeros((nt,nsim))
    pc_complete=0
    for i in range(nsim):
        # Set sweat
        sweat_rate=np.random.uniform(sweat_lower,sweat_upper)
        swprod=sweat_rate/AD
        # Set met
        met=np.random.uniform(met_lower,met_upper)
        # Set wind
        va=np.random.uniform(va_lower,va_upper)
        
        ## Compute critical t for every rh combination ##
        store[:,i]=CRIT_T(rhs,met,va,swprod,n=nt,out=np.zeros(nt))
    
    # Plotting the sensitivity 
    pcl=np.percentile(store,5,axis=1)
    pcu=np.percentile(store,95,axis=1)
    pcmid=np.percentile(store,50,axis=1)
    # Compute reference Tw and MDI
    ref_ta=np.linspace(np.min(store),np.max(store),100)
    ref_rh=np.linspace(np.min(rhs),np.max(rhs),100)
    ref_ta,ref_rh=np.meshgrid(ref_ta,ref_rh)
    nr,nc=ref_ta.shape
    nt=len(ref_ta.flatten())
    ref_tw=utils._TW(nt,ref_ta.flatten()+273.15,
                  ref_rh.flatten(),np.ones(nt)*101300.)-273.15
    # Rehshape
    ref_mdi=(0.75*ref_tw+0.3*ref_ta.flatten()).reshape(nr,nc)
    ref_tw=ref_tw.reshape(nr,nc)
    
    fig,ax=plt.subplots(1,1)
    ax.plot(pcl,rhs,alpha=1,color='grey',linestyle='--')
    ax.plot(pcu,rhs,alpha=1,color='grey',linestyle='--')
    ax.plot(pcmid,rhs,alpha=1,color='black',linestyle='-')
    ax.contour(ref_ta,ref_rh,ref_mdi,levels=[crit_sed,],colors='red')
    ax.contour(ref_ta,ref_rh,ref_tw,levels=[35,],colors='blue')
    ax.grid()
    ax.set_xlim(30,55)
    ax.set_ylim(20,100)
    ax.set_ylabel("Relative humidity (%)")
    ax.set_xlabel("Air temperature ($^{\circ}$C)")
    fig.savefig(figdir+"isosweat-%.0f.png"%isoeff,dpi=300)
    
    
    
    # for i in range(nsim):
        # Compute the wetbulb temp and 
    # ax.plot(store[:,i],rhs,alpha=0.05,color='grey')
