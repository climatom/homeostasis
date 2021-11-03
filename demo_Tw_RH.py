#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we compare maximum Tw computed with daily mean RH and daily max T
with daily max T and concurrent RH
"""
import numpy as np, xarray as xa
import matplotlib.pyplot as plt
from src import utils
from scipy.interpolate import interpn
from matplotlib import cm
from matplotlib.colors import Normalize 
import statsmodels.api as sm

def density_scatter( x , y, fig=None, ax=None,sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , 
                data , np.vstack([x,y]).T , method = "splinef2d", 
                bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs, )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm,**kwargs), ax=ax,)
    cbar.ax.set_ylabel('Density')

    return ax

def harmReg(x,y):
    """
    Fit single harmonic to y, using x as index
    """
    x=x/np.max(x)*np.pi*2 # Now now in range 0->2*pi
    xfit=np.column_stack((np.cos(x),np.sin(x)))
    xfit=sm.add_constant(xfit)
    # Fit the model 
    mod = sm.OLS(y,xfit).fit()
    # Compute the clim for the return index
    ref=np.linspace(0,np.pi*2,len(x))
    xpred=sm.add_constant(np.column_stack((np.cos(ref),np.sin(ref))))
    sim=mod.predict(xpred)
    return ref/(2*np.pi)*24.,sim 
    
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

# Directory to save figures to 
figdir="/home/lunet/gytm3/Homeostasis/Figures/"
fin="/home/lunet/gytm3/Homeostasis/ERA5Land/TwRHDemo"
data=xa.open_dataset(fin)
nt,nr,nc=data.t2m.shape
rh=xa.apply_ufunc(
    utils._dewRH3D,
    nt,nr,nc,
    data["t2m"],data["d2m"])
rh=xa.Dataset({
    "rh":rh*100.
    })

# Compute daily max T and extract rh at concurrent time
doy=data.time.dt.dayofyear
udoy=np.unique(doy)
nd=len(udoy)
max_t=np.zeros((nd,nr,nc))
max_rh=np.zeros((nd,nr,nc))
mean_rh=np.zeros((nd,nr,nc))
tw_mean_rh=np.zeros((nd,nr,nc))
tw_max_rh=np.zeros((nd,nr,nc))
tw_mean_rh_stull=np.zeros((nd,nr,nc))
tw_max_rh_stull=np.zeros((nd,nr,nc))
demo_t=np.zeros((8,927))*np.nan
demo_rh=np.zeros((8,927))*np.nan
demo_tw=np.zeros((8,927))*np.nan
demo_p=np.zeros((8,927))*np.nan
demo_tw_mean=np.zeros((927))*np.nan
demo_tw_max=np.zeros((927))*np.nan
demo_time=np.zeros((8,927))*np.nan
lon2,lat2=np.meshgrid(data.longitude.values,data.latitude.values)
time_ref=np.arange(0,24,3)
hit_no=0
for i in range(len(udoy)):
    ti=data.t2m[doy==udoy[i],:,:].data
    rhi=rh.rh[doy==udoy[i],:,:].data
    idx=np.argmax(ti,axis=0)
    
    max_t[i,:,:]=np.take_along_axis(ti,idx[None,:,:],axis=0)
    max_rh[i,:,:]=np.take_along_axis(rhi,idx[None,:,:],axis=0)
    mean_rh[i,:,:]=np.mean(rhi,axis=0)   
    p=np.mean(data["sp"][doy==udoy[i],:,:],axis=0)
    
    tw_mean_rh[i,:,:]=utils._TW2D(nr,nc,max_t[i,:,:],mean_rh[i,:,:],p.values)-273.15
    tw_max_rh[i,:,:]=utils._TW2D(nr,nc,max_t[i,:,:],max_rh[i,:,:],p.values)-273.15
    tw_mean_rh_stull[i,:,:]=utils._TWStull(max_t[i,:,:]-273.15,mean_rh[i,:,:])
    tw_max_rh_stull[i,:,:]=utils._TWStull(max_t[i,:,:]-273.15,max_rh[i,:,:])
    
    # Test for exceedance
    idx=tw_mean_rh[i,:,:]>=35
    if idx.any():
        rows,cols=np.nonzero(idx)
        sub_t=ti[:,rows,cols]
        sub_rh=rhi[:,rows,cols]
        sub_tw_mean=tw_mean_rh[i,rows,cols]
        sub_tw_max=tw_max_rh[i,rows,cols]
        sub_p=data["sp"][doy==udoy[i],:,:].data[:,rows,cols]
        sub_lon=lon2[rows,cols]
        for item in range(len(rows)):
                demo_t[:,hit_no]=sub_t[:,item]
                demo_rh[:,hit_no]=sub_rh[:,item]
                demo_p[:,hit_no]=sub_p[:,item]
                demo_tw[:,hit_no]=utils._TW(8,demo_t[:,hit_no],demo_rh[:,hit_no],
                                            demo_p[:,hit_no])
                demo_tw_mean[hit_no]=sub_tw_mean[item]
                demo_tw_max[hit_no]=sub_tw_max[item]
                demo_time[:,hit_no]=(time_ref+sub_lon[item]/15.)%24.
                hit_no+=1
    print("processed day %.0f"%udoy[i])

# Find max difference where tw_max_rh>35
d=tw_max_rh-tw_mean_rh
idx=tw_mean_rh>35
sub=tw_mean_rh[idx]

# Setup plots
fig,ax=plt.subplots(2,2)
fig.set_size_inches(7,5)

### 
# Figure out the density/plot for tw (normal)
###
x=(tw_max_rh.max(axis=0)).flatten()
y=(tw_mean_rh.max(axis=0)).flatten()
refl=np.linspace(0,40,100)
# Calculate the point density
vidx=np.logical_and(~np.isnan(x),~np.isnan(y))
xy = np.vstack([x[vidx],y[vidx]])
# Time max 
cscat=density_scatter(xy[0,:], xy[1,:], fig=fig, ax=ax.flat[1],
                   bins=[30,30],cmap="turbo" )
ps=np.polyfit(xy[0,:],xy[1,:],1)
yi1=np.polyval(ps,refl)
ps_rev=np.polyfit(xy[1,:],xy[0,:],1)


### 
# Repeat, Stull
###
x=(tw_max_rh.max(axis=0)).flatten()
y=(tw_max_rh_stull.max(axis=0)).flatten()
# Calculate the point density
vidx=np.logical_and(~np.isnan(x),~np.isnan(y))
xy = np.vstack([x[vidx],y[vidx]])
# Time max 
cscat=density_scatter(xy[0,:], xy[1,:], fig=fig, ax=ax.flat[0],
                   bins=[30,30],cmap="turbo" )
yi2=np.polyval(np.polyfit(xy[0,:],xy[1,:],1),refl)

### 
# Plot profiles
###
# tplot_mean=np.mean(demo_t,axis=1)
# rhplot_mean=np.mean(demo_rh,axis=1)
# twplot_mean=np.mean(demo_rh,axis=1)
ax2=ax.flat[3].twinx()
for k in range(demo_time.shape[1]):
    xi,tplot=harmReg(demo_time[:,k],demo_t[:,k]-273.15)
    xi,rhplot=harmReg(demo_time[:,k],demo_rh[:,k])
    xi,twplot=harmReg(demo_time[:,k],demo_tw[:,k]-273.15)
    ax.flat[3].plot(xi,tplot,alpha=0.01,color='r')
    # ax.flat[2].plot(xi,twplot,alpha=0.01,color='k')
    ax2.plot(xi,rhplot,alpha=0.01,color='b')

ax.flat[3].grid()
ax.flat[3].set_xlabel("Local time (hour)")
ax.flat[3].set_ylabel("Air temperature ($^{\circ}$C)",color='red')
ax2.set_ylabel("Relative humidity (%)",color='blue')

# Contour of sensitivity
reft=np.linspace(15,50,100)
refrh=np.linspace(1,100,100)
reft,refrh=np.meshgrid(reft,refrh)
refp=np.ones(reft.shape)*101300.
# _TW2D(nr,nc,ta,rh,p)
reftw=utils._TW2D(100,100,reft+273.15,refrh,refp)-273.15
CS=ax.flat[2].contour(reft,refrh,reftw,levels=[10,15,20,25,30,35,40],colors='k')
ax.flat[2].clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=8)
ax.flat[2].set_xlabel("Air temperature ($^{\circ}$C)")
ax.flat[2].set_ylabel("Relative humidity (%)")



### Tidy plots
ax.flat[1].grid()
ax.flat[1].set_xlabel("Tw, concurrent RH ($^{\circ}$C)")
ax.flat[1].set_ylabel("Tw, mean RH ($^{\circ}$C)")
ax.flat[1].plot(refl,refl,color='red')
ax.flat[1].set_xlim([0,40])
ax.flat[1].set_ylim([0,40])
ax.flat[1].axhline(35,color='r',linestyle=":")
ax.flat[1].axvline(35,color='r',linestyle=":")
ax.flat[1].plot(refl,yi1,color='k')


ax.flat[0].grid()
ax.flat[0].set_xlabel("Tw, iteration ($^{\circ}$C)")
ax.flat[0].set_ylabel("Tw, Stull ($^{\circ}$C)")
ax.flat[0].plot(refl,refl,color='red')
ax.flat[0].set_xlim([0,40])
ax.flat[0].set_ylim([0,40])
ax.flat[0].axhline(35,color='r',linestyle=":")
ax.flat[0].axvline(35,color='r',linestyle=":")
ax.flat[0].plot(refl,yi2,color='k')
ax.flat[3].set_xlim(0,24)
plt.tight_layout()
plt.gcf().savefig(figdir+"MeanRHDemo.png",dpi=300)

# Update on difference in means
dm=np.nanmean(y)-np.nanmean(x)
dx=np.nanpercentile(y,99)-np.nanpercentile(x,99)
print("diff mean = %.2fC, diff 99 percentile = %.2fC" % (dm,dx))
print("Ls fit says when mean method gives 35C, concurrent = %.2fC"\
      % np.polyval(ps_rev,[35.]))



## for lecture
fig,ax=plt.subplots(1,1)
CS=ax.contour(reft,refrh,reftw,levels=[10,15,20,25,30,35,40],colors='k')
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=8)
ax.set_xlabel("Air temperature ($^{\circ}$C)")
ax.set_ylabel("Relative humidity (%)")
fig.savefig(figdir+"TeachingDemo.png",dpi=300)
    
