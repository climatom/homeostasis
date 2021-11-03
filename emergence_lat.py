#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we assess area >crit as f(lat,dt). If signal emerges first outside of the
tropics, well...
"""
import pickle
import numpy as np, xarray as xa
import matplotlib.pyplot as plt
from src import utils

# ============================================================================
# Parameters and script settings
# ============================================================================


# Reference files
tw_ref_f="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_TW_1981-2020.p"
mdi_ref_f="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_1981-2020.p"

# Scaling
scale_tw_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-tw.nc"
scale_mdi_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-mdi.nc"

# Cell area file
cell_f="/home/lunet/gytm3/Homeostasis/CMIP6/cellarea.nc"

# Mask file (1=land; 0=ocean)
mf="/home/lunet/gytm3/Homeostasis/CMIP6/mask.nc"

# Directory to save figures to 
figdir="/home/lunet/gytm3/Homeostasis/Figures/"

# Latitude band for plot
dlat=2.5

# Temp increment for plot
dt=0.25

# Upper limit of warming experiment
tu=9

# Reference/historical years
yr_ref_start=1995
yr_ref_stop=2014

# Warming amount of ref period
dT=0.87

# Critical values
crit_tw=35.
crit_mdi=28/0.74

# ============================================================================
# MAIN
# ============================================================================

# Read reference tw/mdi and compute the mean over the reference years
ref_years=np.arange(1981,2021)
ref_idx=np.logical_and(ref_years>=yr_ref_start,ref_years<=yr_ref_stop)
tw=pickle.load(open(tw_ref_f,"rb"))[ref_idx,:,:].max(axis=0)-273.15
mdi=pickle.load(open(mdi_ref_f,"rb"))[ref_idx,:,:].max(axis=0)
cellarea=xa.open_dataset(cell_f)
mask=xa.open_dataset(mf)
lats=cellarea.latitude[:].data
lons=cellarea.longitude[:].data
lons,lats=np.meshgrid(lons,lats)
cellarea=np.squeeze(cellarea.cell_area[:,:].data*mask.d2m[:,:].data)

# Scaling
slope_mdi=xa.open_dataset(scale_mdi_f).mdi[:,:].data
slope_tw=xa.open_dataset(scale_tw_f).tw[:,:].data

# Logic here is to iterate over lats and warming amounts
dts=np.arange(tu,0,-dt) # Remember to add dT for warming since PI
dlats=np.arange(-60,60+dlat,dlat)
out_mdi=np.zeros((len(dts),len(dlats)-1))*np.nan
out_tw=np.zeros((len(dts),len(dlats)-1))*np.nan
ax_lat=np.zeros(len(dlats)-1)
for t in range(len(dts)):
    # Gen scens
    scen_mdi=mdi+dts[t]*slope_mdi
    scen_tw=tw+dts[t]*slope_tw
    for l in range(1,len(dlats)):
        idx=np.logical_and(lats>=dlats[l-1],lats<dlats[l])
        valid_area=np.nansum(cellarea[idx])
        if valid_area>0:
            idx_mdi=np.logical_and(scen_mdi>crit_mdi,idx)
            idx_tw=np.logical_and(scen_tw>crit_tw,idx)
            out_mdi[t,l-1]=np.sum(cellarea[idx_mdi])/valid_area*100. #  %
            out_tw[t,l-1]=np.sum(cellarea[idx_tw])/valid_area*100. #  %
        if t==0:
            ax_lat[l-1]=np.mean([dlats[l-1],dlats[l]])


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
fig,ax=plt.subplots(1,2)
fig.set_size_inches(8,4)
x,y=np.meshgrid(ax_lat,dts)
y+=dT # Now in warming since PI
levs=np.linspace(0,np.nanmax(out_mdi))
p1=ax.flat[0].contourf(x,y,out_mdi,cmap="Reds",levels=levs)    
ax.flat[0].contour(x,y,out_mdi,colors=["k",],levels=[0.1,])   
ax.flat[1].contourf(x,y,out_tw,cmap="Reds",levels=levs)      
ax.flat[1].contour(x,y,out_tw,colors=["k",],levels=[0.1,]) 
ax.flat[0].set_ylabel("Global warming since pre-indstrial ($^{\circ}$C)")
ax.flat[0].set_xlabel("Latitude ($^{\circ}$N)")
ax.flat[1].set_xlabel("Latitude ($^{\circ}$N)")
ax.flat[0].grid()
ax.flat[1].grid()   
ax.flat[1].axvline(20,linestyle='--',color='k') 
ax.flat[1].axvline(-20,linestyle='--',color='k') 
ax.flat[0].axvline(20,linestyle='--',color='k') 
ax.flat[0].axvline(-20,linestyle='--',color='k') 
plt.subplots_adjust(right=0.8)
cax=fig.add_axes([0.82,0.12,0.02,0.76])
cb=plt.colorbar(p1,orientation='vertical',cax=cax)
cb.set_ticks([15,30,45,60,75])
cb.set_label("Area >threshold (%)")
fig.savefig(figdir+"LatEmergence.png",dpi=300)