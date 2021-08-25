#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use CMIP6 pattern-scaled coefficients and observed heat extremes to 
generate scenarios
"""
import xarray as xa, numpy as np
import src, pickle
from netCDF4 import Dataset
import matplotlib.pyplot as plt

thresh_tw=35.
thresh_mdi=28/0.74

# Obs file
obs_tw_f="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_TW_1981-2020.p"
obs_mdi_f="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_1981-2020.p"

# Pattern scaling files
scale_tw_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-tw.nc"
scale_mdi_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-mdi.nc"
un_mdi_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/enstd-mdi.nc"
un_tw_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/enstd-tw.nc"
# Read in
scale_tw=Dataset(scale_tw_f)
scale_mdi=Dataset(scale_mdi_f)
un_tw=Dataset(un_tw_f)
un_mdi=Dataset(un_mdi_f)

# cell area file
cell_f="/home/lunet/gytm3/Homeostasis/CMIP6/cellarea.nc"
# Read in
cellarea=Dataset(cell_f)
cellarea=cellarea.variables["cell_area"][:,:].data

# Read in these pickled objects, and immediately average last 20 years
obs_tw=pickle.load(open(obs_tw_f,"rb"))[-20:,:,:].mean(axis=0)-273.15
obs_mdi=pickle.load(open(obs_mdi_f,"rb"))[-20:,:,:].mean(axis=0)

# Create scenarios
scens=np.arange(0.1,4,0.2)
nscens=len(scens)
out=np.zeros((nscens,6))*np.nan
for i in range(nscens):
    proj_tw=obs_tw+scens[i]*scale_tw.variables["tw"][:,:].data
    proj_tw_lower=obs_tw+(scens[i]*(scale_tw.variables["tw"][:,:].data - \
                          un_tw.variables["tw"][:,:].data))
    proj_tw_upper=obs_tw+(scens[i]*(scale_tw.variables["tw"][:,:].data + \
                          un_tw.variables["tw"][:,:].data))
        
    proj_mdi=obs_mdi+scens[i]*scale_mdi.variables["mdi"][:,:].data
    proj_mdi_lower=obs_mdi+(scens[i]*(scale_mdi.variables["mdi"][:,:].data - \
                          un_mdi.variables["mdi"][:,:].data))
    proj_mdi_upper=obs_mdi+(scens[i]*(scale_mdi.variables["mdi"][:,:].data + \
                          un_mdi.variables["mdi"][:,:].data))   
    
    idx_tw=proj_tw>thresh_tw
    idx_tw_lower=proj_tw_lower>thresh_tw
    idx_tw_upper=proj_tw_upper>thresh_tw
    
    idx_mdi=proj_mdi>thresh_mdi
    idx_mdi_lower=proj_mdi_lower>thresh_mdi
    idx_mdi_upper=proj_mdi_upper>thresh_mdi
    
    out[i,0]=np.nansum(cellarea[idx_tw_lower])/1e6
    out[i,1]=np.nansum(cellarea[idx_tw])/1e6
    out[i,2]=np.nansum(cellarea[idx_tw_upper])/1e6
    out[i,3]=np.nansum(cellarea[idx_mdi_lower])/1e6
    out[i,4]=np.nansum(cellarea[idx_mdi])/1e6
    out[i,5]=np.nansum(cellarea[idx_mdi_upper])/1e6

    print("Finished with warming scenario %.1fC"%scens[i])
    
# Plot it
dt=1.# Warming of 2000-2020, relative to PI
fig,ax=plt.subplots(1,1)
ax.fill_between(scens+dt,out[:,0],out[:,2],color="blue",alpha=0.2)
ax.fill_between(scens+dt,out[:,3],out[:,5],color="red",alpha=0.2)
ax.plot(scens+dt,out[:,1],color="blue")
ax.plot(scens+dt,out[:,4],color="red")
ax.grid()
ax.set_xlabel("Warming since PI ($^{\circ}$C)")
ax.set_ylabel("Area above threshold (km$^{2}$)")