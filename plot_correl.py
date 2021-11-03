#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot correlations between (20-year smoothed) ann-max Tw/mdi and global mea
air temperature
"""

import pickle, os
import numpy as np, xarray as xa
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# import matplotlib.ticker as mticker

# File / folder names
mdi_corr_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/_mdi_correl-ensmean.nc"
tw_corr_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/_tw_correl-ensmean.nc"
figdir="/home/lunet/gytm3/Homeostasis/Figures/"

# Read in
mdi=xa.open_dataset(mdi_corr_f)
tw=xa.open_dataset(tw_corr_f)
lats=mdi.latitude[:].data
lons=mdi.longitude[:].data
lons,lats=np.meshgrid(lons,lats)
r_mdi=mdi.r[:].data
r_tw=tw.r[:].data

fig = plt.figure(figsize=(6, 6))
# Tw
ax = fig.add_subplot(2, 1, 1, projection=crs.Robinson())
ax.set_global()
ax.coastlines(linewidth=0.2)
p=ax.pcolormesh(lons,lats,r_tw,cmap="turbo", transform=crs.PlateCarree(),
            vmin=0.9,
            vmax=1)
ax.gridlines(draw_labels=False)
ax.add_feature(cf.BORDERS,linewidth=0.15)

# mdi
ax = fig.add_subplot(2, 1, 2, projection=crs.Robinson())
ax.set_global()
ax.coastlines(linewidth=0.2)
p=ax.pcolormesh(lons,lats,r_mdi,cmap="turbo", transform=crs.PlateCarree(),
            vmin=0.9,
            vmax=1)
ax.gridlines(draw_labels=False)
ax.add_feature(cf.BORDERS,linewidth=0.15)
plt.subplots_adjust(bottom=0.2)
cax=fig.add_axes([0.36,0.15,0.31,0.02])
cb=plt.colorbar(p,orientation='horizontal',cax=cax)
cax.set_xlabel("Pearson r")
fig.savefig(figdir+"Correls.png",dpi=300)