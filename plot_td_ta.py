#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we plot Tdew and Tair at the time of max MDI
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# File/diectory names
ta_name="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_T_1981-2020.p"
td_name="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_TD_1981-2020.p"
mdi_name="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI.p"
figdir="/home/lunet/gytm3/Homeostasis/Figures/"

# Percentile to limit plot (only plot values above this: )
75

# Read
ta=pickle.load(open(ta_name,"rb")).max(axis=0)
ta[ta<0]=np.nan; ta-=273.15 # Now missing => NaN and in C
td=pickle.load(open(td_name,"rb")).max(axis=0)
td[td<0]=np.nan; td-=273.15 # Now missing => NaN and in C
mdi=pickle.load(open(mdi_name, "rb" )).max(axis=0)
mdi-=(273.15*0.75+273.15*0.3) 

# Correct to C (values inflated from K step)
idx=mdi<np.nanpercentile(mdi,75.)

# Specify lat/lon
lats=np.arange(90,-90.1,-0.1)
lons=np.arange(0,360,0.1)

# Plot
fig = plt.figure(figsize=(6, 6))
# Ta
z=ta.copy(); ta[idx]=np.nan
ax = fig.add_subplot(2, 1, 1, projection=crs.Robinson())
ax.set_global()
ax.coastlines(linewidth=0.2)
p1=ax.pcolormesh(lons,lats,ta,cmap="turbo", transform=crs.PlateCarree())
ax.gridlines(draw_labels=False)
ax.add_feature(cf.BORDERS,linewidth=0.15)
plt.subplots_adjust(hspace=0.27)
cax=fig.add_axes([0.36,0.54,0.31,0.02])
cb=plt.colorbar(p1,orientation='horizontal',cax=cax)

# Tdew
z=td.copy(); td[idx]=np.nan
ax = fig.add_subplot(2, 1, 2, projection=crs.Robinson())
ax.set_global()
ax.coastlines(linewidth=0.2)
p2=ax.pcolormesh(lons,lats,td,cmap="turbo", transform=crs.PlateCarree())
ax.gridlines(draw_labels=False)
ax.add_feature(cf.BORDERS,linewidth=0.15)

plt.subplots_adjust(bottom=0.2)
cax=fig.add_axes([0.36,0.15,0.31,0.02])
cb=plt.colorbar(p2,orientation='horizontal',cax=cax)
cax.set_xlabel("$^{\circ}$C")
fig.savefig(figdir+"MaxTaTd.png",dpi=300)