#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script uses the pattern scaling Tw and MDI coefficients to assess
how much warming is required until the city has a mean annual max > threshold
"""
import pickle, xarray as xa, numpy as np, pandas as pd
import numba as nb
from numba import njit, prange
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# (simple) Functions 
# ============================================================================ 

@njit(fastmath=True)
def dist(lat1,lng1,lat2,lng2):
    
    """ 
    Calculate the great-circle distance bewteen two points on the Earth surface.

    :input: two scalars (lat1,lng1) and two vectors (lat2,lng2)

    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))

    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.

    """
    R = 6378.1370 # equatorial radius of Earth (km)
    # convert all latitudes/longitudes from decimal degrees to radians
    lat1=np.radians(lat1); lat2=np.radians(lat2); lng1=np.radians(lng1); 
    lng2=np.radians(lng2)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * R * np.arcsin(np.sqrt(d))
    return h  # in kilometers

# Find the nearest grid cell to city that is NOT a missing value
@njit(fastmath={"nnan":False},parallel=True)
def closest_row_col(grid_lat,grid_lon,target_lat,target_lon,grid_res,n,mask,
                    nr,nc):
    row_col=np.zeros((n,2),dtype=nb.int64)
    for i in prange(n):
        row_col[i,0]=np.floor((grid_lat[0,0]-target_lat[i])/grid_res)
        row_col[i,1]=np.floor((target_lon[i]-grid_lon[0,0])/grid_res)
        
        # More costly search based on distance to cells
        if np.isnan(mask[row_col[i,0],row_col[i,1]]):
            # print("[... intense search triggered...")
            d=dist(target_lat[i],target_lon[i],grid_lat.flatten(),
                    grid_lon.flatten())
            d=d.reshape(nr,nc)
            min_d=1e10
            for r in prange(nr):
                for c in prange(nc):
                    if np.isnan(mask[r,c]): continue
                    if d[r,c] < min_d:
                            row_col[i,0]=r
                            row_col[i,1]=c                            
                            min_d=d[r,c]
            # print("...concluded]")                       
    return row_col



# ============================================================================

# ============================================================================
# Parameters and script settings
# ============================================================================
cityf="/home/lunet/gytm3/Homeostasis/Data/city_in.csv"

# Reference files
tw_ref_f="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_TW_1981-2020.p"
mdi_ref_f="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_1981-2020.p"

# Pattern scaling files
scale_tw_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-tw.nc"
scale_mdi_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-mdi.nc"
scale_rh_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-rh.nc"
un_mdi_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/enstd-mdi.nc"
un_tw_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/enstd-tw.nc"

# Mask file (1=land, 0=ocean)
mf="/home/lunet/gytm3/Homeostasis/CMIP6/mask.nc"

# Reference/historical years
yr_ref_start=1995
yr_ref_stop=2014

# Warming of 1995-2014 relative to first 20 years in HadCRT5?
dT=0.87

# Critical values
crit_tw=35.
crit_mdi=28/0.74

# ============================================================================
# MAIN
# ============================================================================


# Read city data into array
city=pd.read_csv(cityf,index_col=0)

# Read reference tw/mdi and compute the mean over the reference years
ref_years=np.arange(1981,2021)
ref_idx=np.logical_and(ref_years>=yr_ref_start,ref_years<=yr_ref_stop)
ref_tw=pickle.load(open(tw_ref_f,'rb'))[ref_idx,:,:].mean(axis=0)-273.15
ref_mdi=pickle.load(open(mdi_ref_f,'rb'))[ref_idx,:,:].mean(axis=0)

# Read in the mask 
mask=xa.open_dataset(mf)
grid_lat=mask.latitude[:].data
grid_lon=mask.longitude[:].data
mask=np.squeeze(mask["d2m"][:,:].data)
grid_res=grid_lat[0]-grid_lat[1]
grid_lon,grid_lat=np.meshgrid(grid_lon,grid_lat)
nr,nc=grid_lat.shape

# Read in the slopes and uncertainty
scale_tw=xa.open_dataset(scale_tw_f)
scale_mdi=xa.open_dataset(scale_mdi_f)
un_tw=xa.open_dataset(un_tw_f)
un_mdi=xa.open_dataset(un_mdi_f)

# Get row/col indices of closest cells to cities
target_lat=city["lat"].values[:]
target_lon=city["lon"].values[:]
n=len(city)
print("Finding city row/cols...")
row_col=\
closest_row_col(grid_lat,grid_lon,target_lat,target_lon,grid_res,n,mask,
                nr,nc).astype(np.int)

# Use these indices to extract ref values
city_tw_ref=ref_tw[row_col[:,0],row_col[:,1]]
city_mdi_ref=ref_mdi[row_col[:,0],row_col[:,1]]

# .. tw slope + uncertainty
city_tw_slope=np.squeeze(scale_tw["tw"][:,:]).data[row_col[:,0],row_col[:,1]]
city_tw_un=np.squeeze(un_tw["tw"][:,:]).data[row_col[:,0],row_col[:,1]]
city_mdi_slope=np.squeeze(scale_mdi["mdi"][:,:]).data[row_col[:,0],row_col[:,1]]
city_mdi_un=np.squeeze(un_mdi["mdi"][:,:]).data[row_col[:,0],row_col[:,1]]

# Now compute the gtas warming required to push mdi and tw over the limit
# This is simply (thresh-ref)/slope
city["dt_tw"]=(crit_tw-city_tw_ref)/city_tw_slope
city["dt_mdi"]=(crit_mdi-city_mdi_ref)/city_mdi_slope

# Lower/upper
city["dt_tw_upper"]=(crit_tw-city_tw_ref)/(city_tw_slope-city_tw_un)
city["dt_tw_lower"]=(crit_tw-city_tw_ref)/(city_tw_slope+city_tw_un)
city["dt_mdi_upper"]=(crit_mdi-city_mdi_ref)/(city_tw_slope-city_mdi_un)
city["dt_mdi_lower"]=(crit_mdi-city_mdi_ref)/(city_tw_slope+city_mdi_un)

# Closest on mdi? -- Give top 100
city_mdi_sort=city.sort_values(by="dt_mdi").iloc[:100]
city_tw_sort=city.sort_values(by="dt_tw").iloc[:100]
