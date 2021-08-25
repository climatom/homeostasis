#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute pattern-scaling coefficients
"""

from numba import njit, prange
import numpy as np
import xarray as xa
import os, sys, src
sys.path.append(".")


# Numba regression (see: 
    # https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4)
@njit(fastmath=True)
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0],deg + 1))
    const = np.ones_like(x)
    mat_[:,0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_
    
@njit(fastmath=True)
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_
 
@njit(fastmath=True)
def fit_poly(x, y, deg):
    # print("Got x: ",x)
    # print("Got y:",y)
    a = _coeff_mat(x, deg)

    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]


@njit(parallel=True,fastmath={"nnan":False})
def reg(grid,covariate,nr,nc,window):
    rates=np.zeros((nr,nc))*np.nan
    for row in prange(nr):
       for col in prange(nc): 
           if np.isnan(grid[0,row,col]): 
               continue
           else:
               # Smooth and compute slope
               x=np.convolve(covariate,np.ones(window))[window-1:1-window]\
                             /np.float(window)
               y=np.convolve(grid[:,row,col],np.ones(window))[window-1:1-window]\
                             /np.float(window)
  
               rates[row,col]=fit_poly(x,y,1)[0]
    return rates

# Smoothing paramater (years)
window=20

# GTAS files
tasdir="/home/lunet/gytm3/Homeostasis/CMIP6/GTAS/"
tasfiles=[tasdir+ii for ii in os.listdir(tasdir) if ".nc" in ii]

# MDI files
#moddir="/home/lunet/gytm3/Homeostasis/CMIP6/Merged/Regridded/"
moddir="/home/lunet/gytm3/Homeostasis/CMIP6/Merged/"
mdi_files=[moddir+ii for ii in os.listdir(moddir) if ".mdi." in ii]

# Outdir (where to save the regression coefficients)
odir="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/"

# Outdir for the regridded regressions
odir_regrid=odir+"Regridded/"

# TW files
tw_files=[moddir+ii for ii in os.listdir(moddir) if ".tw." in ii]

# Target grid 
target_grid="/home/lunet/gytm3/Homeostasis/CMIP6/target_grid.txt"

# Mask (to set ocean pixels to missing)
mask="/home/lunet/gytm3/Homeostasis/CMIP6/mask.nc"

# Read in the tw files
fproc=0
res={}
for ii in range(len(tw_files)):
    # Extract model
    mod_tw=tw_files[ii].split("/")[-1].split(".")[1]
    # Find that model in the tasfiles
    scan=0
    while True:
        if mod_tw not in tasfiles[scan]: scan +=1
        else: gtas_name = tasfiles[scan]; print("Found model for %s"%mod_tw); break
        assert scan <= len(tasfiles), "Can't find gtas for: %s"%mod_tw

    # Read in TW
    tw_f=xa.open_dataset(tw_files[ii])
    mdi_f=xa.open_dataset(tw_files[ii].replace(".tw.",".mdi."))
    tas_f=xa.open_dataset(tw_files[ii].replace(".tw.",".tas_tw."))
    huss_f=xa.open_dataset(tw_files[ii].replace(".tw.",".huss_tw."))
    ps_f=xa.open_dataset(tw_files[ii].replace(".tw.",".ps_tw."))

    # Preallocate if this is first file to be processed
    if fproc==0:
        lat=tw_f["lat"]
        lon=tw_f["lon"]
        out=np.zeros((len(tw_files),len(lat),len(lon)))*np.nan

    # Extract years
    tw_years=tw_f["time"].dt.year.data[:]
    
    # Read in gtas file
    gtas_f=xa.open_dataset(gtas_name)
    
    # And gtas years
    gtas_years=gtas_f["time"].dt.year.data[:]
    
    # Match index
    idx_gtas=np.isin(gtas_years,tw_years)
    
    # Extract gtas to match 
    gtas=np.squeeze(gtas_f["tas"].data[idx_gtas])
    
    # Read in the tw array
    tw=tw_f["tw"].data[:,:,:]
    
    # Compute the regression coefs
    write=xa.DataArray(data=reg(tw,gtas,tw.shape[1],tw.shape[2],window),
                       dims=["lat","lon"],
                       coords={"lat":lat,"lon":lon},name="tw",
                       attrs={"units":"dTdTg"}).\
                       to_netcdf(odir+mod_tw+"_tw.nc")
    del tw
    print("Finished with tw")
    
    # Read in the mdi and repeat the regression
    mdi=mdi_f["mdi"].data[:,:,:]
    write=xa.DataArray(data=reg(mdi,gtas,mdi.shape[1],mdi.shape[2],window),
                       dims=["lat","lon"],
                       coords={"lat":lat,"lon":lon},name="mdi",
                       attrs={"units":"dTdTg"}).\
                       to_netcdf(odir+mod_tw+"_mdi.nc")                       
    del mdi
    print("Finished with mdi")
    
    # Read in tas and huss. We are going to use this to evaluate 
    # RH during peak TW
    tas=tas_f["tas_tw"].data[:,:,:]  
    huss=huss_f["huss_tw"].data[:,:,:]
    ps=ps_f["ps_tw"].data[:,:,:] 
    nt,nr,nc=tas.shape
    satq=src.utils._satQ3d(tas,ps,nt,nr,nc)
    rh=huss/satq
    write=xa.DataArray(data=reg(rh,gtas,nr,nc,window),
                       dims=["lat","lon"],
                       coords={"lat":lat,"lon":lon},name="rh",
                       attrs={"units":"dRHdfracTg"}).\
                       to_netcdf(odir+mod_tw+"_rh.nc")   
    del tas, huss, ps,  rh
    
    # Use cdo to regrid and mask all the files just written
    for fr in ["_tw.nc","_mdi.nc","_rh.nc"]:
        cmd="cdo -O -s mul -remapbil,%s %s %s %s" %(target_grid,
                                                    odir+mod_tw+fr,
                                                    mask,
                                                    odir_regrid+mod_tw+fr)
        fail=os.system(cmd); assert fail==0, "This failed - %s"%cmd 
        
        
    print("Processed model %s"%mod_tw)
    
print("Merging all into ensemble means and stds...")
for fr in ["_tw.nc","_mdi.nc","_rh.nc"]:
    fs=" ".join([odir_regrid+ii for ii in os.listdir(odir_regrid) \
                 if fr in ii])
    oname_mean=odir_regrid+fr.replace("_","ensmean-")
    oname_std=odir_regrid+fr.replace("_","enstd-")
    #Mean
    cmd="cdo -O -s ensmean %s %s" % (fs,oname_mean)
    fail=os.system(cmd); assert fail==0, "This failed - %s"%cmd
    #Std
    cmd="cdo -O -s ensstd %s %s" % (fs,oname_std)
    fail=os.system(cmd); assert fail==0, "This failed - %s"%cmd    
    

    
    
    
    







