#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterates over all of the merged CMIP6 projections and generates (bias corrected)
predictions of future humid heat via the change factor method. 
"""

import os, pickle
import numpy as np
import xarray as xa
import matplotlib.pyplot as plt
import warnings
from numba import njit, prange
warnings.filterwarnings("ignore")

# ============================================================================
# (simple) Functions 
# ============================================================================ 
@njit(fastmath={"nnan":False},parallel=True)
def runxceed(window,years,nt,nr,nc,grid,refgrid,tas,crit_value,cell_area):
    # Here we compute rolling means, always differenced from the initial 
    # window timesteps. These differences are added on to the refgrid/tas
    # to generate projections; the projections are tested for exceedance
    # of crit value as a function of warming. We output the total area that
    # exceeds the threshold (km**2) for each time step. 
    out=np.zeros((nt-window,4))
    for i in prange(window,nt):
        # Idea here is to loop over all cells for this time slice, compute 
        # the mean, and if it exceeds crit_value, store cell area. 
        dummy=np.zeros((nr,nc))*np.nan
        out[i-window,0]=years[i-window]
        out[i-window,1]=years[i-1]
        for _r in prange(nr):
            for _c in prange(nc):
                if np.isnan(grid[i-window,_r,_c]): continue
            
                if (np.mean(grid[i-window:i,_r,_c])-\
                    np.mean(grid[:window,_r,_c])+refgrid[_r,_c]) > crit_value:
                    dummy[_r,_c]=cell_area[_r,_c]
        
        out[i-window,2]=np.nanmean(tas[i-window:i])-np.nanmean(tas[:window]) 
        out[i-window,3]=np.nansum(dummy)
    return out

# ============================================================================
# Parameters and script settings
# ============================================================================

# Yr min and yr max
yr_min=1995
yr_max=2200

# Reference/historical years
yr_ref_start=1995
yr_ref_stop=2014

# gtas range to assess response over
x=np.linspace(1.5,5.5,100)

# Warming of 1995-2014 relative to first 20 years in HadCRT5?
dT=0.87

# Which warming amounts do we want to examine in detail?
query_warm=np.array([1.5,2,3,4,5])-dT
meta={}; 
for i in query_warm: meta[i+dT]={}

# Can be this close to count as having the "same climate" as out target points
tol=0.1

# Critical values
crit_tw=35.
crit_mdi=28/0.74

# Window  length defining the 'climate'
window=20

# Reference file
tw_ref_f="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_TW_1981-2020.p"
mdi_ref_f="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_1981-2020.p"

# Pattern scaling files
scale_tw_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-tw.nc"
scale_mdi_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-mdi.nc"
scale_rh_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/ensmean-rh.nc"
un_mdi_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/enstd-mdi.nc"
un_tw_f="/home/lunet/gytm3/Homeostasis/CMIP6/Regressions/Regridded/enstd-tw.nc"

# Cell area file
cell_f="/home/lunet/gytm3/Homeostasis/CMIP6/cellarea.nc"

# Directory holding netcdf files:
di="/home/lunet/gytm3/Homeostasis/CMIP6/Merged/Regridded/"

# Mask file (1=land; 0=ocean)
mf="/home/lunet/gytm3/Homeostasis/CMIP6/mask.nc"

# Directory (and files) with the global mean T series
di_tas ="/home/lunet/gytm3/Homeostasis/CMIP6/GTAS/"

# Directory to save figures to 
figdir="/home/lunet/gytm3/Homeostasis/Figures/"

# Do we need to generate tw/mdi/meta summaries, or can we skip?
skip_process=True
# Are we plotting?years
plot=True

# ... Otherwise the following files *must* exist
resdir="/home/lunet/gytm3/Homeostasis/CMIP6/Results/"
fout_mdi=resdir+"mdi.p"
fout_tw=resdir+"mtw.p"
fout_tas=resdir+"gtas.p"
fout_years=resdir+"years.p"
fout_ps_tw=resdir+"ps_tw.p"
fout_ps_mdi=resdir+"ps_mdi.p"
bundlenames=[fout_tw,fout_mdi,fout_tas,fout_years,fout_ps_tw,fout_ps_mdi]
# ============================================================================
# Main
# ============================================================================

# ----------------------------------------------------------------------------
# Init processing
# ----------------------------------------------------------------------------

if not skip_process:
    
    print("***********\nProcessing\n***********")
    
    # Read reference tw/mdi and compute the mean over the reference years
    ref_years=np.arange(1981,2021)
    ref_idx=np.logical_and(ref_years>=yr_ref_start,ref_years<=yr_ref_stop)
    ref_tw=pickle.load(open(tw_ref_f,'rb'))[ref_idx,:,:].mean(axis=0)-273.15
    ref_mdi=pickle.load(open(mdi_ref_f,'rb'))[ref_idx,:,:].mean(axis=0)
    
    # Read in the cell areas
    cellarea=xa.open_dataset(cell_f)
    lat=cellarea["latitude"][:]
    lon=cellarea["longitude"][:]
    cellarea=np.squeeze(cellarea["cell_area"][:,:].data)/1e6 # km**2
    mask=~np.isnan(ref_tw)
    total_area=np.sum(cellarea[mask])
    
    # Shape of the grid
    nr,nc=ref_tw.shape
    
    # Find gtas files
    fs=[di_tas+i for i in os.listdir(di_tas) if "tropics" not in i and ".nc" in i]
    
    # Find the CMIP6 Tw and mdi files
    twfs=[di+ii for ii in os.listdir(di) if ".tw." in ii]
    mdifs=[di+ii for ii in os.listdir(di) if ".mdi." in ii]
    
    # Preallocate the raw "out" arrays: 
    # dims = [yr_max-yr_min+1,nmods], noting than nmods is at least len(twfs)*2
    out_tw=np.zeros((yr_max-yr_min+1-window,len(twfs)))*np.nan
    out_mdi=np.zeros(out_tw.shape)*np.nan
    out_tas=np.zeros((out_tw.shape))*np.nan
    
    # Iterate over the tas files, finding the matching Tw and mdi files
    count=0
    for item_no in range(len(fs)):
        stem="."+"_".join(fs[item_no].split("/")[-1].split("_")[:-3])+"."
        skip=True
        for cand in twfs:
            if stem in cand: skip=False; break
        if skip: continue  
    
        # 'cand' is the full tw path. Read in here
        twf=xa.open_dataset(cand)
        tw_years=twf.time.dt.year
        
        # mdi name is, well, easy to guess. Note that mdi years match tw years
        mdif=xa.open_dataset(cand.replace(".tw.",".mdi."))
        
        # Read in gtas
        gtf=xa.open_dataset(fs[item_no])
        tas_years=gtf.time.dt.year
        
        # Get indices -- matching tas/tw years, plus no less/more than yr_min/max
        tw_idx=np.logical_and(\
               np.logical_and(tw_years>=yr_min,tw_years<=yr_max),
               tw_years.isin(tas_years))
        tas_idx=np.logical_and(\
               np.logical_and(tas_years>=yr_min,tas_years<=yr_max),
               tas_years.isin(tw_years))
    
        # Read inthe right range of years
        tw=np.squeeze(twf["tw"][tw_idx,:,:].data)
        tas=np.squeeze(gtf["tas"][tas_idx].data)
        
        #---------------------------------
        # Compute exceedances as f(gtas)
        #---------------------------------
        nt=tw.shape[0]
        ntout=len(tas_years[tas_idx])-window
        out_tw[:ntout,count]=runxceed(window,
                        tas_years[tas_idx].data,nt,nr,nc,tw,ref_tw,tas,
                        crit_tw,cellarea)[:,-1] #Area xceed
        
        del tw # make space
        #---------------------------------
        
        #---------------------------------
        # Repeat mdi
        #---------------------------------
        mdi=np.squeeze(mdif["mdi"][tw_idx,:,:].data)
        _=runxceed(window,tas_years[tas_idx].data,
                         nt,nr,nc,mdi,ref_mdi,tas,crit_mdi,cellarea)
        out_mdi[:ntout,count]=_[:,-1] # Area xceed
        out_tas[:ntout,count]=_[:,2]+dT # dtas since PI
        
        # Take opportunity to assess when model showed warming closest to our
        # target
        for amount in query_warm:
            row=np.argmin(np.abs(_[:,2]-amount))
            if np.abs(_[row,2]-amount)>tol: continue
            meta[amount+dT][cand]=_[row,:2]
        
        count+=1
        #---------------------------------  
    
        
        print("Finished with file: %.0f"%count)
        
        del mdi # Make space   
        
    #-------------------------------------------------
    # Pattern-scaling approach (outside file loop)
    #-------------------------------------------------
    # Read in mean slope and uncertainty
    scale_tw=xa.open_dataset(scale_tw_f)
    scale_mdi=xa.open_dataset(scale_mdi_f)
    un_tw=xa.open_dataset(un_tw_f)
    un_mdi=xa.open_dataset(un_mdi_f)
    
    # Preallocate Tw and mdi arrays -- shape=[len(x) | 4]
    # --> [tscen (since PI) | lower | mean | upper]
    ps_tw=np.zeros((len(x),4))*np.nan
    ps_mdi=np.zeros(ps_tw.shape)*np.nan
    pcount=0
    for scen in x: # x is vector of temp scenarios (relative to ref period); 
    # add dT to convert to T change since PI
        
        # Generate mean/lower/upper for tw
        proj_tw=ref_tw+scen*scale_tw["tw"][:,:].data
        proj_tw_lower=ref_tw+(scen*(scale_tw["tw"][:,:].data - \
                              un_tw["tw"][:,:].data))
        proj_tw_upper=ref_tw+(scen*(scale_tw["tw"][:,:].data + \
                              un_tw["tw"][:,:].data))
            
        # Repeat for mdi    
        proj_mdi=ref_mdi+scen*scale_mdi["mdi"][:,:].data
        proj_mdi_lower=ref_mdi+(scen*(scale_mdi["mdi"][:,:].data - \
                              un_mdi["mdi"][:,:].data))
        proj_mdi_upper=ref_mdi+(scen*(scale_mdi["mdi"][:,:].data + \
                              un_mdi["mdi"][:,:].data))  
            
        # Bundle into list for repetitive processing
        bundletw=[proj_tw_lower,proj_tw,proj_tw_upper]
        bundlemdi=[proj_mdi_lower,proj_mdi,proj_mdi_upper]
        ps_tw[pcount,0]=scen+dT; ps_mdi[pcount,0]=scen+dT
        for item_no in range(len(bundletw)):
            ps_tw[pcount,item_no+1]=\
                np.sum(cellarea[bundletw[item_no]>crit_tw])
            ps_mdi[pcount,item_no+1]=\
                np.sum(cellarea[bundlemdi[item_no]>crit_mdi])                
            
        pcount+=1
            
    # Write out
    bundleout=[out_tw,out_mdi,out_tas,tas_years[tas_idx],ps_tw,ps_mdi]
    for i in range(len(bundleout)):
        pickle.dump(bundleout[i],open(bundlenames[i],"wb"))

# ----------------------------------------------------------------------------
# Secondary processing/plotting
# ----------------------------------------------------------------------------   
if plot:
    # Read in the cell areas
    cellarea=xa.open_dataset(cell_f)
    cellarea=np.squeeze(cellarea["cell_area"][:,:].data)/1e6 # km**2
    mask=xa.open_dataset(mf)["d2m"][:,:].data
    ta=np.nansum(cellarea*mask)/10000. # Dividing by 10000 means fractions 
    # in %e+02
    
    # Check we have all files
    for i in bundlenames: 
        assert os.path.isfile(i),"Missing %s -- check / re-run"%i
        
    # Passed check, so read in and get to work
    mdi=pickle.load(open(fout_mdi,"rb"))
    mdi_ps=pickle.load(open(fout_ps_mdi,"rb"))
    tw=pickle.load(open(fout_tw,"rb"))
    tw_ps=pickle.load(open(fout_ps_tw,"rb"))
    gtas=pickle.load(open(fout_tas,"rb"))
       

x=np.linspace(1.5,5.5,100)# Interp points
tw_i=np.zeros((len(x),tw.shape[1]))*np.nan # Allocate tw
mdi_i=np.zeros(tw_i.shape)*np.nan # Allocate mdi

for i in range(tw.shape[1]):
    idx_in=np.logical_and(~np.isnan(gtas[:,i]),~np.isnan(tw[:,i]))
    idx_out=np.logical_and(x>=np.nanmin(gtas[:,i]),x<=np.nanmax(gtas[:,i]))
    
    tw_i[idx_out,i]=np.interp(x[idx_out],gtas[idx_in,i],tw[idx_in,i])
    mdi_i[idx_out,i]=np.interp(x[idx_out],gtas[idx_in,i],mdi[idx_in,i])

# Compute percentiles etc., for plotting
tw_lower=np.nanmin(tw_i,axis=1)
tw_med=np.nanmean(tw_i,axis=1)
tw_upper=np.nanmax(tw_i,axis=1)

mdi_lower=np.nanmin(mdi_i,axis=1)
mdi_med=np.nanmean(mdi_i,axis=1)
mdi_upper=np.nanmax(mdi_i,axis=1)

# Plot
fig,ax=plt.subplots(1,1)
ax.fill_between(x,mdi_lower/ta,mdi_upper/ta,color='red',alpha=0.7)
ax.plot(x,mdi_med/ta,color='red',label="MDI")
ax.fill_between(x,tw_lower/ta,tw_upper/ta,color='blue',alpha=0.7)
ax.plot(x,tw_med/ta,color='blue',label="Tw")
ax.grid()
ax.plot(mdi_ps[:,0],mdi_ps[:,1]/ta,color='orange',linestyle=":",linewidth=3)
ax.plot(mdi_ps[:,0],mdi_ps[:,2]/ta,color='orange',linewidth=3,label="MDI$_{s}$")
ax.plot(mdi_ps[:,0],mdi_ps[:,3]/ta,color='orange',linestyle=":",linewidth=3)
ax.plot(mdi_ps[:,0],tw_ps[:,1]/ta,color='purple',linestyle=":",linewidth=3)
ax.plot(mdi_ps[:,0],tw_ps[:,2]/ta,color='purple',linewidth=3,label="Tw$_{s}$")
ax.plot(mdi_ps[:,0],tw_ps[:,3]/ta,color='purple',linestyle=":",linewidth=3)
ax.set_ylim([0,np.max(mdi_ps[:,2])/ta*0.2])
ax.set_xlim([1.5,5.5])
ax.set_ylabel(r"Area above threshold (%$\times$ 100)")
ax.set_xlabel(r"Warming since pre-industrial ($^{\circ}$C)")
ax.legend()
fig.set_size_inches(6,5)
fig.savefig(figdir+"CMIP6_projections.png",dpi=300)