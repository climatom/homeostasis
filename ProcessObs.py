#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Process the ERA5 land data. Note that we store: 
    
    1. The max annual MDI
    2. The Ta and Tw at the time of annual max MDI
    3. The annual max Tw.

"""
import pickle, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import utils
from netCDF4 import Dataset
import cartopy.crs as crs
import numba as nb
import cartopy.feature as cf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import h5py
from src import HeatBalance as hb
import seaborn as sns

met=62.5 #W/m**2
swrate_lower=(1240-620)/1000. # kg/hour
swrate_higher=(1240+620)/1000. 
AD=1.92 # surface area of average male (m**2): Foster et al. (2021)
va_lower=0.2 # m/s
va_higher=3.5 # m/s

@nb.jit(nopython=True)
def extract_closest(loclats,loclons,candlats,candlons,data):
    n=len(loclats)
    out=np.zeros(n)
    idx_log=np.zeros(n)
    idx_valid=~np.isnan(data)
    for i in range(n):        
        dists=utils.haversine_fast(loclats[i],loclons[i],candlats[idx_valid],
                                candlons[idx_valid],miles=False)
        idx=np.argmin(dists)
        out[i]=data[idx_valid][idx]
        idx_log[i]=idx
        assert ~np.isnan(out[i]),"Found a NaN!"
    return idx_log,out

@nb.jit('float64[:,:](int64,int64,float64[:,:],float64[:,:],float64,float64,float64)')
def gridHB(nr,nc,tagrid,rhgrid,met,va,swrate):
    swprod=swrate/AD
    # Calls heat balance; returns 1 if uncompensable; otherwise 0
    out=np.zeros((nr,nc))*np.nan
    for r in range(nr):
        for c in range(nc):
            if np.logical_and(~np.isnan(tagrid[r,c]),~np.isnan(rhgrid[r,c])):
                #HB(ta,rh,met,va,swprod)
               out[r,c]=hb.HB(tagrid[r,c],rhgrid[r,c],met,va,swprod)

    return out

@nb.jit('float64[:,:](int64,int64,float64[:,:],float64[:,:],float64,float64,float64)')
def findHottest(nr,nc,heat,lat,lon,temp,dew):
    pc=np.nanpercentile(heat,99.999)
    for r in range(nr):
        for c in range(nc):
            if heat[r,c]>pc:
                print("heat = %.2f (t=%.1f, dew=%.2f). Lat = %.2f, lon=%.2f"\
                      %(heat[r,c],temp[r,c],dew[r,c],lat[r,c],lon[r,c]))

    return 0

# Only set process to True if we need to compute the max files
process=False
yst=1981
ystp=2020
di="/home/lunet/gytm3/Homeostasis/ERA5Land/"
mdi_name="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_%.0f-%.0f.p"%\
    (yst,ystp)
cellname="/home/lunet/gytm3/Homeostasis/CMIP6/cellarea.nc"
cityname="/home/lunet/gytm3/Homeostasis/cities.csv"
figdir="/home/lunet/gytm3/Homeostasis/Figures/"
datadir="/home/lunet/gytm3/Homeostasis/Data/"
tw_name="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_TW_%.0f-%.0f.p"%\
    (yst,ystp)
# Note - max met var during max mdi
ta_name="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_T_%.0f-%.0f.p"%\
    (yst,ystp)
td_name="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_TD_%.0f-%.0f.p"%\
    (yst,ystp)
sp_name="/home/lunet/gytm3/Homeostasis/ERA5Land/ANN_MAX_MDI_SP_%.0f-%.0f.p"%\
    (yst,ystp)  
isdfile="/home/lunet/gytm3/Homeostasis/Data/finalarrays.mat"
meta=pd.read_csv("/home/lunet/gytm3/Homeostasis/Data/meta.csv",
                 names=["code","elev","lat","lon"])
regiondir="/home/lunet/gytm3/Homeostasis/ERA5Regions/"
regions=["Europe","Russia","Kuwait","Australia","PNW","Chicago"]
gmeanfile="/home/lunet/gytm3/Homeostasis/ERA5/Gmean.nc"
trmeanfile="/home/lunet/gytm3/Homeostasis/ERA5/Tropical_mean.nc"
nr=1801
nc=3600
crit_sed=28/0.74
crit_tw=35.

# For plotting...
prox_thresh=-10 # TW/MDI within this classed as 'close'

if process:
    
    nt=(ystp-yst)+1
    
    # Preallocate arrays as large as (ystp-yst+1)
    out_mdi=np.zeros((nt,nr,nc))*np.nan
    out_tw=np.zeros(out_mdi.shape)*np.nan
    out_ta=np.zeros(out_mdi.shape)*np.nan
    out_td=np.zeros(out_mdi.shape)*np.nan
    out_sp=np.zeros(out_mdi.shape)*np.nan
    # Tw..
    out_ta_tw=np.zeros(out_mdi.shape)*np.nan
    out_td_tw=np.zeros(out_mdi.shape)*np.nan
    out_sp_tw=np.zeros(out_mdi.shape)*np.nan
    
    ycount=0
    for y in range(yst,ystp+1):
        
        # Find year files
        fs_mdi=[ii for ii in os.listdir(di) if ii[:4] == "%.0f"%y and "_mdi.p" in ii]
        
        # Iterate over each of the year files 
        fcount=0
        for f in fs_mdi:
            # Get time max
            mdi=pickle.load(open(di+f, "rb" ))
            ta=pickle.load(open(di+f.replace("mdi","ta"), "rb" ))
            td=pickle.load(open(di+f.replace("mdi","td"), "rb" ))
            tw=pickle.load(open(di+f.replace("mdi","tw"), "rb" ))
            sp=pickle.load(open(di+f.replace("mdi","sp"), "rb" ))
            
            # Repeat for time of TW max (e.g., 20210526_ta_tw.p from 20210601_mdi.p) 
            ta_tw=pickle.load(open(di+f.replace("mdi","ta_tw"), "rb" ))
            td_tw=pickle.load(open(di+f.replace("mdi","td_tw"), "rb" ))
            sp_tw=pickle.load(open(di+f.replace("mdi","sp_tw"), "rb" ))           
            
            if fcount==0:
                out_mdi[ycount,:,:]=mdi[:,:]
                out_ta[ycount,:,:]=ta[:,:]
                out_td[ycount,:,:]=td[:,:]
                out_tw[ycount,:,:]=tw[:,:]
                out_sp[ycount,:,:]=sp[:,:]
                # tw
                out_ta_tw[ycount,:,:]=ta_tw[:,:]
                out_td_tw[ycount,:,:]=td_tw[:,:]                
                out_sp_tw[ycount,:,:]=sp_tw[:,:]   
                
            else:
                idx=mdi>out_mdi[ycount,:,:]
                out_mdi[ycount,idx]=mdi[idx]
                out_ta[ycount,idx]=ta[idx]
                out_td[ycount,idx]=td[idx]
                out_sp[ycount,idx]=sp[idx]
                
                # Also store max tw
                idx_tw=tw>out_tw[ycount,:,:]
                out_tw[ycount,idx_tw]=tw[idx_tw]
                out_ta_tw[ycount,idx_tw]=ta_tw[idx_tw]
                out_td_tw[ycount,idx_tw]=td_tw[idx_tw]
                out_sp_tw[ycount,idx_tw]=sp_tw[idx_tw]
                assert np.nanmax(out_tw) < (273.15+35.),\
                "Found a hot one here - check! [file = %s]"%f
                
            fcount+=1
            print("Processed one file!")
            
            
        ycount+=1
        print("\n\nPROCESSED YEAR %.0f!\n\n"%y)
    
    ## Write out
    pickle.dump(out_mdi,open(mdi_name, "wb" ) )
    pickle.dump(out_tw,open(mdi_name.replace("MDI","TW"), "wb" ) )
    pickle.dump(out_ta,open(mdi_name.replace("MDI","MDI_T"), "wb" ) )
    pickle.dump(out_td,open(mdi_name.replace("MDI","MDI_TD"), "wb" ) )
    pickle.dump(out_sp,open(mdi_name.replace("MDI","MDI_SP"), "wb" ) )
    # tw
    pickle.dump(out_ta_tw,open(mdi_name.replace("MDI","TW_T"), "wb" ) )
    pickle.dump(out_td_tw,open(mdi_name.replace("MDI","TW_TD"), "wb" ) )
    pickle.dump(out_sp_tw,open(mdi_name.replace("MDI","TW_SP"), "wb" ) )

# Deliberate break point (comment out if desired)
# assert 1==2
    
# Make sure MDI name refers to the full dataset -- not early/late chunks. We
# will read those early/late chunks in seperately. 
mdi_name=mdi_name.replace("_%.0f-%.0f"%(yst,ystp),"")
assert os.path.isfile(mdi_name), "\nDon't have file:\n %s\n\n Fix it!"%mdi_name
mdi=pickle.load(open(mdi_name, "rb" ))
# mdi-=(273.15*0.75+273.15*0.3) # Correct to C (values inflated from K step)

# Read in cell weights
ncfile=Dataset(cellname)
lats=ncfile.variables["latitude"][:].data
lons=ncfile.variables["longitude"][:].data  
lon2,lat2=np.meshgrid(lons,lats)  
wts=ncfile.variables["cell_area"][:,:].data/1e6 # sq km
   
# Read in top TW, plus Ta, Td, and sp during the top MDI
tw=pickle.load(open(tw_name, "rb" ))
ta=pickle.load(open(ta_name, "rb" ))
td=pickle.load(open(td_name, "rb" ))
sp=pickle.load(open(sp_name, "rb" ))
ta[ta<-100]=np.nan; td[td<-100]=np.nan;
scratch=mdi.copy()
scratch[np.isnan(scratch)]=-100
idx=np.nanargmax(scratch,axis=0)
scratch=tw.copy()
scratch[np.isnan(scratch)]=-100
idx_tw=np.nanargmax(scratch,axis=0)
mdi_max=np.squeeze(np.take_along_axis(mdi,idx[None,:,:],axis=0))
tw_max=np.squeeze(np.take_along_axis(tw,idx_tw[None,:,:],axis=0))-273.15
ta_max=np.squeeze(np.take_along_axis(ta,idx[None,:,:],axis=0))-273.15
td_max=np.squeeze(np.take_along_axis(td,idx[None,:,:],axis=0))-273.15
sp_max=np.squeeze(np.take_along_axis(sp,idx[None,:,:],axis=0))-273.15
del ta, td, sp # Make space

_=findHottest(nr,nc,mdi_max,lat2,lon2,ta_max,td_max)


# Repeat for Ta, etc. during top Tw
ta_tw=pickle.load(open(ta_name.replace("MDI","TW"), "rb" ))
td_tw=pickle.load(open(td_name.replace("MDI","TW"), "rb" ))
sp_tw=pickle.load(open(sp_name.replace("MDI","TW"), "rb" ))
ta_tw[ta_tw<-100]=np.nan; td_tw[td_tw<-100]=np.nan; sp_tw[sp_tw<-100]=np.nan
ta_tw_max=np.squeeze(np.take_along_axis(ta_tw,idx_tw[None,:,:],axis=0))-273.15
td_tw_max=np.squeeze(np.take_along_axis(td_tw,idx_tw[None,:,:],axis=0))-273.15
sp_tw_max=np.squeeze(np.take_along_axis(sp_tw,idx_tw[None,:,:],axis=0))-273.15
del ta_tw, td_tw, sp_tw # Again, make space. 

# 
## Here we can evaluate the ME for maximum Tw and MDI
me_max=utils._ME2D(nr,nc,ta_max+273.15,td_max+273.15,sp_max)
me_max_tw=utils._ME2D(nr,nc,ta_tw_max+273.15,td_tw_max+273.15,sp_tw_max)

# Compute the relative humidity (during max) from ta and td
nr,nc=ta_max.shape
rh=utils._satVp2D(td_max+273.15,nr,nc)/utils._satVp2D(ta_max+273.15,nr,nc)*100.; 
rh[rh>100]=100

# * * * 
# HadISD work 
# * * * 
# arrays={}
# f=h5py.File(isdfile)
# for k,v in f.items():
#     arrays[k]=np.array(np.squeeze(v[2,:,:]))
# years=np.squeeze(v[0,:,:])

# # Compute the (modified) discomfort index (see Kenney et al)
# arrays["mdi"]=0.75*arrays["twarray"]+0.30*arrays["tarray"]

# # Extract t and td for the days on which mdi reaches its peak
# nloc=arrays["mdi"].shape[1]
# tsel=np.zeros(nloc)
# tdsel=np.zeros(tsel.shape)
# pi=np.zeros(nloc)
# for i in range(nloc):
#     idx=arrays["mdi"][:,i]==np.nanmax(arrays["mdi"][:,i])
#     tsel[i]=\
#     arrays["tarray"][idx,i][0]
#     tdsel[i]=\
#     arrays["tdarray"][idx,i][0]
# isd_mdi_max=np.nanmax(arrays["mdi"],axis=0)
# max_above=np.nanmax(isd_mdi_max)-crit_sed
# max_above =2 # Hard code
# isd_idx=isd_mdi_max>(crit_sed-max_above)
# isd_lon=meta["lon"].loc[isd_idx]
# isd_lat=meta["lat"].loc[isd_idx]
# isd_mdi_max=isd_mdi_max[isd_idx]

# # Read in three-hourly hadISD
# had3=pd.read_csv("/home/lunet/gytm3/Homeostasis/Data/discomfindex.txt",
#                   names=["lat","lon","z","mdi","t","td"])
# del arrays

# # #  END HadISD

# * * * 
# Region work 
# * * * 
regions_plot={}
for r in regions:
    d=Dataset(regiondir+r+".nc")
    rta=np.squeeze(d["t2m"][:,:,:].data)
    rtd=np.squeeze(d["d2m"][:,:,:].data)
    rsp=np.squeeze(d["sp"][:,:,:].data)
    nt,nr,nc=rta.shape
    rtw=utils._TW3D(nt,nr,nc,rta,rtd,rsp)-273.15
    rmdi=0.75*rtw+0.3*(rta-273.15)
    idx=np.nanargmax(rmdi)
    
    regions_plot[r]=[np.nanmax(rmdi),rta.flatten()[idx]-273.15,
            utils._satVP(rtd.flatten()[idx])/utils._satVP(rta.flatten()[idx])*100.,
                     rsp.flatten()[idx]]

del d, rta, rtd, rsp, rtw, rmdi
# End region work


# # * * * 
# # Future tropical extremes work
# # * * * 
# gmean=Dataset(gmeanfile)
# trmean=Dataset(trmeanfile)
# gmean=pd.Series(np.squeeze(gmean.variables["t2m"][:].data)).rolling(30).mean()
# trmean=pd.Series(np.squeeze(trmean.variables["t2m"][:].data)).rolling(30).mean()
# gmeanidx=np.logical_and(~np.isnan(gmean),~np.isnan(trmean))
# trsens,intercept=np.polyfit(gmean.values[gmeanidx],trmean.values[gmeanidx],1)
# # trsens=1.4/1.5
# print("\n...Tropical sensitivity is %.2f"%trsens)

# # Now we use trsens to scale Tw max. Note that we read in lat/lon here. Need
# # to do this so that we can extract the tropical latitudes.
# ncfile=Dataset(cellname)
# lats=ncfile.variables["latitude"][:].data
# lons=ncfile.variables["longitude"][:].data  
# lon2,lat2=np.meshgrid(lons,lats)  
# wts=ncfile.variables["cell_area"][:,:].data/1e6 # sq km
# tropics_idx=np.abs(lats)<=20
# tropical_wts=wts[tropics_idx]
# total_area=np.sum(tropical_wts[~np.isnan(tw_max[tropics_idx])])
# scens= np.arange(0.0,10,0.1) # Amounts of global warming
# area_above=np.zeros((len(scens),4))
# count=0
# for scen in scens:
#     tw_max_scale=tw_max[tropics_idx,:]+scen*trsens # *** Scaled Tw_max
#     _nr,_nc=tw_max_scale.shape
    
#     # Now we iteratively solve for Td, given the new Tw and the current RH
#     sp_dummy=np.ones(tw_max_scale.shape)*101300 # Will need to be updated w/act. sp
#     ta_max_scale=\
#         utils._TW_TD_2D(_nr,_nc,tw_max_scale+273.15,rh[tropics_idx],sp_dummy)-273.15
#     mdi_max_scale=tw_max_scale*0.75+ta_max_scale*0.3
    
#     # Fill in area exceeding crit TW
#     area_above[count,0]=np.sum(tropical_wts[tw_max_scale>=35])/total_area*100
    
#     # Repeat for crit MDI
#     area_above[count,1]=\
#         np.sum(tropical_wts[mdi_max_scale>=crit_sed])/total_area*100
    
#     # Compute the heat balance using new t and old rh
#     # With low sweat rate and no fan
#     shigh=gridHB(_nr,_nc,ta_max_scale,rh[tropics_idx],met,va_lower,swrate_lower)
#     area_above[count,2]=np.sum(tropical_wts[shigh>0])/total_area*100
    
#     # High sweat rate and fan
#     slow=gridHB(_nr,_nc,ta_max_scale,rh[tropics_idx],met,va_higher,swrate_higher) 
#     area_above[count,3]=np.sum(tropical_wts[slow>0])/total_area*100
    
#     # Increment counter 
#     count+=1

# # Compute ToE with a simple interpolation (ToE defined as 1% of tropical land
# # area crossing the threshold)
# toe_tw=np.interp(1,area_above[:,0],scens+1)# +1 is for warming since PI
# toe_mdi=np.interp(1,area_above[:,1],scens+1)

# # # End future tropical extremes work


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# Plotting
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 


# * * * 
# Plot future exceedances
# * * *
# fig,ax=plt.subplots(1,1)
# scens_plot=scens+1 # To account for warming since PI
# ax.plot(scens_plot,area_above[:,0],color='red')
# ax.plot(scens_plot,area_above[:,1],color='blue')
# ax.fill_between(scens_plot,area_above[:,3],area_above[:,2],color='black',
#                 alpha=0.2)
# ax.plot(scens_plot,area_above[:,0],color='blue')
# ax.plot(scens_plot,area_above[:,1],color='red')
# ax.set_xlim(min(scens_plot),max(scens_plot))
# ax.axvline(toe_tw,color="blue",linestyle='--')
# ax.axvline(toe_mdi,color="red",linestyle='--')
# ax.set_ylim(0,100)
# ax.grid()
# ax.set_xlabel("Global warming since Pre-Industrial ($^{\circ}$C)")
# ax.set_ylabel("Uncompensable tropical area (%)")
# fig.savefig(figdir+"Future_Exceedances.png",dpi=300)


# * * * 
# 2d Hist with regions 
# * * * 
tmin=25
tmax=55
rhmin=1
rhmax=100
nx=50
ny=50
# ta_vector=np.linspace(20,55,nx)
# rh_vector=np.linspace(1,100,ny)
h,xedge,yedge=np.histogram2d(ta_max.flatten(),rh.flatten(),bins=[nx,ny],\
                             range=[[tmin,tmax],[rhmin,rhmax]],
                             weights=wts.flatten()/1e6,
                             normed=False)   
rh_ref=np.arange(5,91,1)
tbord_low=np.zeros(len(rh_ref))
tbord_high=np.zeros(len(rh_ref))
for kk in range(len(rh_ref)):
    rhidx=np.logical_and(rh>=rh_ref[kk]-2.5,rh<rh_ref[kk]+2.5)
    tbord_low[kk]=np.min(ta_max[rhidx])
    tbord_high[kk]=np.max(ta_max[rhidx])
    
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1, 1, 1)
h[h==0]=np.nan
h[h<np.nanpercentile(h,5)]=np.nan
xi=1/2.*(xedge[:-1]+xedge[1:])
yi=1/2.*(yedge[:-1]+yedge[1:])
xi,yi=np.meshgrid(xi,yi)
# Generate reference Tw (for 1000 hPa=pref)
pref=np.ones((nx,ny))*101300.
tw_z=utils._TW2D(nx,ny,xi+273.15,yi,pref)-273.15
mdi_z=0.75*tw_z+xi*0.3
levs=np.linspace(np.nanpercentile(h,15),np.nanpercentile(h,99.99),20)
im=ax.contourf(xi,yi,h,cmap="Reds",levels=levs)
ax.contour(xi,yi,mdi_z,levels=[crit_sed,],colors=["k",],linewidths=[3,])
ax.contour(xi,yi,tw_z,levels=[35.,],colors=["grey",],linewidths=[3,])
# ax.contour(xi,yi,mdi_z,levels=[mdi_999_early,],colors=["purple",],
#            linewidths=[3,],linestyles=["-"])
# ax.contour(xi,yi,mdi_z,levels=[mdi_999_late,],colors=["purple",],
#            linewidths=[3,],linestyles=["--"])
ax.plot(tbord_high,rh_ref,color='red',linestyle="--")
ax.set_ylabel("Relative humidity (%)")
ax.set_xlabel("Air temperature ($^{\circ}$C)")
ax.set_xlim(25,54)
ax.grid()

# Iterate over regions and plot them
jigglex=5
jiggley=10
for r in regions:
    if r=="PNW": xfactor=5
    if r=="Russia": xfactor=-20; yfactor=20
    elif r=="Chicago": yfactor=10; xfactor=-2
    else: xfactor=0; yfactor=0
    ax.scatter(regions_plot[r][1],regions_plot[r][2],color='k')
    ax.annotate(text=r,xy=(regions_plot[r][1],regions_plot[r][2]),
                xytext=(regions_plot[r][1]-jigglex*2+xfactor,
                        regions_plot[r][2]-jiggley*2+yfactor),
                textcoords='data',
                arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),
                xycoords='data')
    jigglex*=-1; jiggley*=-1

cax = fig.add_axes([0.78, 0.12, 0.02, 0.76])  
plt.subplots_adjust(right=0.77)
cb=fig.colorbar(im, cax=cax, orientation='vertical')
cb.set_ticks([0.1,0.2,0.3,0.4,0.5])
cb.set_label(r"Area (km $\times$ 10$^{-6}$)")
fig.savefig(figdir+"ERA5LandHistRegionspng",dpi=300)
assert 1==2  
# # * * * 
# # ISD
# # * * * 
# ax = fig.add_subplot(2, 1, 2, projection=crs.Robinson())
# cscat=ax.scatter(x=isd_lon,y=isd_lat,transform=crs.PlateCarree(),
#            c=isd_mdi_max,cmap="seismic",vmin=crit_sed-max_above,
#            vmax=crit_sed+max_above,alpha=0.5,s=10)
# ax.set_global()
# ax.coastlines(linewidth=0.2)
# ax.gridlines(draw_labels=False)
# plt.subplots_adjust(bottom=0.2)
# cax=fig.add_axes([0.32,0.15,0.39,0.02])
# cb=plt.colorbar(cscat,orientation='horizontal',cax=cax)
# plt.subplots_adjust(hspace=0.5)
# cb.set_label("MDI ($^{\circ}$C)")
# cb.set_ticks([37,38,39])
# fig.savefig(figdir+"ERA5LandHistMap.png",dpi=300)
# # assert 1==2

# * * * 
# Cartopy maps
# * * *
nlevs=25
lower_p=75
upper_p=99
zon_roll=5 # degrees
delta_tw=tw_max-crit_tw; delta_mdi=mdi_max-crit_sed
zon_tw=np.nanpercentile(delta_tw,99,axis=1)
zon_mdi=np.nanpercentile(delta_mdi,99,axis=1)
zon_delta=pd.Series(zon_mdi-zon_tw); zon_delta_roll=\
    zon_delta.rolling(10*zon_roll,center=True).mean()
zon_tw=pd.Series(zon_tw); zon_tw_roll=zon_tw.rolling(10*zon_roll,center=True).mean()
zon_mdi=pd.Series(zon_mdi); zon_mdi_roll=zon_mdi.rolling(10*zon_roll,center=True).mean()
xticks=[-60, -30, 0,30,60]
# levs=np.linspace(1,np.nanpercentile(delta_tw,99.99),nlevs)

###### TW 
z=tw_max.copy()
z[z<25]=np.nan
# delta_tw[delta_tw<prox_thresh]=np.nan
# vmin=np.nanmin(delta_tw); print(vmin)
# vmax=np.nanmax(delta_tw); print(vmax)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(2, 1, 1, projection=crs.Robinson())
ax.set_global()
ax.coastlines(linewidth=0.2)
p=ax.pcolormesh(lons,lats,z,cmap="turbo", transform=crs.PlateCarree(),
            vmin=25,
            vmax=37)
ax.gridlines(draw_labels=False)
ax.add_feature(cf.BORDERS,linewidth=0.15)
# ax.scatter(topn_lon_tw,topn_lat_tw,transform=crs.PlateCarree(),color='black',
#            s=100,alpha=0.5)
# cb=plt.colorbar(p,shrink=0.5)

###### MDI 
z=mdi_max.copy()
z[z<25]=np.nan
# delta_mdi[delta_mdi<prox_thresh]=np.nan
# vmin=np.nanmin(delta_mdi);print(vmin)
# vmax=np.nanmax(delta_mdi); print(vmax)
ax = fig.add_subplot(2, 1, 2, projection=crs.Robinson())
ax.set_global()
ax.coastlines(linewidth=0.2)
p=ax.pcolormesh(lons,lats,z,cmap="turbo", transform=crs.PlateCarree(),
            vmin=25,
            vmax=37)
# ax.scatter(topn_lon,topn_lat,transform=crs.PlateCarree(),color='black',
#            s=100,alpha=0.5)
ax.add_feature(cf.BORDERS,linewidth=0.15)
ax.gridlines(draw_labels=False)
plt.subplots_adjust(bottom=0.2)
cax=fig.add_axes([0.36,0.15,0.31,0.02])
cb=plt.colorbar(p,orientation='horizontal',cax=cax)
cb.set_ticks([25,28,31,34,37])
cax.set_xlabel("$^{\circ}$C")
fig.savefig(figdir+"ObsMax.png",dpi=300)


# Difference in proximity
z=delta_mdi-delta_tw
z[(mdi_max-crit_sed)<prox_thresh]=np.nan
vmin=np.nanmin(z); print(vmin)
vmax=np.nanmax(z); print(vmax)
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(1, 1, 1, projection=crs.Robinson())
ax.set_global()
p=ax.pcolormesh(lons,lats,z,cmap="turbo", transform=crs.PlateCarree(),
              vmin=vmin,vmax=vmax)
ax.coastlines(linewidth=0.2)
ax.add_feature(cf.BORDERS,linewidth=0.15)
ax.gridlines(draw_labels=False)
ax.add_feature(cf.BORDERS,linewidth=0.15)
plt.subplots_adjust(bottom=0.2,left=0.22,right=0.78)
zon_ax=fig.add_axes([0.1,0.335,0.1,0.40])
zon_ax.plot(zon_tw_roll,lats,color='blue',label="TW")
zon_ax.plot(zon_mdi_roll,lats,color='red',label="MDI")
cax=fig.add_axes([0.35,0.27,0.3,0.02])
cb=plt.colorbar(p,orientation='horizontal',cax=cax)
cax.set_xlabel("$^{\circ}$C")
zon_ax2=fig.add_axes([0.8,0.335,0.1,0.40])
zon_ax2.plot(zon_delta_roll,lats,color='black',label="MDI-TW")
zon_ax.set_xlim(-10,-2)
zon_ax2.set_xlim(-1,2)
zon_ax.set_xlabel("$^{\circ}$C from threshold")
zon_ax2.set_xlabel("MDI-TW ($^{\circ}$C)")
zon_ax2.set_yticklabels([""])
zon_ax.legend(loc=4,fontsize=6)
fig.savefig(figdir+"ObsDelta.png",dpi=300)

# Scatter me -- if desired. 
# me_idx=np.logical_and(me_max_tw>np.nanpercentile(me_max,75),~np.isnan(me_max_tw))
# fig,ax=plt.subplots(1,1)
# sns.kdeplot(x=me_max_tw[me_idx]/1000.,
#             y=me_max[me_idx]/1000.,alpha=0.1,cmap="Reds",shape=True)
# refl=np.linspace(np.min(me_max)/1000.-1000,np.max(me_max)/1000.+1000,1000)
# ax.plot(refl,refl)
# ax.grid()
# ax.set_xlabel("J/g [Tw]")
# ax.set_ylabel("J/g [MDI]")
# fig.savefig(figdir+"ME_scatter.png",dpi=300)
# Where are the super hot places within 20 deg of the equator?
# subset_idx=np.abs(lat2)<=15
# maxt=np.nanmax(ta_max[subset_idx])
# inds=np.nanargmax(ta_max[subset_idx])
# lonq=lon2[subset_idx].flatten()[inds]; latq=lat2[subset_idx].flatten()[inds]

