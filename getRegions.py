#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script gets ERA5[Land] data for specific regions and times
"""
import cdsapi, os
do="/home/lunet/gytm3/Homeostasis/ERA5Regions/"
#dataset='reanalysis-era5-single-levels'
dataset='reanalysis-era5-land'
freq=1# hourly resolution
regions={
         "Europe":{"extent":['60','-15','42','10',],
                   "month":['07','08',],
                   "year":['2003',]},
         "Russia":{"extent":['60','25','52','55',],
                   "month":['07','08',],
                   "year":['2010',]},
         "Kuwait":{"extent":['32','45','28','51',],
                   "month":['07',],
                   "year":['2016',]},        
         "Australia":{"extent":['-26','129','-38','141',],
                   "month":['01','12',],
                   "year":['2019','2020',]},         
         "PNW":{"extent": ['60','-125','42','-110',],
                "month":["06","07"],
                "year" :["2021",]},       
         "Chicago":{"extent": ['45','-95','40','-80',],
                "month":["07",],
                "year" :["1995",]}      
         } 


c = cdsapi.Client()
for r in regions.keys():
    oname=do+r+".nc"
    if os.path.isfile(oname): continue
    print("\nRetrieving data for region: %s..."%r)
    fail=-999
    while fail !=0:
        c.retrieve(dataset,
            {
              'format': 'netcdf',
              'product_type': 'reanalysis',
              'variable': ['2m_dewpoint_temperature', 
              '2m_temperature', 'surface_pressure',],
                'year': regions[r]["year"][:],
                'month': regions[r]["month"][:], 
                # 'day': ['1',],
                # 'time':['00:00',],
                'day': ["%.02d"%i for i in range(1,32)][:],
                'time': ["%02d:00"%i for i in range(0,24,freq)][:],
                'area': regions[r]["extent"][:],
              },                                  
             oname)
        fail=0
