import cdsapi,os, numpy as np
from calendar import monthrange
do="/home/lunet/gytm3/Homeostasis/ERA5Land/"
exe="/home/lunet/gytm3/Homeostasis/Code/processERA5Land.py"
c = cdsapi.Client()
chunksize=5 # days
for y in range(1981,2022):
    for m in range(1,13):
        nd=monthrange(y,m)[1]
        # Break into multi-day chunks
        days = np.array_split(np.arange(1,nd+1),chunksize)
        for block in days:
            d=["%02d"%i for i in block]
            oname=do+"%.0f%02d%s.nc"%(y,m,d[0])
            if os.path.isfile(oname.replace(".nc","_sp_tw.p")):continue
            # if os.path.isfile(oname):continue
            fail=999.
            while fail !=0:
                c.retrieve(
                        'reanalysis-era5-land',
                        {
                        'format': 'netcdf',
                        'variable': [
                            '2m_dewpoint_temperature', '2m_temperature', 'surface_pressure',
                        ],
                        'year': '%.0f'%y,
                           'month':'%02d'%m, 
                        'day': d,
                        'time': [
                            '00:00',# '01:00', '02:00',
                            '03:00',# '04:00', '05:00',
                            '06:00',# '07:00', '08:00',
                            '09:00',# '10:00', '11:00',
                            '12:00',# '13:00', '14:00',
                            '15:00',# '16:00', '17:00',
                            '18:00',# '19:00', '20:00',
                            '21:00',# '22:00', '23:00',
                        ],
                        },
                    oname)
                fail=os.system("python %s %s"%(exe,oname))
                if fail != 0: print("\n\n*FAILED -- trying again...\n\n")
            os.remove(oname)
