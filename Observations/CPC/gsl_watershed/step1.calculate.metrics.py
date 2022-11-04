import numpy as np
import xarray as xr
import matplotlib.pyplot as plt



# !!!!!! Activate climate2 environment !!!!!! 

save = '/work1/siiri/projects/CAS/Observations/CPC/gsl_watershed/binary_files/1950-2021/'
ppwd = '/work1/siiri/data/Observations/CPC/'


# ------------------------------------------------------------------
# Step 1) Load data and clip GSL shapefile from CPC gridded dataset
# ------------------------------------------------------------------

# Load CPC data (1950-2021)
# ---------------------------
ds = xr.open_mfdataset(ppwd + 'precip.v1.0.1948-2021.nc').sel(
	               time = slice('1950', '2021'),
                       lon = slice(245, 252),
                       lat = slice(36, 43))['precip']
print(ds), exit()
#x = ds.coords['lon']
#y = ds.coords['lat']
time = ds.coords['time']
lon = ds.coords['lon']
lat = ds.coords['lat']

print(type(ds))
precip = ds.to_numpy()
print('done converting to numpy'), exit()

#np.save(save + 'cpc.dailypreciprate.npy', precip)

# Find annual sum of daily precipitation
aprecip = ds.resample(time = 'AS').mean()
aprecip = aprecip.to_numpy()
print('done converting to numpy')

np.save(save + 'annual.1950-2021.cpcgrid.surplusint.npy', aprecip)
exit()

'''
# ----------------------------------
# Step 2) Calculate 'deficit days'
# ----------------------------------

# Identifying defifict days in a numpy array because of computational efficiency

precip = np.load(save + 'cpc.dailypreciprate.npy') # (26298, 28, 28)

time = len(precip[:,0,0])
lat = len(precip[0,:,0])
lon = len(precip[0,0,:])

def_day = np.zeros((time, lat, lon))
#dryspell = np.zeros((time, lat, lon)) 
#drydays = np.zeros((time, lat, lon))


for latx in range(lat):
    for lonx in range(lon):
        for day in range(time):
            print('lat, day=', latx , day)
            p = precip[day, latx, lonx]
            if p  < 0.25:
                def_day[day, latx, lonx] = 1
            else:
                def_day[day, latx, lonx] = False
    
#        d = def_day[year, :, lat, lon]
#        drydays[year, lat, lon] = np.nansum(d)
#        dTS = np.diff(np.where(np.concatenate(([d[0]],
#                                d[:-1] != d[1:],
#                                [True])))[0])[::2]
        
#        dryspell[year, lat, lon] = dTS.mean()


#np.save(save + 'gsl.1950-2021.cpcgrid.defdays.npy', def_day)
'''

# -----------------------------------------
# Step 3) Calculate dry days and dry spell
# -----------------------------------------

defdays = np.load(save + 'gsl.1950-2021.cpcgrid.defdays.npy')

# Convert numpy array back into xarray
ddays = xr.DataArray(data = defdays,
                     coords = dict(time = time,
                                   lat = lat,
                                   lon = lon),
                     attrs = dict(description = 'deficit days',
                                  units = 'dd'),
                     name = 'deficit days')


# Calculate average annual length of dry spell
dryspell = np.zeros((72, 28, 28))
years = np.arange(1950, 2022)

for latx in range(len(lat)):
    for lonx in range(len(lon)):
        for y, year in enumerate(years):
            print('lat, year = ', latx, year)
            d = ddays.sel(time = str(year)).isel(lat = latx, lon = lonx)
            d = np.array(d)
            #print(d), exit()
            annual = np.diff(np.where(np.concatenate(([d[0]],
                                d[:-1] != d[1:],
                                [True])))[0])[::2]
            dryspell[y, latx, lonx] = np.nanmean(annual)#.nanmean() # or .sum()
        #print(dryspell[:, 0, 0]), exit()

np.save(save + 'annual.1950-2021.cpcgrid.dryspell.npy', dryspell)

#drydays = ddays.resample(time = 'AS').sum() # annual sum of drydays
#np.save(save + 'annual.1950-2021.cpcgrid.drydays.npy', drydays)

