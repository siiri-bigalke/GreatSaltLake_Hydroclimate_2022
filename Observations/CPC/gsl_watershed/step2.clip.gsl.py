import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray
from shapely.geometry import Polygon
import cartopy.crs as ccrs
from shapely.geometry import mapping
import geopandas as gpd
import cartopy.feature as cfeature



# !!!!!! Activate climate2 environment !!!!!! 

save = '/work1/siiri/projects/CAS/Observations/CPC/gsl_watershed/binary_files/1950-2021/'
ppwd = '/work1/siiri/data/Observations/CPC/'


# ------------------------------------------------------------------
# Step 1) Load data and clip GSL shapefile from CPC gridded dataset
# ------------------------------------------------------------------

dryspell = np.load(save + 'annual.1950-2021.cpcgrid.dryspell.npy') # (72, 28, 28)
drydays = np.load(save + 'annual.1950-2021.cpcgrid.drydays.npy') # (72, 28, 28)
surplus = np.load(save + 'annual.1950-2021.cpcgrid.surplusint.npy') # (72, 28, 28)


# Load CPC data (1950-2021) to get original coordinate information
# ----------------------------------------------------------------
ds = xr.open_mfdataset(ppwd + 'precip.v1.0.1948-2021.nc').sel(
	               time = slice('1950', '2021'),
                       lon = slice(245, 252),
                       lat = slice(36, 43))['precip']

ds = ds.resample(time='AS').mean()

x = ds.coords['lon']
y = ds.coords['lat']
time = ds.coords['time']
lon = ds.coords['lon']
lat = ds.coords['lat']


# Load Great Salt Lake watershed shapefile information
# -----------------------------------------------------

gsl_shp = gpd.read_file('shapefile/GSLboundary_west_desert.shp')
crs0 = ccrs.AlbersEqualArea(central_longitude = 248)#.Mercator()
crs_proj4 = crs0.proj4_init
gslshape = gsl_shp.to_crs(crs_proj4)

'''
# -----------------------------------------
# Step 2) Clip CPC data with GSL shapefile
# -----------------------------------------

drydays = xr.DataArray(data = drydays,
                      dims = ['time', 'lat', 'lon'],
                      coords = dict(time = time,
                                   lat = lat,
                                   lon = lon),
                      attrs = dict(description = 'Annual sum of dry days',
                              units = 'dd'),
                      name = 'drydays')

dryspell = xr.DataArray(data = dryspell,
                        dims = ['time', 'lat', 'lon'],
                        coords = dict(time = time,
                                   lat = lat,
                                   lon = lon),
                        attrs = dict(description = 'Average annual length of consecutive'\
                                  'dry days', units = 'ds'),
                        name = 'dryspell') 

precip = xr.DataArray(data = surplus,
                      dims = ['time', 'lat', 'lon'],
                      coords = dict(time = time,
                                    lat = lat,
                                    lon = lon),
                      attrs = dict(description = 'Annual average daily precipitation'\
                              '(mm)', units = 'mm'),
                      name = 'surplus')


# Save to .nc format
# -------------------
drydays = drydays.to_netcdf(save + '1950-2021.cpc.drydays.nc')
dryspell = dryspell.to_netcdf(save + '1950-2021.cpc.dryspell.nc')
precip = precip.to_netcdf(save + '1950-2021.cpc.precip.nc')


exit()

# Real quick spatial plotting of metrics to check for errors ... !
# -----------------------------------------------------------------

states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

tcrs = ccrs.PlateCarree()
pcrs = ccrs.AlbersEqualArea(central_longitude=248)

fig, ax = plt.subplots(1, figsize = (4, 5), subplot_kw={'projection':pcrs})

cf = ax.contourf(x, y, precip[0,:,:], transform = tcrs)
ax.add_feature(states_provinces, edgecolor = 'gray')
ax.add_feature(cfeature.LAKES.with_scale('50m'),fc='darkslategray')

gslshape.plot(ax = ax, edgecolor = 'red', color = 'none')
cbar = fig.colorbar(cf)
plt.show(), exit()

'''

# -----------------------------------------
# Step 3) Clip CPC data with GSL shapefile
# -----------------------------------------

dd = xr.open_dataset(save + '1950-2021.cpc.drydays.nc')
ds = xr.open_dataset(save + '1950-2021.cpc.dryspell.nc')
pr = xr.open_dataset(save + '1950-2021.cpc.precip.nc')


def clip_gsl(var, vname):

    # Rewrite coordinates so they are compatible with GSL shapefile
    new = var.assign_coords(lon = ((var.lon + 180) % 360) - 180)
    print(new), exit()
    # Set spatial dimensions    
    d = new.rio.set_spatial_dims(x_dim = 'lon',
                                 y_dim = 'lat',
                                 inplace = True)
    
    dis = d.rio.write_crs(4326, inplace = True)

    # Clip GSL shapefile from CPC data
    clipped = dis.rio.clip(gslshape.geometry.apply(mapping),
                       gslshape.crs,
                       drop = False)
    clipped.to_netcdf(save + 'gsl.1950-2021.cpc.'+vname+'.nc')

    # Also save an GSL watershed averaged annual numpy array
    aavg = clipped.mean(['lat', 'lon']).to_numpy()
    np.save(save + 'annualts.gsl.1950-2021.cpc.'+vname+'.npy', aavg)

clip_gsl(dd.drydays, 'drydays')
clip_gsl(ds.dryspell, 'dryspell')
clip_gsl(pr.surplus, 'precip')


print('done!'), exit()
