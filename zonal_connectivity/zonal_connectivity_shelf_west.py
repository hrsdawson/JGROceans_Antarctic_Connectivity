#!/usr/bin/env python
# coding: utf-8

# # - Script for PROJECT 01: Pathways and Timescales of Connectivity at the Antarctic Margin - ##
# # - Calculates zonal connectivity along shelf - ##

# Importing the relevant modules. 
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
import netCDF4 as nc
import time

######## get run count argument that was passed to python script:
import sys
i = int(sys.argv[1])
region = sys.argv[2]
print(i, region)


## ---------------------------------------------------------------------
## Define particle data files
#npart = 130146
datadir = '/g/data/e14/hd4873/runs/parcels/output/AntConn/data/xarray_files/traj_chunked_basin/'
files = sorted(glob(datadir+'CircumAntarcticParticles_*.nc')) 

# read in shelf mask: Shelf = 1, Southern Ocean = 0
mask = xr.open_dataset('/g/data/e14/hd4873/runs/parcels/output/AntConn/data/basin_masks/zonal_connectivity_mask_ht.nc')

# read in starting values
startfile = '/g/data/e14/hd4873/runs/parcels/output/AntConn/data/CircumAntarcticParticles_initial_values.nc'
ds_iv = xr.open_dataset(startfile)

# Define preprocess function to drop unneccesary variables. 
def preprocess(ds):
    return ds.drop_vars(['psal','thermo','mixedlayershuffle','shelf','mldepth', 
                         'unbeachCount', 'z', 'basin',  
                         'lon', 'lat',
                        ])
# load dataset
print("Loading basin_ZonalConn particle dataset")
ds = xr.open_mfdataset(files, preprocess=preprocess, parallel=True)#.sel(trajectory=slice(0,100))#.load()
print(ds.nbytes/1000**3)
ds = ds.load()
print("Dataset loaded, calculating basindiff")

# calculate difference between basin values along particle trajectories and convert to xarray DataArray
basindiff = np.copy(ds.basin_ZonalConn)
basindiff[:,:] = 0
basindiff[:-1,:] = ds.basin_ZonalConn[:-1,:].values - ds.basin_ZonalConn[1:,:].values
basindiff = xr.DataArray(basindiff, coords=[ds.time, ds.trajectory], dims=["time", "trajectory"])
# add basindiff to dataset
ds["basindiff"] = basindiff

# Define preprocess function to drop all but lat and lon.
def preprocesslatlon(ds):
    return ds.drop_vars(['psal','thermo','mixedlayershuffle','shelf','mldepth', 
                         'unbeachCount', 'z', 'basin',  
                         #'lon', 'lat',
                         'basin_ZonalConn'
                        ])
# read in dataset (don't load at this stage)
print("Reading in lat-lon particle dataset")
dslatlon = xr.open_mfdataset(files, preprocess=preprocesslatlon, parallel=True)
print(dslatlon.nbytes/1000**3)


## -------------------------------------------------------------------------------
## Set up arrays and reference variables for analysis
print("Setting up arrays and variables")
source_regions = ['shelf_east', 'asc_east', 'asc_north', 'asc_west', 'shelf_west', 'shelf_asc']
connectivity_count = np.zeros((21,6))
connectivity_count = xr.DataArray(connectivity_count, coords=[mask.region[0:21], source_regions], dims=["shelf_basin", "source_region"])
connectivity_trans = np.zeros((21,6))
connectivity_trans = xr.DataArray(connectivity_trans, coords=[mask.region[0:21], source_regions], dims=["shelf_basin", "source_region"])
connectivity_count_pct = np.zeros((21,6))
connectivity_count_pct = xr.DataArray(connectivity_count_pct, coords=[mask.region[0:21], source_regions], dims=["shelf_basin", "source_region"])
connectivity_trans_pct = np.zeros((21,6))
connectivity_trans_pct = xr.DataArray(connectivity_trans_pct, coords=[mask.region[0:21], source_regions], dims=["shelf_basin", "source_region"])
# Combined into one Dataset
connectivity_count = xr.Dataset({"conn_count": connectivity_count, 
                                 "conn_trans": connectivity_trans, 
                                 "conn_count_pct": connectivity_count_pct, 
                                 "conn_trans_pct": connectivity_trans_pct, 
                                })

# Define difference values for each basin based on direction of travel
shelf_east_diffs = np.arange(5, 47, 2.0)
shelf_east_diffs[-1] = -480
shelf_west_diffs = np.arange(-3, -45, -2.0)
shelf_west_diffs[0] = 480
asc_east_diffs = np.unique(mask.basins)[22:43] - np.unique(mask.basins)[0:21]
asc_east_diffs[-1] = 45
asc_west_diffs = np.zeros(21)
asc_west_diffs[1:] = np.unique(mask.basins)[21:41] - np.unique(mask.basins)[1:21]
asc_west_diffs[0] = 1849-4
asc_north_diffs = np.unique(mask.basins)[21:42] - np.unique(mask.basins)[0:21]
shelf_asc_diffs = -np.copy(asc_north_diffs)
print(shelf_east_diffs)
print(shelf_west_diffs)
print(asc_east_diffs)
print(asc_west_diffs)
print(asc_north_diffs)
print(shelf_asc_diffs)

# define longitudes, latitudes and distance indices for intersections between zonal basin connectivity mask regions
loncrossing = np.asarray([-280, -262.6, -245.4, -227.6, -208.9, -197.6, -179.6, -166.9, -138.5, -115.3, -90.25, -62.75, -56.85, np.nan, np.nan, 
                          -38.05, -14.65, 6.75, 26.95, 43.45, 61.25])
latcrossing = np.asarray([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -66.28, -61.37, -62.45, -68.99, 
                          np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
distancecrossing = np.asarray([0, 260, 514, 773, 1027, 1309, 1611, 1940, 2303, 2628, 2918, 
                               3213,  3469,  3696,  3940, 4253, 4586, 4888, 5176, 5465, 5736], dtype='int32')
# define basin_values
basin_values = np.unique(mask.basins)
basin_values = basin_values[:-1]
print('Basin values:', basin_values[0:21])


## -------------------------------------------------------------------------------------
## Calculating contour distance
num_points = 6002 
north_index = 510
grid_file2 = '/g/data/v45/akm157/model_data/mom01_unmasked_ocean_grid.nc'
gridFile = nc.Dataset(grid_file2)
dxu = gridFile.variables['dxu'][...]
dyt = gridFile.variables['dyt'][:north_index,:]
xt_ocean = gridFile.variables['xt_ocean'][...]
yt_ocean = gridFile.variables['yt_ocean'][...]
lon_t = gridFile.variables['geolon_t'][:north_index,...]
lat_t = gridFile.variables['geolat_t'][:north_index,...]

# if there is a bend in the contour, add the distance using length of diagonal, not sum of
# 2 edges, to be more representative.
contour_depth = 1000
data = np.load('/g/data/e14/hd4873/model_data/access-om2/Antarctic_slope_contour_1000m.npz')
mask_y_transport_numbered = data['mask_y_transport_numbered']
mask_x_transport_numbered = data['mask_x_transport_numbered']
contour_masked_above = data['contour_masked_above']

lon_along_contour = np.zeros((num_points))
lat_along_contour = np.zeros((num_points))
distance_along_contour = np.zeros((num_points))
x_indices = np.sort(mask_x_transport_numbered[mask_x_transport_numbered>0])
y_indices = np.sort(mask_y_transport_numbered[mask_y_transport_numbered>0])
skip = False
for count in range(1,num_points):
    if skip == True:
        skip = False
        continue
    if count in y_indices:
        if count + 1 in y_indices:
            # note dxu and dyt do no vary in x:
            jj = np.where(mask_y_transport_numbered==count)[0]
            # me added
            kk = np.where(mask_y_transport_numbered==count)[1]
            # ----
            distance_along_contour[count-1] = (dxu[jj,990])[0]
            lon_along_contour[count-1] = lon_t[0, kk]
            lat_along_contour[count-1] = lat_t[jj, 990]
        else:
            jj0 = np.where(mask_y_transport_numbered==count)[0]
            jj1 = np.where(mask_x_transport_numbered==count+1)[0]
            
            # me added
            kk0 = np.where(mask_y_transport_numbered==count)[1]
            kk1 = np.where(mask_x_transport_numbered==count+1)[1]
            # --
            diagonal_distance = 0.5*np.sqrt((dxu[jj0,990])[0]**2+\
                (dyt[jj1,990])[0]**2)
            distance_along_contour[count-1] = diagonal_distance
            distance_along_contour[count] = diagonal_distance
            
            lon_along_contour[count-1] = lon_t[0, kk0] 
            lat_along_contour[count-1] = lat_t[jj0, 990]
            lon_along_contour[count] = lon_t[0, kk1] 
            lat_along_contour[count] = lat_t[jj0, 990]
            # skip to next count:
            skip = True
    # count in x_indices:
    else:
        if count + 1 in x_indices:
            jj = np.where(mask_x_transport_numbered==count)[0]
            # me added
            kk = np.where(mask_x_transport_numbered==count)[1]
            # ----
            distance_along_contour[count-1] = (dyt[jj,990])[0]
            lon_along_contour[count-1] = lon_t[0, kk]
            lat_along_contour[count-1] = lat_t[jj, 990]
        else:
            jj0 = np.where(mask_x_transport_numbered==count)[0]
            jj1 = np.where(mask_y_transport_numbered==count+1)[0]
            
            # me added
            kk0 = np.where(mask_x_transport_numbered==count)[1]
            kk1 = np.where(mask_y_transport_numbered==count+1)[1]
            # --
            diagonal_distance = 0.5*np.sqrt((dyt[jj0,990])[0]**2+\
                (dxu[jj1,990])[0]**2)
            distance_along_contour[count-1] = diagonal_distance
            distance_along_contour[count] = diagonal_distance
            
            lon_along_contour[count-1] = lon_t[0, kk0] 
            lat_along_contour[count-1] = lat_t[jj0, 990]
            lon_along_contour[count] = lon_t[0, kk1] 
            lat_along_contour[count] = lat_t[jj0, 990]
            
            # skip to next count:
            skip = True

# fix last value:
if distance_along_contour[-1] == 0:
    count = count + 1
    if count in y_indices:
        jj = np.where(mask_y_transport_numbered==count)[0]
        #kk = np.where(mask_y_transport_numbered==count)[1]
        distance_along_contour[-1] = (dxu[jj,990])[0]
        #lon_along_contour[count-1] = lon_t[0, kk]
        #lat_along_contour[count-1] = lat_t[jj, 990]
    else:
        jj = np.where(mask_x_transport_numbered==count)[0]
        #kk = np.where(mask_x_transport_numbered==count)[1]
        distance_along_contour[-1] = (dyt[jj,990])[0]
        #lon_along_contour[count-1] = lon_t[0, kk]
        #lat_along_contour[count-1] = lat_t[jj, 990]

# units are 10^3 km:
distance_along_contour = np.cumsum(distance_along_contour)/1e3/1e3
# units are now in km
distance_along_contour = distance_along_contour*1e3

# create distance array
lats_arr = xr.DataArray(lat_along_contour, coords={'distance': distance_along_contour}, dims=['distance'])
lons_arr = xr.DataArray(lon_along_contour, coords={'distance': distance_along_contour}, dims=['distance'])
distance_arr = xr.Dataset({'lon': lons_arr, 'lat': lats_arr})
print(distance_arr.lat.mean().values) 


## --------------------------------------------------------------------------
## Connectivity Analysis
## Find particle counts, tranport and percentages that make it to adjacent shelf region
## AND which also travel 400km (~10 degrees longitude at average latitude of shelf break isobath)
print(i, f'Starting connectivity analysis for basin: {basin_values[i]}')
#shelf_region = 'shelf_east'
bas = basin_values[i]
outfile = f'/g/data/e14/hd4873/runs/parcels/output/AntConn/data/zonal_conn/zonal_connectivity_{region}_{i:02d}.nc'
print(i, bas, outfile)
# find first [-1, since distances increase to west not east] distance index 400 km  downstream of zonal connectivity basin crossing
distidx = np.where(distance_arr.distance >= distance_arr.distance[distancecrossing[i]] + 400)[0][0]
# find the longitude and latitude at this distance downstream/upstream
distlon = distance_arr.lon[distidx].values
distlat = distance_arr.lat[distidx].values
distidx2 = np.where(distance_arr.distance >= distance_arr.distance[distancecrossing[i]] + 800)[0][0]
distlon2 = distance_arr.lon[distidx2].values
distlat2 = distance_arr.lat[distidx2].values

# find indices where particles cross from basin in west, to adjacent basin in east
idx = np.unique(np.where(ds.basindiff == shelf_west_diffs[i])[1])
print(len(idx))
# load subset datasets
dstmp = ds.isel(trajectory = idx)
dslatlontmp = dslatlon.isel(trajectory = idx).load()
dsivtmp = ds_iv.isel(trajectory = idx)

# to speed up loop below, I'm going to find the particles (indices) that ALSO cross to the next basin downstream 
# (on the shelf) and assume these travel along the shelf for the 400km required. 
# this reduces the size of my loop.
idxtmp = np.unique(np.where(dstmp.basindiff == shelf_west_diffs[i])[1])
#if i == 0:
#    idx2 = np.unique(np.where(dstmp.basindiff == shelf_east_diffs[-1])[1])
#else:
#    idx2 = np.unique(np.where(dstmp.basindiff == shelf_east_diffs[i-1])[1])


# invert .isin() test so that values NOT in idx2 come up as true and select out only these indices
#subidx = idxtmp[np.isin(idxtmp, idx2, invert=True)]
# create ampty array to save indices into
arr = np.array([])
# first save all the indices that cross to the next basin downstream (the particles from idx that are also in idx2)
#arr = np.append(arr, idxtmp[np.isin(idxtmp, idx2)])
#print(len(subidx), len(arr), len(subidx)+len(arr))
print(len(idxtmp))

# Now loop through the remaining indices and find which ones travel 400 km downstream
for j, k in enumerate(idxtmp):
    if j % 20000 == 0:
        print(j,k)
    crossid = np.where(dstmp.basindiff[:,k] == shelf_west_diffs[i])[0]
    if len(crossid) > 1:
        crossid = int(crossid[0])
    else:
        crossid = int(crossid)
            
    if i == 12:
        timeindxs = np.where((dslatlontmp.lon[crossid:,k] >= distlon))[0]
    elif i == 13:
        timeindxs = np.where((dslatlontmp.lat[crossid:,k] <= distlat) & (dslatlontmp.lat[crossid:,k] >= distlat2) & (dslatlontmp.lon[crossid:,k] <= -50))[0]
    elif i == 14:
        timeindxs = np.where((dslatlontmp.lat[crossid:,k] <= distlat) & (dslatlontmp.lon[crossid:,k] >= -65))[0]
    else:
        # find if particles crosses west of 400km downstream while on the shelf
        timeindxs = np.where((dslatlontmp.lon[crossid:,k] >= distlon) & (dslatlontmp.lon[crossid:,k] <= distlon2))[0]
        # maybe particle doesn't need to still be on the shelf 400 km downstream?? 
        # afterall this is just to avoid counting particles that meander into that shelf region and then turn back
        # i.e. I want ones that continue west/east a sufficient distance, doesn't matter if they go back off the shelf after this
        #timeindxs = np.where((dslatlontmp.lon[crossid:,j] <= distlon) & (dstmp.basin_ZonalConn[crossid:,j] == bas))[0]
    if len(timeindxs > 0):
        arr = np.append(arr, int(k))

# sort array
arr = np.sort(arr)
arr = np.asarray(arr, dtype='int32')
# find indices of all particles in upstream (source) basin to east
if i == 0:
    transids = np.unique(np.where(ds.basin_ZonalConn == basin_values[20])[1])
else:
    transids = np.unique(np.where(ds.basin_ZonalConn == basin_values[i-1])[1])

# save counts, transport and percentages to connectivity_count array
connectivity_count.conn_count[i,4] = len(arr)
connectivity_count.conn_count_pct[i,4] = len(arr)/len(transids)*100
connectivity_count.conn_trans[i,4] = np.abs(dsivtmp.trans[arr]).sum()
connectivity_count.conn_trans_pct[i,4] = connectivity_count.conn_trans[i,4] / np.abs(ds_iv.trans[transids]).sum()*100


# save to file
connectivity_count.to_netcdf(outfile)
print('Analysis finished')
