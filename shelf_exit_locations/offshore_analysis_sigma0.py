#!/usr/bin/env python
# coding: utf-8

# Author: Hannah Dawson
# Date: 22 August 2021
# Description: Find locations of shelf exit equatorward across the 1000m isobath

# BEGIN --------------------------------------------------------------------------

# Importing the relevant modules. 
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
import gsw
from datetime import datetime

# Get submission arguments (passed to python script with submit_python_script):
import sys
filenum = int(sys.argv[1])
num1 = 27.8
num2 = 30


# Define particle files
npart = 130146
datadir = '/g/data/e14/hd4873/runs/parcels/output/AntConn/data/xarray_files/traj_chunked_basin/'
files = sorted(glob(datadir+'CircumAntarcticParticles_*.nc')) 

# read in shelf mask: Shelf = 1, Southern Ocean = 0 
# NOTE: This mask is based on u grid, not T grid. So 
# mask = xr.open_dataset('/g/data/e14/hd4873/runs/parcels/output/proj01/data/basin_masks/antarctic_shelf_mask_hu.nc')

# read in starting values
startfile = '/g/data/e14/hd4873/runs/parcels/output/AntConn/data/CircumAntarcticParticles_initial_values.nc'
npart = 130146
t1 = filenum*npart
t2 = t1+npart
ref = xr.open_dataset(startfile).isel(trajectory=slice(t1, t2))
p = gsw.p_from_z(-ref.shelf_exit_z, ref.shelf_exit_lat)
sa = gsw.SA_from_SP(ref.shelf_exit_S, p, ref.shelf_exit_lon, ref.shelf_exit_lat)
ref['sigma0'] = gsw.density.sigma0(sa, ref.shelf_exit_T)
ref['sigma1'] = gsw.density.sigma1(sa, ref.shelf_exit_T)


## set histogram parameters
xstart, xend = -280, 80  # starting longitude, final longitude
xbinlim = [xstart, xend]  # longitude bin limits
ystart, yend = -81, -55    # starting latitude,  final latitude
ybinlim = [ystart, yend]  # loatitude bin limits
dx_map, dy_map = 1, 0.5     # x-grid size, y-grid size for 2D pdf maps 
xbins = int(360/dx_map)   # number of longitude bins
ybins = int(26/dy_map)    # number of latitude bins
xarange = np.arange(xstart, xend+dx_map, dx_map)
yarange = np.arange(ystart, yend+dy_map, dy_map)
xmid = (xarange[1:]+xarange[:-1])/2. # longitude midpoints
ymid = (yarange[1:]+yarange[:-1])/2. # latitude midpoints

# create emtpy array for 2D histogram count
in_box_off = np.zeros((xbins,ybins))
in_box_off_norm = np.zeros((xbins,ybins))
in_box_off_transnorm  = np.zeros((xbins,ybins))
in_box_off_trans = np.zeros((xbins, ybins))

# Don't include particles that are still on the shelf at the end of the simulation
exitparts = np.where(ref.shelf_exit_indx < 1512)[0] 
lons_off = ref.shelf_exit_lon[exitparts]
lats_off = ref.shelf_exit_lat[exitparts]
trans_off = ref.trans[exitparts]
norm_off = ref.shelf_weighted_val[exitparts]
z_off = ref.shelf_exit_z[exitparts]
sigma0_off = ref.sigma0[exitparts]
sigma1_off = ref.sigma1[exitparts]

# subset for sigma0 range
ids = np.where((sigma0_off >= num1) & (sigma0_off < num2))
lons_off = lons_off[ids]
lats_off = lats_off[ids]
trans_off = trans_off[ids]
norm_off = norm_off[ids]
z_off = z_off[ids]
sigma0_off = sigma0_off[ids]

# Now run loop
for n in range(len(lons_off)):
    if n % 40000 == 0:
        print(n)
    x_off = np.array((lons_off[n].values, np.nan)) # last shelf longitude of particle n
    y_off = np.array((lats_off[n].values, np.nan))  # last shelf latitude of particle n
    H_off = np.histogram2d(x_off,y_off,[xbins,ybins],[xbinlim,ybinlim])
    boxind_off = np.nonzero(H_off[0])
    in_box_off[boxind_off[0],boxind_off[1]] = in_box_off[boxind_off[0],boxind_off[1]]+1  # count
    in_box_off_norm[boxind_off[0],boxind_off[1]] = in_box_off_norm[boxind_off[0],boxind_off[1]]+(1*norm_off[n].values) # Normalised
    in_box_off_trans[boxind_off[0],boxind_off[1]] = in_box_off_trans[boxind_off[0],boxind_off[1]]+(1*trans_off[n].values) # Normalised
    in_box_off_transnorm[boxind_off[0],boxind_off[1]] = in_box_off_transnorm[boxind_off[0],boxind_off[1]] + (1*norm_off[n].values*np.abs(trans_off[n].values)) # Normalised with transport

# convert to DataArrays
count_offshore = xr.DataArray(data = in_box_off, dims=["lon", "lat"], coords = {"lat": ymid, "lon": xmid})
count_offshore = count_offshore.where(count_offshore>0, np.nan)
count_offshore.attrs["Long name"] = "Count of last location (binned) before particles exit the continental shelf"
count_offshore.attrs["More info"] = "Particle count is not normalised. Each particle carries same weighting"
count_offshore_norm = xr.DataArray(data = in_box_off_norm, dims=["lon", "lat"], coords = {"lat": ymid, "lon": xmid})
count_offshore_norm = count_offshore_norm.where(count_offshore_norm>0, np.nan)
count_offshore_transnorm = xr.DataArray(data = in_box_off_transnorm, dims=["lon", "lat"], coords = {"lat": ymid, "lon": xmid})
count_offshore_transnorm = count_offshore_transnorm.where(count_offshore_transnorm>0, np.nan)
count_offshore_trans = xr.DataArray(data = in_box_off_trans, dims=["lon", "lat"], coords = {"lat": ymid, "lon": xmid})
count_offshore_trans = count_offshore_trans.where(count_offshore_trans>0, np.nan)

# Add to dataset
ds_count = count_offshore.to_dataset(name="count")
ds_count["count_norm"] = count_offshore_norm
ds_count["count_norm"].attrs["Long name"] = "Normalised count of last location (binned) before particles exit the continental shelf"
ds_count["count_norm"].attrs["More info"] = "Particle count in each bin is normalised by particle weighting (ref.shelf_weighted_val)"
ds_count["count_transnorm"] = count_offshore_transnorm
ds_count["count_transnorm"].attrs["Long name"] = "Transport normalised count of last location (binned) before particles exit the continental shelf"
ds_count["count_transnorm"].attrs["More info"] = "Particle count in each bin is normalised by particle weighting (ref.shelf_weighted_val) * particle transport"
ds_count["count_trans"] = count_offshore_trans
ds_count["count_trans"].attrs["Long name"] = "Transport sum of last location (binned) before particles exit the continental shelf"
ds_count["count_trans"].attrs["More info"] = "Particle count in each bin is * particle transport"
ds_count.attrs["Date created"] = str(datetime.now())[0:10]
ds_count.attrs["Resolution"] = '{} degree longitude x {} degree latitude'.format(dx_map, dy_map)


# Define output file
dataoutdir = '/g/data/e14/hd4873/runs/parcels/output/AntConn/data/offshore/'
outfile = dataoutdir+'OffshoreParticles_{}x{}degree_file-{:02d}_{}-{}sigma0.nc'.format(dx_map, dy_map, filenum, num1, num2)

# Save to netCDF
print("Saving to netCDF file")
encod={}
for var in ds_count.data_vars:
    encod[var]={'zlib':True}
ds_count.to_netcdf(outfile)

# END --------------------------------------------------------------------------