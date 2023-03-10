# # Custom functions to process and anlyses OceanParcels trajectory data
# #
# #
# # Author: Hannah Dawson
# # Email: hannah.dawson@unsw.edu.au
# # Date: 23 April 2021
# #-----------------------------------------------------------------------


def shelf_mask_isobath(var):
    '''
    Masks ACCESS-OM2-01 variables by the region polewards of the 1000m isobath as computed using 
    a script contributed by Adele Morrison.
    Only to be used with ACCESS-OM2-0.1 output!
    '''

    import numpy as np
    import xarray as xr

    contour_file = np.load('/g/data/e14/hd4873/runs/parcels/input/Antarctic_slope_contour_hu_1000m.npz')
    shelf_mask = contour_file['contour_masked_above']
    yu_ocean = contour_file['yu_ocean']
    xu_ocean = contour_file['xu_ocean']
    
    # in this file the points along the isobath are given a positive value, the points outside (northwards) 
    # of the isobath are given a value of -100 and all the points on the continental shelf have a value of 0 
    # so we mask for the 0 values 
    shelf_mask[np.where(shelf_mask!=0)] = np.nan
    shelf_mask = shelf_mask+1
    shelf_map = np.nan_to_num(shelf_mask)
    shelf_mask = xr.DataArray(shelf_mask, coords = [('yu_ocean', yu_ocean), ('xu_ocean', xu_ocean)])
    shelf_map = xr.DataArray(shelf_map, coords = [('yu_ocean', yu_ocean), ('xu_ocean', xu_ocean)])
    
    # then we want to multiply the variable with the mask so we need to account for the shape of the mask. 
    # The mask uses a northern cutoff of 59S.
    masked_var = var.sel(yu_ocean = slice(-90, -59.03)) * shelf_mask
    return masked_var, shelf_map

def shelf_mask_isobath_tcell(var):
    '''
    Masks ACCESS-OM2-01 variables by the region polewards of the 1000m isobath as computed using 
    a script contributed by Adele Morrison.
    Only to be used with ACCESS-OM2-0.1 output!
    '''

    import numpy as np
    import xarray as xr

    contour_file = np.load('/g/data/e14/hd4873/runs/parcels/input/Antarctic_slope_contour_ht_1000m.npz')
    shelf_mask = contour_file['contour_masked_above']
    yt_ocean = contour_file['yt_ocean']
    xt_ocean = contour_file['xt_ocean']
    
    # in this file the points along the isobath are given a positive value, the points outside (northwards) 
    # of the isobath are given a value of -100 and all the points on the continental shelf have a value of 0 
    # so we mask for the 0 values 
    shelf_mask[np.where(shelf_mask!=0)] = np.nan
    shelf_mask = shelf_mask+1
    shelf_map = np.nan_to_num(shelf_mask)
    shelf_mask = xr.DataArray(shelf_mask, coords = [('yt_ocean', yt_ocean), ('xt_ocean', xt_ocean)])
    shelf_map = xr.DataArray(shelf_map, coords = [('yt_ocean', yt_ocean), ('xt_ocean', xt_ocean)])
    
    # then we want to multiply the variable with the mask so we need to account for the shape of the mask. 
    # The mask uses a northern cutoff of 59S.
    masked_var = var.sel(yt_ocean = slice(-90, -59.03)) * shelf_mask
    return masked_var, shelf_map

def add_release_time(ds, particles_per_release):
    '''
    Add a variable 'release_time' to an Xarray dataset ds using the number of particles_per_release.
    '''
    import numpy as np
    import xarray as xr

    release_time = np.zeros((len(ds.trajectory), len(ds.time)))
    for i in range(73):
        init = i*particles_per_release
        finl = (i+1)*particles_per_release
        release_time[init:finl, :] = ds.time[i]
    # add variable to dataset
    release_time = xr.DataArray(release_time, coords=[ds.trajectory, ds.time], dims=["trajectory", "time"])
    ds['release_time'] = release_time
    
    return ds
