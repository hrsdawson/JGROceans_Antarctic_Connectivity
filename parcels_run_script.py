#!/usr/bin/env python
# coding: utf-8

# # - Parcels run script for PROJECT 01: Pathways and Timescales of Connectivity at the Antarctic Margin - ##
# # - works with ACCESS-OM2-01 output - ##

# Importing the relevant modules. 
import numpy as np
import xarray as xr
import math
from glob import glob
from datetime import timedelta
import time

# import Parcels from home directory (not hh5 version)
MODULE_PATH = "/home/561/hd4873/software/parcels/parcels/__init__.py"
MODULE_NAME = "parcels"
import importlib
import sys
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from parcels import FieldSet, ParticleSet, Variable, JITParticle, ScipyParticle, AdvectionRK4_3D, ErrorCode, VectorField, ParcelsRandom
import parcels
# check we are at least one version beyond parcels 2.1.4, in which the chunking doesn't work:
print('parcels', parcels.__version__)
print(parcels.__file__)

######## get run count argument and gdata_dir that was passed to python script:
run_count = int(sys.argv[1])
run_name = sys.argv[2]
print(run_name)
# change this to your output path on /g/data/:
gdata_path = '/g/data/e14/hd4873/runs/parcels/output/'+run_name+'/data/parcels_files/OtherYears/'
# max number of time loops to run:
n_loops = sys.argv[3]

######## Parcels Code below
# Define ACCESS-OM2-01 (MOM5 ocean) NetCDF files.
data_path = '/g/data/ik11/outputs/access-om2-01/01deg_jra55v13_ryf9091/'
ufiles = sorted(glob(data_path+'output19*/ocean/ocean_daily_3d_u_*.nc')) + sorted(glob(data_path+'output2*/ocean/ocean_daily_3d_u_*.nc'))
vfiles = sorted(glob(data_path+'output19*/ocean/ocean_daily_3d_v_*.nc')) + sorted(glob(data_path+'output2*/ocean/ocean_daily_3d_v_*.nc'))
wfiles = sorted(glob(data_path+'output19*/ocean/ocean_daily_3d_wt_*.nc')) + sorted(glob(data_path+'output2*/ocean/ocean_daily_3d_wt_*.nc'))
tfiles = sorted(glob(data_path+'output19*/ocean/ocean_daily_3d_temp_*.nc')) + sorted(glob(data_path+'output2*/ocean/ocean_daily_3d_temp_*.nc'))
sfiles = sorted(glob(data_path+'output19*/ocean/ocean_daily_3d_salt_*.nc')) + sorted(glob(data_path+'output2*/ocean/ocean_daily_3d_salt_*.nc'))
mldfiles = sorted(glob(data_path+'output19*/ocean/ocean_daily.nc')) + sorted(glob(data_path+'output2*/ocean/ocean_daily.nc'))
mesh_mask = '/g/data/e14/hd4873/runs/parcels/input/proj01/access-om2-01_Southern_Ocean_depth3d_coords.nc'

filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': mesh_mask, 'data': ufiles},
             'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': mesh_mask, 'data': vfiles},
             'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': mesh_mask, 'data': wfiles},
             'T': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': mesh_mask, 'data': tfiles},
             'S': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': mesh_mask, 'data': sfiles},
             'ML': {'lon': mesh_mask, 'lat': mesh_mask, 'data': mldfiles},
             }

# Define a dictionary of the variables (`U`, `V` and `W`) and dimensions (`lon`, `lat`,`depth` and `time`). 
# All variables must have the same lat/lon/depth dimensions (even though the data doesn't).
variables = {'U': 'u',
             'V': 'v',
             'W': 'wt',
             'T': 'temp',
             'S': 'salt',
             'ML': 'mld',
             }
dimensions = {'U': {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'sw_ocean', 'time': 'time'},
              'V': {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'sw_ocean', 'time': 'time'},
              'W': {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'sw_ocean', 'time': 'time'},
              'T': {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'sw_ocean', 'time': 'time'},
              'S': {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'sw_ocean', 'time': 'time'},
              'ML':{'lon': 'xu_ocean', 'lat': 'yu_ocean', 'time': 'time'},
              }

# Define the chunksizes
cs = {"U": {"lon": ("xu_ocean", 400), "lat": ("yu_ocean", 300), "depth": ("st_ocean", 1), "time": ("time", 1)}, 
      "V": {"lon": ("xu_ocean", 400), "lat": ("yu_ocean", 300), "depth": ("st_ocean", 1), "time": ("time", 1)},
      "W": {"lon": ("xt_ocean", 400), "lat": ("yt_ocean", 300), "depth": ("sw_ocean", 1), "time": ("time", 1)},
      "T": {"lon": ("xt_ocean", 400), "lat": ("yt_ocean", 300), "depth": ("st_ocean", 1), "time": ("time", 1)},
      "S": {"lon": ("xt_ocean", 400), "lat": ("yt_ocean", 300), "depth": ("st_ocean", 1), "time": ("time", 1)},
      "ML": {"lon": ("xt_ocean", 400), "lat": ("yt_ocean", 300), "time": ("time", 1)},
      }

# Read in the fieldset using the Parcels `FieldSet.from_mom5` function. 
fieldset = FieldSet.from_mom5(filenames, variables, dimensions,
                              mesh = 'spherical', 
                              #chunksize=cs, 
                              tracer_interp_method = 'bgrid_tracer')

# Add constant to fieldset to convert from geometric to geographic coordinates (m to degrees)
fieldset.add_constant('geo', 1/(1852*60))
# Add a Wmax constant (1 cm/s) for the mixed layer shuffling kernel 
fieldset.add_constant('mixedlayerwmax', 0.01)


# Add unbeaching vectorfield to fieldset
fileUB = str('/g/data/e14/hd4873/runs/parcels/input/proj01/MOM5_unbeach_vel_using_ucell_corners.nc')
mesh_mask = '/g/data/e14/hd4873/runs/parcels/input/proj01/access-om2-01_Southern_Ocean_depth3d_coords.nc'
filenamesUB = {'Ub': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': mesh_mask, 'data': fileUB},
               'Vb': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': mesh_mask, 'data': fileUB},
               'land': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': mesh_mask, 'data': fileUB},
              }
variablesUB = {'Ub': 'unBeachU',
               'Vb': 'unBeachV',
               'land': 'land',
               }
dimsUB = {"Ub": {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'sw_ocean', 'time': 'time'},
          "Vb": {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'sw_ocean', 'time': 'time'},
          "land": {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'depth': 'sw_ocean', 'time': 'time'},
          }
#csUB = {"Ub": {"lon": ("xu_ocean", 400), "lat": ("yu_ocean", 300), "time": ("time", 1)}, 
#        "Vb": {"lon": ("xu_ocean", 400), "lat": ("yu_ocean", 300), "time": ("time", 1)},
#        "land": {"lon": ("xu_ocean", 400), "lat": ("yu_ocean", 300), "time": ("time", 1)},
#       }
fieldsetUB = FieldSet.from_mom5(fileUB, variablesUB, dimsUB, 
                                #chunksize=csUB,
                                mesh = 'spherical',
                                allow_time_extrapolation=True)

# add to fieldset and set units and interp_method
fieldset.add_field(fieldsetUB.Ub, 'Ub')
fieldset.add_field(fieldsetUB.Vb, 'Vb')
fieldset.add_field(fieldsetUB.land, 'land')
fieldset.Ub.units = parcels.tools.converters.GeographicPolar()
fieldset.Vb.units = parcels.tools.converters.Geographic()
fieldset.land.units = parcels.tools.converters.UnitConverter()
fieldset.Ub.interp_method = 'bgrid_velocity'
fieldset.Vb.interp_method = 'bgrid_velocity'
fieldset.land.interp_method = 'bgrid_velocity'
# Add vector field
UVunbeach = VectorField('UVunbeach', fieldset.Ub, fieldset.Vb)
fieldset.add_vector_field(UVunbeach)


# Add 2D sea level contour field as northern boundary for deleting particles
file_contour = str('/g/data/e14/hd4873/runs/parcels/input/proj01/sea_level_contour_ryf9091_N_cutoff.nc')
variables_contour = {'NBound': 'sea_level_contour',}
dim_contour = {'NBound': {'lon': 'xt_ocean', 'lat': 'yt_ocean'},}
fieldsetC = FieldSet.from_b_grid_dataset(file_contour, variables_contour, dim_contour,
                                                allow_time_extrapolation=True)
# Add Northern contour boundary to fieldset
fieldset.add_field(fieldsetC.NBound, 'NBound')


# Add 2D shelf field for defining particles on the shelf
shelf_file = str('/g/data/e14/hd4873/runs/parcels/input/proj01/continental_shelf_field_ht_expanded_ytocean_dim.nc')
variables_shelf = {'shelf': 'shelf_region',}
dim_shelf = {'shelf': {'lon': 'xt_ocean', 'lat': 'yt_ocean'},}
fieldsetD = FieldSet.from_b_grid_dataset(shelf_file, variables_shelf, dim_shelf,
                                                allow_time_extrapolation=True)
# Add shelf field to fieldset
fieldset.add_field(fieldsetD.shelf, 'shelf')


# Add a halo for periodic zonal boundaries.  
fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
fieldset.add_periodic_halo(zonal=True)

# -----------------------------------------------------------------------------
# Define custom particle class and kernels

# Define custom particle class and kernels
class SampleParticle(JITParticle):
    thermo = Variable('thermo', dtype=np.float32, initial = 0.)
    psal = Variable('psal', dtype=np.float32, initial = 0.)
    mldepth = Variable('mldepth', dtype=np.float32, initial = 0.)
    beached = Variable('beached', dtype=np.float32, initial=0., to_write=False)
    unbeachCount = Variable('unbeachCount', dtype=np.float32, initial=0.)
    mixedlayershuffle = Variable('mixedlayershuffle', dtype=np.float32, initial = 0.)
    onshelf = Variable('onshelf', dtype=np.float32, initial = 1.) 

# Kernel that can move the particle from one side of the domain to the other.
def periodicBC(particle, fieldset, time):
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west

# Kernel to delete particles if they cross the Northern Boundary
def NorthTest(particle, fieldset, time):
    north_val = fieldset.NBound[0., 0., particle.lat, particle.lon]
    if north_val == 0.:
        #print('WARNING: Particle crossed northern boundary. NorthTest deleted a particle.')
        particle.delete() 

# Recovery kernel that will be invoked with particles encounter ErrorOutOfBounds. This deletes particles that leave the domain.
def DeleteParticle(particle, fieldset, time):
    particle.delete()

# Kernel to sample T, S and ML
def SampleFields(particle, fieldset, time):
    particle.thermo = fieldset.T[time, particle.depth, particle.lat, particle.lon]
    particle.psal = fieldset.S[time, particle.depth, particle.lat, particle.lon]
    particle.mldepth = fieldset.ML[time, 0, particle.lat, particle.lon]

# Advection RK4 kernel that only acts on particles which NOT beached
def AdvectionRK4_3D_B(particle, fieldset, time):
    if particle.beached == 0:
        (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
        lon1 = particle.lon + u1*.5*particle.dt
        lat1 = particle.lat + v1*.5*particle.dt
        dep1 = particle.depth + w1*.5*particle.dt
        (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
        lon2 = particle.lon + u2*.5*particle.dt
        lat2 = particle.lat + v2*.5*particle.dt
        dep2 = particle.depth + w2*.5*particle.dt
        (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
        lon3 = particle.lon + u3*particle.dt
        lat3 = particle.lat + v3*particle.dt
        dep3 = particle.depth + w3*particle.dt
        (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
        particle.depth += (w1 + 2*w2 + 2*w3 + w4) / 6. * particle.dt

# Kernel to test if particles are beached
def BeachTest(particle, fieldset, time):
    ld = fieldset.land[0., particle.depth, particle.lat, particle.lon]
    if ld >= 0.985:
        particle.beached += 1
    else:
        particle.beached = 0

def UnBeaching_constUV(particle, fieldset, time):
    if particle.beached >= 1:
        #extract unbeaching velocities
        (ub, vb) = fieldset.UVunbeach[0., particle.depth, particle.lat, particle.lon]
        # define velocities to be 1 m/s
        ubx = fieldset.geo * (1/math.cos(particle.lat*(math.pi/180)))
        # move particle by 1 m/s using sign from ub and vb
        particle.lon += math.copysign(ubx, ub) * math.fabs(particle.dt)
        particle.lat += math.copysign(fieldset.geo, vb) * math.fabs(particle.dt)
        # increase unbeaching count
        particle.unbeachCount += 1
        # check that particle is now unbeached and change beached indicator if so
        ld2 = fieldset.land[0., particle.depth, particle.lat, particle.lon]
        if ld2 < 0.985:
            particle.beached = 0

def MLshuffle(particle, fieldset, time):
    mld = fieldset.ML[time, 0., particle.lat, particle.lon]
    if particle.depth < mld:
        maxdepthchange = fieldset.mixedlayerwmax * particle.dt
       
        # find distance between particle depth and both surface and mld depth
        z0_dist = particle.depth
        zmld_dist = mld - particle.depth
        
        # if distance to surface or mld is < maxdepthchange, update the maxdepthchange to that smaller distance
        if z0_dist < maxdepthchange:
            maxdepthchange = z0_dist
        if zmld_dist < maxdepthchange:
            maxdepthchange = zmld_dist
        
        # define new min and max depth changes
        mindepth = particle.depth - maxdepthchange
        if mindepth <= 0:
            mindepth = 0.2
        maxdepth = particle.depth + maxdepthchange
        if maxdepth > mld:
            maxdepth = mld

        particle.depth = ParcelsRandom.uniform(mindepth,maxdepth)
        particle.mixedlayershuffle += 1

def ShelfTest(particle, fieldset, time):
    shelfbool = fieldset.shelf[0., 0., particle.lat, particle.lon]
    if shelfbool >= 0.985:
        particle.onshelf = 1


#------------------------------------------------------------------------------------
# Define particle set here for first run:
# for first run locate release locations, else load from file:
print('Starting run '+str(run_count))
if run_count == 0:
    ds = xr.open_dataset('/g/data/e14/hd4873/runs/parcels/input/proj01/release_locations.nc')

    # Define particle set
    pset = ParticleSet.from_list(fieldset=fieldset,   # the fields on which the particles are advected
                             pclass=SampleParticle,   # the type of particles (JITParticle or ScipyParticle),
                             time = 0, 
                             lon = ds.lons,           # vector of release longitudes 
                             lat = ds.lats,           # vector of release latitudes
                             depth = ds.depths,       # vector of release depths
                             ) 
# it should never actually get here:
elif run_count == n_loops:
    print('Last loop, exiting parcels')
    sys.exit()
elif run_count > 0 and run_count < 73:
    outfiles = sorted(glob(gdata_path+'*.nc'))
    # pick last file to restart particles from:
    restart_file = outfiles[-1]
    # Define particle set from file:
    pset = ParticleSet.from_particlefile(fieldset=fieldset,pclass=SampleParticle,
                                         filename=restart_file,restart=True)
    # read in particle release locations and define new particle set:    
    ds = xr.open_dataset('/g/data/e14/hd4873/runs/parcels/input/proj01/release_locations.nc')
    time_restart = run_count*5*24*60*60 # restart time in seconds
    pset1 = ParticleSet.from_list(fieldset=fieldset,  # the fields on which the particles are advected
                             pclass=SampleParticle,   # the type of particles (JITParticle or ScipyParticle),
                             time = time_restart, 
                             lon = ds.lons,           # vector of release longitudes 
                             lat = ds.lats,           # vector of release latitudes
                             depth = ds.depths,       # vector of release depths
                             ) 
    # add new particle set (pset1) to original particle set (pset)
    pset.add(pset1)
elif run_count >= 73:
    outfiles = sorted(glob(gdata_path+'*.nc'))
    # pick last file to restart particles from:
    restart_file = outfiles[-1]
    # Define particle set from file:
    pset = ParticleSet.from_particlefile(fieldset=fieldset,pclass=SampleParticle,
                                         filename=restart_file,restart=True)

# run parcels:
output_file= pset.ParticleFile(name="CircumAntarcticParticles_"+str(run_count).zfill(3)+".nc", 
                               outputdt=timedelta(days=5)) # the file name and the time step of the outputs
if run_count < 73:
    pset.execute(AdvectionRK4_3D_B
             + pset.Kernel(periodicBC)
             + pset.Kernel(NorthTest)
             + pset.Kernel(BeachTest)
             + pset.Kernel(UnBeaching_constUV)
             + pset.Kernel(MLshuffle)
             + pset.Kernel(SampleFields)
             + pset.Kernel(ShelfTest),
             runtime=timedelta(days=5),    # the total length of the run
             dt=timedelta(minutes=60),     # the timestep of the kernel
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle, ErrorCode.ErrorInterpolation: DeleteParticle})
    output_file.close()  # export the trajectory data to a netcdf file
    print('Run '+str(run_count)+' finished')
else: 
    pset.execute(AdvectionRK4_3D_B
             + pset.Kernel(periodicBC)
             + pset.Kernel(NorthTest)
             + pset.Kernel(BeachTest)
             + pset.Kernel(UnBeaching_constUV)
             + pset.Kernel(MLshuffle)
             + pset.Kernel(SampleFields)
             + pset.Kernel(ShelfTest),
             runtime=timedelta(days=50),    # the total length of the run
             dt=timedelta(minutes=60),      # the timestep of the kernel
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle, ErrorCode.ErrorInterpolation: DeleteParticle})
    output_file.close()  # export the trajectory data to a netcdf file
    print('Run '+str(run_count)+' finished')
