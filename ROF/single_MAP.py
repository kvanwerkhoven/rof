
import nwm_processing as nwm
import importlib
importlib.reload(nwm)
import numpy as np
import geopandas as gpd
import pandas as pd
import os
from datetime import datetime, timedelta
import time

domain = 'conus'

###########***REPLACE THIS STUFF TO TEST #################

# shapefile at least 1 attribute for polygon ID (could be gage ID, HUC ID...)
shp_file = 'C:/repos/git/nwm/rof/output/shp/20210328_t00/ROF_mask_20210328_t00_conus.shp'
# header of the polygon ID column in shapefile (e.g. when read in via geopandas, this is a column label)
shp_header = 'HUC10'
# NWM forcing grid
grid_path = 'D:/NWM_Cache/nwm.20210328/forcing_short_range/nwm.t00z.short_range.forcing.f001.conus.nc'

########### shapefile info, check projection ##############

# read shapefile into geodataframe, get list of polygon IDs
print('Reading shapefile for MAP processing and checking projection')
gdf_poly = gpd.read_file(shp_file)
polyids = gdf_poly[shp_header].values   

# check shapefile projection, reproject if needed to match nwm_grid projection
gdf_poly, is_reproj = nwm.shp_to_crs(gdf_poly, domain)

if is_reproj: 
    shp_file = shp_file.split('.')[0] + '_proj.shp'
    print('Exporting reprojected geodataframe as shapefile to use in MAP functions', shp_file)
    gdf_poly.to_file(shp_file)
        
################### get filename, calculate MAP ##############
    
print('Calculating mean areal values for grid: ', grid_path)
print('Number polygons:', gdf_poly.shape[0])

t_start = time.time()

df_zstats = nwm.get_grid_stats(grid_path, shp_file, polyids, shp_header)

t_stop = time.time()
print('Elapsed time this file', t_stop - t_start)   

##############################################################

def shp_to_crs(gdf, domain, **kwargs):

    # WKT strings extracted from NWM grids
    wkt = 'PROJCS["Sphere_Lambert_Conformal_Conic",GEOGCS["GCS_Sphere",DATUM["D_Sphere",SPHEROID["Sphere",6370000.0,0.0]],\
    PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["false_easting",0.0],\
    PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-97.0],PARAMETER["standard_parallel_1",30.0],\
    PARAMETER["standard_parallel_2",60.0],PARAMETER["latitude_of_origin",40.0000076294],UNIT["Meter",1.0]]'
    
    wkt_hi = 'PROJCS["Lambert_Conformal_Conic",GEOGCS["GCS_Sphere",DATUM["D_Sphere",SPHEROID["Sphere",6370000.0,0.0]],\
    PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["false_easting",0.0],\
    PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-157.42],PARAMETER["standard_parallel_1",10.0],\
    PARAMETER["standard_parallel_2",30.0],PARAMETER["latitude_of_origin",20.6],UNIT["Meter",1.0]]'

    if domain == 'hawaii':
        crs_name = 'Lambert_Conformal_Conic'
        wkt = wkt_hi
    else:
        crs_name = 'Sphere_Lambert_Conformal_Conic'
       
    check_crs = gdf.crs
    print(check_crs.name)
    
    if check_crs is None:
        gdf = gdf.set_crs(epsg=4269)
        
    elif check_crs.name == crs_name:
        print('Shapefile already in correct projection')
        return gdf, False
   
    # if a grid was passed in as an arg with key "nwm_grid", get the wkt from the grid attributes
    # this slows the function, down, so only use if projection is different from the wkt's above
    if kwargs:
        for key, value in kwargs.items(): 
            if key == 'nwm_grid':
                try:
                    ds = xr.open_dataset(value)
                    attrs = ds['ProjectionCoordinateSystem'].attrs
                    wkt = attrs['esri_pe_string']
                except OSError:
                    print('Cannot open file: ', value, 'to get projection, using default nwm_grid WKT')
            else:
                print('only ""nwm_grid"" supported so far, using default nwm_grid WKT instead')
                
    nwm_crs = CRS.from_string(wkt)
    gdf_reproj = gdf.to_crs(nwm_crs)
    print("Reprojected shapefile to ", gdf_reproj.crs.name)
        
    return gdf_reproj, True
    

def affine_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def get_grid_stats(grid_path, shp_file, polyids, shp_header):

    try:
        ds_sr = xr.open_dataset(grid_path)
    except OSError:
        print('Cannot open file: ', grid_path)
        return pd.DataFrame()
        
    rain = ds_sr["RAINRATE"].values
    lat = ds_sr["y"].values
    lon = ds_sr["x"].values

    rain_xy = rain[0,:,:]

    lat_flip = np.flipud(lat)
    rain_flip = np.flipud(rain_xy)

    trans = Affine.translation(lon[0], lat_flip[0])
    scale = Affine.scale(lon[1] - lon[0], lat_flip[1] - lat_flip[0])    
    transform = trans * scale

    zstats = zonal_stats(shp_file, rain_flip, affine=transform, nodata = -999, stats="count mean")

    df_zstats = pd.DataFrame(zstats)
    df_zstats[shp_header] = polyids 
    df_zstats = df_zstats.set_index(shp_header)

    return df_zstats