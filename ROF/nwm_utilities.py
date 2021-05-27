''' 
    Collection of functions for working with and evaluating NWM
'''

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import geopandas as gpd
from rasterstats import zonal_stats
from affine import Affine
from pyproj import CRS
import requests
import os
import sys
import time
from datetime import datetime, timedelta
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
from shapely.geometry import Point
import math
import shapely.geometry as geometry
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path


##################################################################################
# Evaluation functions
##################################################################################


#def rof_eval_with_download(domain, version, nwm_repo, out_dir, aux_dir, shp_dir, 
def rof_eval_with_download(domain, nwm_repo, in_dir, out_dir,
                           eval_config, verif_config, eval_timing,
                           metric, spatial_agg_method, event_thresh, order_max,
                           use_existing = True, all_touched = False, 
                           **kwargs):

    ###############################################################################################
    #  Get list of references times for the evaluation
    ###############################################################################################

    # get the list of reference times 
    if eval_timing == 'current':
        ref_time_list, version_list = reftimes(domain, eval_config, verif_config, eval_timing)
    else:
        if kwargs:
            for key, value in kwargs.items(): 
                if key == 'start':
                    start_time = value
                elif key == 'end':
                    end_time = value
                else:
                    raise KeyError(f"Error: incorrect keyword {key}")
            
        ref_time_list, version_list = reftimes(domain, eval_config, verif_config, eval_timing,
                                               start = start_time, end = end_time)

    ###############################################################################################
    #  Get filelists (alter to parse out specs as needed for DSTOR or WRDS)
    ###############################################################################################

    #  evaluation variables and configurations to loop through
    var_list = ['forcing','channel']
    config_list = [eval_config, verif_config]    

    # store filelists for later calculation steps
    df_filelists = pd.DataFrame()

    ind = 0
    for i, ref_time in enumerate(ref_time_list):

        version = version_list[i]

        if version == 2.1:
            version_dir = nwm_repo / 'v2_1'
        else:
            version_dir = nwm_repo / 'v2_0'

        for config in config_list:
            for variable in var_list:
                print ("Building file list for:", ref_time, config, variable)

                filelist = build_filelist(ref_time, 
                                              version, 
                                              domain, 
                                              variable, 
                                              config, 
                                              eval_config, 
                                              verif_config)

                # append the list to dataframe
                df_filelists = df_filelists.append(pd.DataFrame({'ref_time' : ref_time,
                                                                 'config' : config, 
                                                                 'variable' : variable, 
                                                                 'filelist' : [filelist]}, 
                                                                  index = [ind]))
                ind += 1
                
    ###############################################################################################
    #  Download netcdf files needed for the evaluation (skip if data already on local source)
    ###############################################################################################

    t_download_start = time.time()   

    # store file status (in cache or failed) for check post-download function
    # do not exit program mid-download due to a missing file in order to continue download of any available files  
    df_in_cache = pd.DataFrame()

    for index, row in df_filelists.iterrows():

        print("\nDownloading nwm output from Google for: ", row['variable'], row['config'], row['ref_time'])
        file_in_cache = download_nwm_from_google(version_dir, row['filelist'])

        df_in_cache = df_in_cache.append(pd.DataFrame({'in_cache' : [file_in_cache]}, 
                                                        index = [index]))

    t_download_end = time.time()
    print("\n---Download complete - Total download time", (t_download_end - t_download_start)/60, " min.")

    df_filelists = df_filelists.join(df_in_cache)


    ###############################################################################################
    #  Begin reference time loop 
    ###############################################################################################

    print("\nReading static huc10 and feature information for domain: ", domain)
    
    # get static feature and huc10 info 
    df_featinfo, df_gages, df_thresh, df_length = read_feature_info(domain, 
                                                                    version,
                                                                    in_dir)   

    df_hucinfo, gdf_huc10_dd, gdf_states = read_huc10_info(domain, in_dir)

    # initialize some stuff
    # subset of nwm output filelists for channel output and forcing output
    df_filelists_channel = df_filelists[df_filelists['variable'] == 'channel']
    df_filelists_forcing = df_filelists[df_filelists['variable'] == 'forcing']
    dict_metrics = {}
    dict_map = {}
    
    # check that all channel_rt output files needed already existing or were successfully downloaded
    # if any channel_rt output is missing - raise exception, exit program
    # if forcing grids are missing, allow ROF calculations to continue, MAP plots will be left blank
    file_check(df_filelists_channel)    

    t_eval_start = time.time() 

    # begin loop
    for i, ref_time in enumerate(ref_time_list):
    
        print ("-------------Beginning Evaluation for Ref_time:", ref_time,"-------------")
        
        version = version_list[i]
        
        t_reftime_start = time.time() 
        
        ##############################################################################
        #  Build flow arrays and calculate metrics (including ROF)
        ##############################################################################
        
        t_read_start = time.time()
        
        print("\nReading nwm output for reference time: ", ref_time)
        
        # read flow data and calculate metrics for each config
        for config in config_list:
        
            print("\nBuilding flow arrays for config: ", config)

            filelist = df_filelists_channel.loc[(df_filelists_channel['ref_time'] == ref_time) & 
                                                (df_filelists_channel['config'] == config), 'filelist'].iloc[0]

            # build dataframe of flow and nudge arrays by reach, timesteps T0 to TN (if forecast, nudge will be empty)
            df_flow, df_nudge, success = build_flow_array(filelist, 
                                                          version_dir, 
                                                          feat_list = [], 
                                                          is_forecast = True)
            
            # TEMPORARY pending further testing
            # if any files missing, it should be caught ealier in 'file_check', left until further tested
            if not success:
                raise FileNotFoundError('NWM output missing, ROF calculations failed')
            
            # build dataframe of flow metrics by reach (including ROF criteria, BAT, others)
            df_metrics, df_flow_sub = get_flow_metrics(df_flow, 
                                                       df_thresh)   
            
            # store metric dataframes by config in a dictionary
            dict_metrics[config] = df_metrics
            
        t_read_end = time.time()
        print("\n---Flow output processing time", (t_read_end - t_read_start), "sec")
            
        ##############################################################################
        #  Compare a specified metric by reach
        ##############################################################################    

        # metric for evaluation defined in header section        
        
        t_metrics_start = time.time()
                
        print("\nSummarizing evaluation metrics by reach for", config_list)
        df_reach = merge_reach_metrics(df_featinfo, 
                                       dict_metrics, 
                                       config_list, 
                                       metric)
                                           
        if order_max > 0:
            df_reach_sub = df_reach[df_reach['order'] <= order_max]
        else:
            df_reach_sub = df_reach
        
        ##############################################################################
        #  Compare aggregated spatial metric by HUC
        ##############################################################################    
        
        print("Summarizing evalution metrics by HUC10 for", config_list)    
        df_huc = sum_huc10_metrics(df_hucinfo, 
                                   df_reach_sub, 
                                   config_list, 
                                   metric, 
                                   order_max = order_max)
        
        t_metrics_end = time.time()
        print("\n---Metric calculation time", (t_metrics_end - t_metrics_start), "sec")
        
        
        ##############################################################################
        #  Identify objects, aka 'regions of interest' (spatial clusters)
        ##############################################################################
        
        t_obj_start = time.time()
        
        print("\nResolving objects (clusters of hucs)")
        gdf_bounds, gdf_region_hucs, df_region_hucs = resolve_objects(
                                                        df_huc,
                                                        gdf_huc10_dd, 
                                                        config_list, metric, 
                                                        spatial_agg_method,
                                                        conv_radius = 1, 
                                                        mask_thresh = 1, 
                                                        gap_thresh = 1, 
                                                        buffer_radius = 0.25)
        
        if df_region_hucs.empty:
            print('No HUCs meet the criteria, skipping this reference time: ', ref_time)
            fig_title, fig_path = rof_fig_text(version, domain, verif_config, ref_time, out_dir, order_max, spatial_agg_method)
            empty_fig(fig_path, fig_title, gdf_states)
            continue
        
        # get subset of reaches in the region and associated object number
        df_region_reaches = df_reach.loc[df_reach['HUC10'].isin(df_region_hucs.index), df_reach.columns]
        df_region_reaches = pd.merge(df_region_reaches, df_region_hucs['obj'], right_index = True, left_on = "HUC10", how = "left")
        
        ##############################################################################
        #  Calculate evaluation stats (by reach and huc for each object)
        ##############################################################################
        
        print("\nCalcuating evalution statistics for each object")

        # reach contingency matrix categories and stats by object
        df_reach_eval, df_stats_reach = eval_stats_reach(df_region_reaches, 
                                                             config_list, 
                                                             metric)
        
        # huc contingency matrix for defined event threshold (event threshold defined at top)
        df_huc_eval, df_stats_huc, col_heads = eval_stats_huc(df_region_hucs, 
                                                                  config_list, 
                                                                  metric,
                                                                  spatial_agg_method = spatial_agg_method,
                                                                  event_thresh = event_thresh)                              
                  
        t_obj_end = time.time()
        print("\n---Object processing time", (t_obj_end - t_obj_start), "sec")

        
        ##############################################################################
        #  Calculate MAPs for HUCs in the regions
        ##############################################################################
        
        print("\nProcessing mean areal precipitation for reference time:", ref_time)
        
        t_map_start = time.time()
        
        print('\nProjecting geometry for MAP calculations')
        gdf_proj, is_reproj = geom_to_crs(gdf_region_hucs, domain)
        
        # calculate MAPs for each config
        map_success = pd.Series([True, True], index = config_list)
        for config in config_list:
        
            print("\nCalculating mean areal precip for config: ", config, ", reference time: ", ref_time)

            filelist = df_filelists_forcing.loc[(df_filelists_forcing['ref_time'] == ref_time) & 
                                                (df_filelists_forcing['config'] == config),'filelist'].iloc[0]
            
            # get map timeseries for timesteps 1-18 for the hucs within the region,
            # (do not include timestep 0 from the filelist)
            # data are also currently written to a csv file - will update to SQLite 
            df_map = huc_mean_areal_precip(version_dir, 
                                       ref_time,
                                       config,
                                       filelist[1:], 
                                       metric, 
                                       gdf_proj, 
                                       shp_tag = "eval_reg", 
                                       use_existing = use_existing, 
                                       all_touched = all_touched)
                                                                       
            
            # store MAP dataframes in a dictionary
            dict_map[config] = df_map
            
        t_map_end = time.time()
        total_time = t_map_end - t_map_start
        print("\n---MAP processing time", total_time, "sec (",total_time/60,"min)")
        
        t_reftime_end = time.time() 
        total_time = t_reftime_end - t_reftime_start
        print("\n---Ref time", ref_time, " total processing time", total_time, "sec (",total_time/60,"min)")
        
        #############################################################################
        #  Create and save graphics
        #############################################################################
          
        print("\nGenerating graphics and output files:", ref_time)
        
        gdf_region_hucs_eval = region_shapefile(dict_map,                                                                
                                                df_huc_eval, 
                                                gdf_huc10_dd, 
                                                eval_config, 
                                                verif_config, 
                                                col_heads)    
        
        fig_title, fig_path = rof_fig_text(version, domain, verif_config, ref_time, out_dir, order_max, spatial_agg_method)
         
        nine_panel_conus(df_stats_huc, 
                        gdf_region_hucs_eval,
                        gdf_bounds,
                        df_reach_sub,
                        event_thresh, 
                        eval_config,
                        verif_config,
                        fig_path,
                        fig_title,
                        gdf_states)
        
        # write output for this reference time
        # shapefile containing eval results
        write_output(ref_time, 
                     out_dir, 
                     domain,
                     gdf_region_hucs_eval, 
                     df_gages, 
                     gdf_bounds)
            
    t_eval_end = time.time()
    total_time = t_eval_end - t_eval_start
    print("\n---Evaluation total time", total_time, "sec (",total_time/60,"min)")
    print("\n")   



##################################################################################
# date and filelist functions
##################################################################################

def reftimes(domain, eval_config, verif_config, eval_timing, **kwargs):

    # check for start/end time and get version
    if eval_timing == 'current':    
        # get current clock utc time (top of the last hour) 
        clock_ztime = datetime.utcnow().replace(second=0, microsecond=0, minute=0)
        version = nwm_version(clock_ztime - timedelta(hours=2))
        
    else:
        if kwargs:
            for key, value in kwargs.items(): 
                if key == 'start':
                    start_reftime = value
                elif key == 'end':
                    end_reftime = value
                else:
                    raise KeyError(f"Error: incorrect keyword {key}")
        else:
            raise KeyError("Error: No start/end times defined for 'past' type evaluation")
                    
                    
        version = nwm_version(start_reftime)
        version_end = nwm_version(end_reftime)
    
        if version != version_end:
            raise ValueError('Date range spans different NWM versions - not yet supported')
         
    # get some needed specs for the eval_config
    df_config = config_specs(eval_config, domain, version)
    duration = df_config.loc[eval_config,"duration_hrs"].item()
    runs_per_day = df_config.loc[eval_config,"runs_per_day"].item()

    if eval_timing == 'current':
    
        if verif_config == 'latest_ana':
            # to update every hour (evaluate most recent possible forecast)
            # get the best available AnA - will be a mix of extended and standard, 
            # depending on clock time   
        
            # evaluate the most recent-possible reftime (clock time - 2 - duration) 
            # (e.g. 18 hrs of the forecast plus 2 hrs due to latency)
            last_reftime = clock_ztime - timedelta(hours=2)
            eval_reftime = last_reftime - timedelta(hours=duration)
            start_reftime = eval_reftime
            end_reftime = eval_reftime
        
        elif verif_config == 'analysis_assim_extend':        
            # to update as often as possible but using only extended AnA, 
            # get current clock time and check the current hour, 
            # if after 19z clock time, the 16z extended AnA for current date should be available.
            # 
            #   Once today's extended AnA becomes available, any forecast that includes a valid time
            #   between 17z yesterday to 16z today can be now be evaluated.  This includes forecasts
            #   with reference times beginning 23z two days ago (last timestep valid time is 17z yesterday)
            #   through 22z yesterday (last time step is 16z today)
            #   Also rerun the 4 prior reftimes given updated Stage IV in overlapping tm27-24
            # In total, run evaluations for clock time T-48 to T-21 hours
            
            if clock_ztime.hour >= 19:
                # current extAnA available
                last_reftime = clock_ztime.replace(hour=16)
            else:
                # use yesterdays extana
                last_reftime = clock_ztime - timedelta(days=1)
                last_reftime = last_reftime.replace(hour=16)
                
            start_reftime = last_reftime - timedelta(hours=duration + 27)
            end_reftime = last_reftime - timedelta(hours=duration)

    # do some more checks if this hour is available - varies by domain and config
    # then build the list
    ref_time_list = build_reftime_list(start_reftime, end_reftime, domain, version, eval_config, runs_per_day)

    # if ref_time_list returns empty - no forecasts were run on the specified reference time
    # e.g. Hawaii runs only every 6 hours (v2.0)
    if not ref_time_list:
        raise ValueError(f"No short_range reference times are available to evaluate for datetime: {eval_reftime}")
    else:
        if eval_timing == 'past':
            print('Evaluation reference times: ', start_reftime, ' through ',end_reftime)
        else:
            print('\nCurrent UTC time: ', clock_ztime)
            print('Last posted reference time: ', last_reftime)
            print('Evaluation reference times: ', start_reftime, ' through ',end_reftime, '\n')
    
    # get version corresponding to the reference time - 
    # currently v2.0 assumed for all datetimes prior to 4-20-2021 13z
    # and v2.1 begins 4-20 at 14z 
    version_list = []
    for ref_time in ref_time_list:
        version_list.append(nwm_version(ref_time))
    
    return ref_time_list, version_list
    
    
def nwm_version(ref_time):
    
    v21_date = datetime(2021, 4, 20, 14, 0, 0)
    if ref_time < v21_date:
        version = 2.0
    else:
        version = 2.1
        
    return version
    
    
def build_reftime_list(start_reftime, end_reftime, domain, version, eval_config, runs_per_day):
    '''
    get a list of forecast reference times within date range based on configuration and domain
    since not all configs/domains (med-term, hawaii) are run every hour
    '''
    
    # first do some checks
    if end_reftime < start_reftime:
        raise ValueError('Start date after end date')
        
    if domain == 'hawaii' and eval_config == 'medium_range':
        raise ValueError('Domain and configuration not compatible') 
        
    if version < 2.0 or version > 2.1:
        raise ValueError('Version must be 2.0 or 2.1')

    # the reference time interval
    # *Note currently in all cases the NWM always runs at fixed intervals beginning at 00z
    # if this changes, will need to update code
    interval = 24/runs_per_day
    
    # if runs_per_day is < 24, check that the starting reftime falls on an hour that exists
    # if not, shift the start time forward to the first existing reference time
    if runs_per_day < 24:
        offset = start_reftime.hour % interval
        if offset > 0:
            shift_forward = interval - start_reftime.hour % interval
            start_reftime = start_reftime + timedelta(hours=shift_forward)
            # recheck that start is before the end, if not return empty list
            if end_reftime < start_reftime:
                return []
                
    # create the list of reference times at the correct interval 
    ref_time_list = pd.date_range(start=start_reftime, end=end_reftime, freq= str(interval) + 'H').to_list()   
        
    return ref_time_list
    
    
def config_specs(config, domain, version):

    if config == 'latest_ana':
        config = 'analysis_assim'

    # base dataframe of config info for all configs
    # note that for medium range, assume only evaluating 'member 1'
    df_config_specs = pd.DataFrame(
        {"dir_suffix" : ["", "_mem1", "", ""],
         "var_str_suffix" : ["","_1","",""],
         "duration_hrs" : [18, 240, 3, 28], 
         "timestep_int" : [1, 3, 1, 1], 
         "runs_per_day" : [24, 4, 24, 24],
         "is_forecast" : [True, True, False, False],
         "abbrev" : ['srf','mrf','stana','exana']},
        index = ["short_range", "medium_range", "analysis_assim", "analysis_assim_extend"])
        
    # adjustments to base info for hawaii domain and version 
    #(for v2.1 will add other domain specifics, PR, AK)
    if domain == 'hawaii':
             
        # hawaii v2.0-2.1 only has 'short-range' and standard AnA
        df_config_specs = df_config_specs["short_range", "analysis_assim"]
        
        # add domain suffix to the directory name for both configs
        df_config_specs.loc[:,'dir_suffix'] = [domain, domain]        
        
        # hawaii short range extends out 60 hours in 2.0 and 48 hours in 2.1
        # hawaii timesteps changes from 1 hour in 2.0 to 15 min in 2.1 (both SRF and AnA)
        # hawaii run interval changes from 6 hours (4 x day) in 2.0 to 12 hours (2 x day) in 2.1
        if version == 2.0:
            df_config_specs.loc['short_range','duration_hrs'] = 60
            df_config_specs.loc['short_range','runs_per_day'] = 4
        elif version == 2.1:
            df_config_specs.loc['short_range','duration_hrs'] = 48  
            df_config_specs.loc[:,'timestep_int'] = [0.25, 0.25]
            df_config_specs.loc['short_range','runs_per_day'] = 2     
            
    df_config_specs = df_config_specs.loc[[config]]
    
    return df_config_specs
    
    
      
def variable_specs(domain):
        
    # build dataframe of variable group info and processing flags
    df_var_specs = pd.DataFrame(
            {"dir_prefix" : ["forcing_", ""], 
             "use_suffix" : [False, True], 
             "var_string" : ["forcing", "channel_rt"],
             "var_out_units" : ["mm hr-1","cms"]},
            index = ["forcing", "channel"])
 
    # adjustments to base info for hawaii domain and version 
    #(for v2.1 will add other domain specifics, PR, AK)
    if domain == 'hawaii':
        
        # turn on flag to add domain suffix (defined in config specs)
        # for both variables (in conus used for medium term "mem1" suffix, 
        # med term does not exist for Hawaii and suffix instead indicates "hawaii"
        df_var_specs['use_suffix'] = [True, True]                          

    return df_var_specs
    


def build_filelist(ref_time, version, domain, variable, config, eval_config, verif_config):

    # if config = 'latest_ana', will be using a mix of std and ext ana, start with standard
    if config == 'latest_ana':
        config = 'analysis_assim'
    
    # for std AnA, flag to use tm02 rather than tm00 when piecing together time series
    # (was an argument, but hard-coding for now until becomes apparent needs to be an argument)
    use_tm02 = True  

    # get base dataframe of config and variable info
    df_config = config_specs(config, domain, version)
    df_var = variable_specs(domain)

    # base configuration directory prefix (e.g. 'forcing')
    dir_prefix = df_var.loc[variable, 'dir_prefix']
    
    # variable string used in filename ('forcing' or 'channel_rt' for now)
    var_string = df_var.loc[variable, 'var_string']    

    # base configuration directory suffix (e.g. 'mem1' for medium_range ensemble member 1)    
    dir_suffix = df_config.loc[config, 'dir_suffix']

    # suffix at the end of the variable name (e.g. '1' for medium_range ensemble mem1)
    var_str_suffix = df_config.loc[config, 'var_str_suffix']

    # flag to trigger using suffixes or not (for forcing or hawaii)
    use_suffix = df_var.loc[variable, 'use_suffix']   
    
    # get duration, time interval and whether it is a forecast 
    # (flag for "f" versus "tm")
    n_hours = df_config.loc[config,"duration_hrs"].item()
    ts_int = df_config.loc[config,"timestep_int"]
    is_forecast = df_config.loc[config,"is_forecast"]
    
    # check and flag if this is a forecast evaluation
    # (needed to indicate how AnA data are stitched together)
    # if True set AnA duration (n_hours) equal to forecast duration
    fcst_eval = False
    if eval_config in ['short_range','medium_range']:
        fcst_eval = True
        # if current config is an AnA, set duration to forecast duration
        if not is_forecast:
            n_hours = config_specs(eval_config, domain, version).loc[eval_config,"duration_hrs"].item()
            #df_fcst = config_specs(eval_config, domain, version)
            #n_hours = df_fcst.loc[config,"duration_hrs"].item()  
        
    # initialize some filename parts
    datedir = ref_time.strftime("nwm.%Y%m%d")  # ref date directory   
    
    # set-up and initialize dataframe of varying filename parts
    # if building filenames for a forecast eval, include T0 
    #       for forecast configs always use stdana T0
    #       for AnA configs being compared to forecast configs, source of T0 varies
    
    n_files = n_hours
    if fcst_eval:
        n_files = n_hours + 1
    
    df_parts = pd.DataFrame(
            {"datedir" : np.full(n_files, datedir),
             "ref_hr_string" : np.full(n_files, 't00z'),
             "config" : np.full(n_files, config),
             "ts_hr_string" : np.full(n_files, 'tm00')},
             index = np.arange(n_files))
    
    #### Get varying filename parts for specific config/case ####
    
    # Case 1 - get timesteps that are run at the ref-time (fcst or AnA)
    # FORECASTS ARE ALWAYS THIS CASE
    # only used for AnA if evaluating a historical sim (i.e., AnA comp to usgs) 
    if is_forecast or not fcst_eval:
        #print('case 1')
        df_parts = get_reftime_fileparts(ref_time, n_hours, df_parts, ts_int,
                                          is_forecast, use_tm02)    
                                         
    # Case 2 - use the best available AnA at the time of the specified ref-time
    # results in a mix of standard or extended AnA, depending on hour of day
    # NOTE  this function starts from end_time and works backwards to n_hours prior
    #######***** get_realtime_ana_pathlist is not yet tested/altered for hawaii or med-range #########                                        
    #elif not is_forecast and (eval_timing == 'hourly'):                               
    elif not is_forecast and (verif_config == 'latest_ana'):     
        #print('case 2')
        end_time = ref_time + timedelta(hours=n_hours)
        df_parts = get_realtime_ana_fileparts(end_time, n_hours, df_parts)
    
    # Case 3 - A single configuration of AnA (extended or standard) is being used
    # to compare to a forecast - pull AnA for datetimes that correspond to forecast valid times   
    else:
        #print('case 3')
        df_parts = get_fcsteval_ana_fileparts(ref_time, config, n_hours, df_parts, ts_int, use_tm02)
 
    # loop through fileparts dataframe, add prefix/suffixes, and build filenames
    filelist = []
    for index, row in df_parts.iterrows():
        
        config_dir = dir_prefix + row['config']
        if use_suffix:
            config_dir = config_dir + dir_suffix
            var_string = var_string + var_str_suffix
            
        # build the filename
        filename_parts = ['nwm',
                           row['ref_hr_string'], 
                           row['config'],
                           var_string,
                           row['ts_hr_string'], 
                           domain,
                          'nc']
                  
        filename = ".".join(filename_parts)
        
        # add the full path to the list
        
        filelist.append(Path(row['datedir']) / config_dir / filename)
 
    return filelist
    
    
    
def get_reftime_fileparts(ref_time, n_hours, df_parts, ts_int, is_forecast, use_tm02):
    '''
    straightforward filenames for all timesteps of a given ref_time and configuration
    '''   
    
    for i in range(0, n_hours + 1, ts_int): # (0 through n_hours, incrementing by ts_int)

        df_parts.loc[i,'datedir'] = ref_time.strftime("nwm.%Y%m%d")
        df_parts.loc[i,'ref_hr_string'] = 't' + ref_time.strftime("%Hz")     
        
        ts_hr = i

        # fill in ts_hr_string, and use std AnA for T0 for forecast configs
        
        if is_forecast:
            if i == 0:
                # T0 comes from standard AnA tm00
                df_parts.loc[i,'config'] = 'analysis_assim'
                df_parts.loc[i,'ts_hr_string'] = 'tm00'
            else:
                df_parts.loc[i,'ts_hr_string'] = "f" + str(ts_hr).zfill(3)
        
        # if AnA configs
        else:         
            if use_tm02 and i < 2:
             # if i is 0 or 1, and only want tm02, skip this iteration
                continue

            df_parts.loc[i,'ts_hr_string'] = "tm" + str(ts_hr).zfill(2)          

    return df_parts        
    
            
            
def get_fcsteval_ana_fileparts(ref_time, config, n_hours, df_parts, ts_int, use_tm02):

    # AnA used for forecast evaluations:
    #   --> pull AnA output that corresponds to valid times of e.g., SRF f001-f018 (or f060, f048)

    for i in range(0, n_hours + 1, ts_int): # (0 through n_hours, incrementing by ts_int)
    
        #include T0
               
        #build filenames to get AnA output that corresponds to valid times of the forecast
        ts_hr = i
        val_time = ref_time + timedelta(hours=ts_hr) # calendar date/time of forecast timestep
        
        #get current clock time to check if "next day AnA" is available yet
        clock_ztime = datetime.utcnow().replace(second=0, microsecond=0, minute=0)
       
        is_today = False
        if val_time.date() == clock_ztime.date():
            is_today = True

        if config == "analysis_assim_extend":
            #e-AnA only runs in 16z cycle, get all output from this ref-time, either current or next date
            ref_hr_string = '16z'  
            
            #Valid hours 0-12 --> align with tm16-tm04 in same date directory
            if val_time.hour < 13:  
                datedir = val_time.strftime("nwm.%Y%m%d")
                ts_hr = 16 - val_time.hour            
                
            #Valid hours 13-23 --> align with tm27-tm17 in next date directory    
            else:              
                nextday = (val_time + timedelta(days=1)).replace(second=0, microsecond=0, minute=0)
                datedir = (nextday).strftime("nwm.%Y%m%d")
                ts_hr = 40 - val_time.hour
                
                #get current clock time to check if "next day AnA" is available yet
                clock_ztime = datetime.utcnow().replace(second=0, microsecond=0, minute=0)
                
                if nextday.replace(hour=19) > clock_ztime:
                    datedir = val_time.strftime("nwm.%Y%m%d")
                    ts_hr = 16 - val_time.hour
                
                
        else:
            #standard AnA runs every cycle, if get_tm02 = True, use tm02 from ref_time + 2, else use tm00 from ref-time
            if use_tm02:
                datedir = (val_time + timedelta(hours=2)).strftime("nwm.%Y%m%d")
                ref_hr_string = (val_time + timedelta(hours=2)).strftime("%Hz")
                ts_hr = 2
            else:
                datedir = val_time.strftime("nwm.%Y%m%d")
                ref_hr_string = val_time.strftime("%Hz")
                ts_hr = 0
        
        df_parts.loc[i,'datedir'] = datedir
        df_parts.loc[i,'ref_hr_string'] = 't' + ref_hr_string    
        df_parts.loc[i,'ts_hr_string'] = 'tm' + str(ts_hr).zfill(2)

    return df_parts
    


def get_realtime_ana_fileparts(end_time, n_hours_back, df_parts):
    '''
    determining which AnA configuration (standard or extended) are available to use to piece
    together for each valid time in the most recent evaluatable forecast (e.g. issued 20 hours earlier)
    Revamp this eventually - may be better/easier way to do this
    
    *note # of filenames created is n_hours_back + 1 (includes end_time)
    '''

    start_time = end_time - timedelta(hours=n_hours_back)

    # get the valid times (time on the ground) between start_time/end_time
    # e.g. for comparison to SRF issued 18-hrs prior, get ref_time (18 hours ago) plus 18 hours
    val_times = pd.date_range(start=start_time, end=end_time, freq='H')      

    # get number of datetimes (number of filenames being generated)
    n_files = len(val_times)
    
    config = 'analysis_assim'    
    for i in np.arange(n_files-3,n_files):
        df_parts.loc[i,'datedir'] = end_time.strftime("nwm.%Y%m%d")
        df_parts.loc[i,'ref_hr_string'] = 't' + end_time.strftime("%Hz")   
        df_parts.loc[i,'ts_hr_string'] = 'tm' + str(n_files - i - 1).zfill(2)
        df_parts.loc[i,'config'] = config      
        
    # the algorithm below determines which timesteps have an extended AnA value available
    #       for the valid time and which have only standard available
    #       and determines which set of output (which date for extana) the valid time is part of

    # keep track of the timestep hour of extended to know when to switch dates
    ext_ts_hour = 0
    
    # starting with 4th to last timestep (prior to most recent std AnA run), work backwards
    for i in range(n_files-4,-1,-1):

        # ivt short for ith valid time
        ivt = val_times[i]
        vt_date = datetime(ivt.year, ivt.month, ivt.day, 0, 0, 0)
        vt_hour = ivt.hour
        
        # if the valid time hour hits 16, the last run ext-ana is available 
        if config == 'analysis_assim' and vt_hour == 16:
            config = 'analysis_assim_extend'
            ext_ts_hour = 0
               
        df_parts.loc[i,'config'] = config   
            
        # if the valid time is pulling from standard, use tm02 from ref time 2 hours ahead
        # currently hard-coded here to use tm02 only for all timesteps prior to most recent avail std AnA
        if config == 'analysis_assim':
            data_time = ivt + timedelta(hours = 2)
            std_ts_hour = 2
       
            df_parts.loc[i,'ts_hr_string'] = 'tm' + str(np.int(std_ts_hour)).zfill(2)
            
        # if pulling from an extended ana run, figure out which date and timestep
        else:
            if ext_ts_hour < 17:
                data_time = datetime(ivt.year, ivt.month, ivt.day, 16, 0, 0)
            else:
                data_time = datetime(ivt.year, ivt.month, ivt.day+1, 16, 0, 0)              
          
            df_parts.loc[i,'ts_hr_string'] = 'tm' + str(np.int(ext_ts_hour)).zfill(2)

            if ext_ts_hour == 27:
                ext_ts_hour = 4
            else:
                ext_ts_hour += 1    
                
        df_parts.loc[i,'datedir'] = data_time.strftime("nwm.%Y%m%d")
        df_parts.loc[i,'ref_hr_string'] = 't' + data_time.strftime("%Hz")   
                
    return df_parts
    
######################################################################################
# data access functions
######################################################################################
                    
def download_nwm_from_google(version_dir, filelist):

    file_in_cache = []
    
    for i, nwm_path in enumerate(filelist):
           
        # parse out nwm directories and filename from full_path
        datedir = nwm_path.parts[0]
        config_dir = nwm_path.parts[1]
        filename = nwm_path.parts[2]
        
        # rebuild with UTF-8 encoding for directory slash   
        slash = "%2F"
        encoded_nwm_path = datedir + slash + config_dir + slash + filename    

        # check if datedir exists
        create_dir_if_not_exist(version_dir / datedir)

        # build netcdf_dir
        netcdf_dir = version_dir / nwm_path.parent

        # check if output directory exists, if not create
        create_dir_if_not_exist(netcdf_dir)

        # build full output path
        full_path = netcdf_dir / filename        

        status = ""    
        
        # check if file exists in cache, if not, download it 
        print(full_path)
        if full_path.exists():            
            print("File already in cache: ", str(full_path))
            status = "in_cache"                   
        else:   
            t_start = time.time()                 
            status = get_from_google(encoded_nwm_path, full_path)
            t_stop = time.time()
            print('Elapsed time this file', t_stop - t_start)  
     
        if status == "failed":
            in_cache = False
        else:
            in_cache = True
                   
        file_in_cache.append(in_cache)
            
    return file_in_cache
    

def get_from_google(encoded_nwm_path, full_path):
    
    # build Google URL
    url_prefix = "https://storage.googleapis.com/download/storage/v1/b/national-water-model/o/"
    url_suffix = "?alt=media"
    url = url_prefix + encoded_nwm_path + url_suffix
    
    print("Fetching from Google Cloud @ URL " + url)

    response = requests.get(url)
    
    if response.status_code == 200:
        print("writing file to: ", full_path)
        with full_path.open("wb") as f:
            f.write(response.content)
        status = 'downloaded'
    else:
        print("Request failed: " + str(response.status_code) + " " + response.reason)
        status = "failed"
    
    return status
    
def file_check(df_filelists_channel):

    # check that all files needed for the evaluation already existing or were successfully downloaded
    # if any channel_rt output is missing - raise exception, exit program
    # if grids are missing, allow ROF calculations to continue 
    
    for index, row in df_filelists_channel.iterrows():

        filelist = row['filelist']
        in_cache = row['in_cache']
    
        for i, file in enumerate(filelist):
            
            if not in_cache[i]:
                raise OSError(f"NWM channel output file missing, cannot proceed, check download source: {file}")

    
def read_feature_info(domain, version, in_dir):

    if domain == 'hawaii':
        feature_file = "NWM_features_info_hawaii_2_0.csv"    
        q_label = '2_0Q'
    else:
        q_label = '1_5Q'
        if version == 2.1:
            feature_file = "NWM_features_info_conus_2_1.csv"
        else:
            feature_file = "NWM_features_info_conus_2_0.csv"

    feature_path = in_dir / feature_file

    # read feature info
    df_featinfo = pd.read_csv(feature_path)
    df_featinfo = df_featinfo.set_index('feature_id')

    df_thresh = df_featinfo[[q_label]]
    df_length = df_featinfo[['length_m']]
    
    # get subset of features with gages
    df_gages = df_featinfo[df_featinfo['gage'] > 0]
    
    return df_featinfo, df_gages, df_thresh, df_length
    
    
def read_huc10_info(domain, in_dir):

    huc_csv = "HUC10_eval_info.csv"

    if domain == 'hawaii':
        huc_shp_dd = "HUC10_Hawaii_dd.shp"
        states_shp_dd = "US_States_Hawaii.shp"
    else:
        huc_shp_dd = "HUC10_Simplified005_dd.shp"
        states_shp_dd = "US_States_CONUS.shp"
        
    #path = in_dir / huc_csv
        
    df_hucinfo = pd.read_csv(in_dir / huc_csv, index_col = 0)
    df_hucinfo = df_hucinfo.loc[:,['tot_feats','Centroid_Lat','Centroid_Lon']]
    
    # rename centroid lat and lon - needed for later functions
    df_hucinfo = df_hucinfo.rename(columns = {"Centroid_Lat" : "lat", 
                                              "Centroid_Lon" : "lon"})
                                                
    # read shapefiles - need projected for MAP processing and dd for graphics/maps
    gdf_huc_dd = gpd.read_file(in_dir / huc_shp_dd)
    gdf_huc_dd["HUC10"] = gdf_huc_dd["HUC10"].astype("int64")
    gdf_huc_dd = gdf_huc_dd.set_index('OBJECTID')
    
    # read states for figures
    gdf_states = gpd.read_file(in_dir / states_shp_dd)
    
    # add polygon centroids to geodataframe
    gdf_huc_dd = gdf_huc_dd.merge(df_hucinfo[['lat','lon']], how = 'left', left_on = 'HUC10', right_index = True)
      
    return df_hucinfo, gdf_huc_dd, gdf_states

######################################################################################
# data processing - read/process flow data
######################################################################################

    
def build_flow_array(filelist, version_dir, feat_list, is_forecast):
    '''
    Loop through filelist, build dataframe of flow for specified feature list
    '''
    
    # initialize dataframes
    df_flow = pd.DataFrame()
    df_nudge = pd.DataFrame()
    
    inds = False
    
    for ts, nwm_file in enumerate(filelist):

        print('Reading', str(nwm_file))
        
        path = version_dir / nwm_file

        #read flow, if is_forecast = True, ts_nudge will return empty
        ts_flow, ts_nudge, success = get_flow_from_path(version_dir, nwm_file, feat_list, is_forecast)           

        if success:
        
            # if first timestep, get list of indexes (feature ids) to confirm subsequent indexes 
            # are in the same order. (stored in nwm output files in same order) 
            # Only reindex if needed - creates big slow down.
            if not inds or ts == 0:
                inds = ts_flow.index 
                index_good = True
            else:
                index_good = (ts_flow.index == inds).all()

            # add the timestep columns to the dataframes, reindexing only if necessary
            if index_good:             
                df_flow[ts] = ts_flow
                df_nudge[ts] = ts_nudge
            else:
                df_flow[ts] = ts_flow[inds]
                df_nudge[ts] = ts_nudge[inds]

        else:
            #print('file read failed: ', str(nwm_file))
            raise OSError(f"NWM file not found {nwm_file}")            
            
    return df_flow, df_nudge, success
        
    
def get_flow_from_path(version_dir, path, feat_list, is_forecast):
    ''' 
    Read channel output and nudge (if app) for specified filename
    '''
    path = version_dir / path

    if path.exists():
        ds = xr.open_dataset(path, engine="netcdf4")
        success = True
        
        # extract flow to series, convert to cfs
        flow = ds["streamflow"].to_series() * 35.31466672  
        
        if is_forecast:
            # if forecast, fill nudge with zeros
            nudge = pd.Series(np.zeros(len(flow)), index = flow.index)
        else:
            # if not a forecast (is an AnA), read and convert nudge values
            nudge = ds["nudge"].to_series() * 35.31466672  
        
        if feat_list:
            flow = flow[feat_list]
            nudge = nudge[feat_list]
        
    else:
        success = False
        #print('file read failed: ', str(path))
        raise OSError(f"NWM file not found {path}")
        
        flow = []
        nudge = []
        
    return flow, nudge, success
    
######################################################################################
# data processing - metrics
######################################################################################
    
def flow_peak(flow, feats):
    '''
    Calculate basic peak flow metrics from a 2d numpy array
    where rows are reaches and columns are timesteps
    '''
    # magnitude of maximum flow in the period
    max_flow = np.amax(flow, axis = 1)
    
    # timestep of maximum flow in the period
    time_to_max = np.argmax(flow, axis=1)
    
    df_metrics = pd.DataFrame({'max_flow' : max_flow,
                               'time_to_max' : time_to_max},
                                index = feats)
                            
    return df_metrics
    
    
    
def flow_double(flow, feats):
    '''
    calculate percent change, 
    doubles - boolean matrix indicating which reaches and timesteps flow doubles
    any_doubles - boolean array indicating which reaches flow doubles at least once
    firstdouble - timestep number when doubling first occurred (or -999 if no doubling)
    '''
    
    # call percent change function
    perchg = percent_change(flow)
    
    # boolean 2d arrays indicating where/when 
    # percent change is > 100 
    flow_doubles = perchg > 100
    
    # flow doubles in any timestep
    any_doubles = np.any(flow_doubles, axis=1)
    
    # first occurences: timestep that doubling first occurs 
    first_double = np.argmax(flow_doubles, axis=1)    
    
    # where no doubling occurs, set to missing
    first_double[~any_doubles] = -999    
    
    df_metrics = pd.DataFrame({'any_doubles' : any_doubles,
                              'first_double' : first_double},
                               index = feats)
    
    return flow_doubles, df_metrics
 
 
def percent_change(flow):
    '''
    calculate percent change from one timestep to the next 
    e.g. T0 to T1, where the percent corresponds to T1
    (substitute 0.004 for 0 - per GID ROF code to avoid div by zero)
    matrices of flow shifted by 1 timestep to calculate percent change
    Resulting matrix will have n-1 columns as compared to original flow matrix
    '''
    
    nts = np.shape(flow)[1]
    
    prev = flow[:,0:nts-1]
    curr = flow[:,1:nts]
    perchg = 100 * (curr - prev) / np.where(prev == 0, .004, prev) 
    
    return perchg
    

def flow_exceed(flow, thresh, feats):
    '''
    test if threshold exceeded in any timestep - yes/no
         if threshold exceeded in all timesteps - yes/no
         number of timesteps exceeded
         if threshold exceeded in T0 - yes/no
         timestep of first exceedence
         timestep of last exceedence
    '''
    
    nts = np.shape(flow)[1]

    # tile threshold array to get matrix to allow faster matrix comparison for exceedence test
    mat_thresh = np.tile(np.transpose(thresh),(nts,1)).transpose()    

    # flow exceeds the defined threshold (e.g. 1.5 yr flow)
    exceed = flow > mat_thresh    # exceed includes T0 here

    # separate exceed matrix for T0 (initial states) and T1-TN (forecast timesteps)
    t0_exceed = exceed[:,0]
    t1tn_exceed = exceed[:,1:]

    # threshold exceeded in: all timesteps, any timestep, and 
    # number of timesteps exceeded
    all_exceed = np.all(t1tn_exceed, axis=1)
    any_exceed = np.any(t1tn_exceed, axis=1)
    nts_exceed = np.sum(t1tn_exceed, axis=1)

    # first occurences: timestep first exceeded and doubling first occurs 
    # where no timesteps exceeded, set to missing    
    first_exceed = np.argmax(t1tn_exceed, axis=1)
    first_exceed[~any_exceed] = -999

    # find last occurence above threshold ('flood end') by reversing
    # the exceed matrix and applying argmax, 
    # then substracting resulting index # from total number timesteps
    # to get the corresponding index number in the original matrix 
    reverse_exceed = np.fliplr(t1tn_exceed)
    reverse_tsmax = np.argmax(reverse_exceed, axis = 1)
    last_exceed = np.shape(t1tn_exceed)[1] - reverse_tsmax
    last_exceed[~any_exceed] = -999
    
    df_metrics = pd.DataFrame({'t0_exceed' : t0_exceed,
                               'any_exceed' : any_exceed,
                               'all_exceed' : all_exceed,
                               'nts_exceed' : nts_exceed,
                               'first_exceed' : first_exceed,
                               'last_exceed' : last_exceed},
                               index = feats)
                               
    return exceed, df_metrics
    
    
def rof_criteria(doubles, t1tn_exceed, feats):
    '''
    Find reaches meeting ROF criteria:
        1) percent change > 100
        2) exceeds threshold within 6 timesteps
    '''
    # get boolean mask indicating which features have a chance
    # i.e., at least one ts exceeds threshold and at least one doubles
    any_doubles = np.any(doubles, axis=1)
    any_exceed = np.any(t1tn_exceed, axis=1)
    mask = np.logical_and(any_exceed, any_doubles)
    
    # get first subset of exceed and doubling boolean 2d arrays for features in mask
    subset_exceed = t1tn_exceed[mask,:]
    subset_doubles = doubles[mask,:]
    subset_rof = np.zeros(np.shape(subset_exceed), dtype=bool)
    subset_feats = feats[mask]

    # number of timesteps
    nsteps = np.shape(subset_exceed)[1]

    # For each timestep, create a mask based on reaches where doubling occurred,
    # use the mask to extract those reaches from the subset_exceed for i to i+6 timesteps
    # anywhere in the mask that exceed is true, ROF is true.
    # Thus, overwrite the rof matrix (for the masked portion) with exceed matrix
    # The resulting rof matrix is a boolean matrix that indicates 'true' where both criteria were met

    for i in range(nsteps):
        # set the end timestep of the mask
        # and constain to last timestep of the matrices
        tsend = i+6
        if tsend > nsteps:
            tsend = nsteps
        
        # get the 'doubling mask' for this timestep, 
        # overwrite rof matrix with exceed matrix for the masked reaches/timesteps
        mask_i = subset_doubles[:,i]
        exceed_i = subset_exceed[mask_i,i:tsend]
        any_exceed_i = np.any(exceed_i, axis=1)
        subset_rof[mask_i,i:tsend] = exceed_i

    # for each reach, if the ROF matrix is true in any timestep
    # that reach meets the ROF criteria for the period
    any_rof = np.any(subset_rof, axis=1)
    n_rof = np.sum(any_rof)
    rof_true = subset_rof[any_rof]

    # remap back into the full feature list
    np_rof = np.zeros(np.shape(any_exceed), dtype=bool)
    np_rof[mask] = any_rof

    # return column dataframe of ROF result (true/false) for all reaches
    df_rof = pd.DataFrame({'rof' : np_rof}, index = feats)
    #df_rof = pd.DataFrame({'rof' : any_rof[any_rof]}, index = subset_feats[any_rof])

    return df_rof
    
    
def get_flow_metrics(df_flow, df_thresh):
    '''
    Call functions to calculate various summary flow metrics, including ROF criteria
    '''

    # Remove all features where the threshold = 0 or Missing
    df_thresh_sub = df_thresh[df_thresh.iloc[:,0] > 0]
    df_flow_sub = df_flow.loc[df_thresh_sub.index]

    # get numpy arrays from the dataframes to speed up calcs
    np_thresh = df_thresh_sub.iloc[:,0].to_numpy()
    np_flow = df_flow_sub.to_numpy()
    feats = df_flow_sub.index.to_numpy()

    # total number of timesteps, including T0
    nts = df_flow_sub.shape[1]

    # Basic Peak Metrics 
    df_peak_metrics = flow_peak(np_flow, feats)

    # Flow Doubling
    flow_doubles, df_double_metrics = flow_double(np_flow, feats)

    # Threshold Exceedence
    exceed, df_exceed_metrics = flow_exceed(np_flow, np_thresh, feats)

    # ROF Criteria
    t1tn_exceed = exceed[:,1:]
    df_rof = rof_criteria(flow_doubles, t1tn_exceed, feats)
    
    df_metrics = pd.concat([df_thresh_sub, df_peak_metrics, df_exceed_metrics, df_double_metrics, df_rof], axis = 1)
    df_flow_sub = pd.concat([df_thresh_sub, df_flow_sub])
    
    return df_metrics, df_flow_sub
    

def get_abbrev(config):
   
    if config == 'latest_ana':
        config = 'analysis_assim'
    
    abbrev = pd.Series(['srf','mrf','stana','exana'],
             index = ["short_range", "medium_range", "analysis_assim", "analysis_assim_extend"])
        
    return abbrev[config]
    

def get_column_headers(config_list, metric = "", suffix = ""):

    col = []
    
    for i, config in enumerate(config_list):
    
        abbrev = get_abbrev(config)
    
        if metric:
            abbrev = abbrev + "_" + metric
        if suffix:
            abbrev = abbrev + "_" + suffix
            
        col.append(abbrev)

    return col
        
    
def merge_reach_metrics(df_featinfo, dict_metrics, configs, metric):
    '''
    merge selected metric results from df_metric for the two configs
    into a single dataframe for evaluation
    
    metric must correspond to a column heading in df_metrics
    '''
    
    # extract the data columns for the defined config and metric
    # ecol (evaluation config header), vcol (verifying config header)
    ecol, vcol = get_column_headers(configs, metric)
    
    # extract results for eval_config (e.g. srf) and verif_config (e.g. ana)
    df_eval_metrics = dict_metrics[configs[0]].rename(columns = {metric : ecol})
    df_verif_metrics = dict_metrics[configs[1]].rename(columns = {metric : vcol})
   
    # get reach info to include in the comparison dataframe
    df_featinfo_sub = df_featinfo.loc[df_eval_metrics.index,['lat','lon','order','length_m','HUC10']]
    
    # merge eval and verif results into new dataframe
    df_reach = pd.concat([df_featinfo_sub, df_eval_metrics[ecol], df_verif_metrics[vcol]], axis = 1)
    
    return df_reach
      

def sum_huc10_metrics(df_hucinfo, df_reach, configs, metric, order_max = 4):
    '''

    '''
    # drop any hucs from main list that do not include reaches (tot_feats = 0)
    df_hucinfo = df_hucinfo[df_hucinfo['tot_feats'] > 0]
    
    # extract the data columns for the defined config and metric
    # ecol (evaluation config header), vcol (verifying config header)
    ecol, vcol = get_column_headers(configs, metric)

    # remove rows with missing values for eval_config or verif_config 
    # (could be missing for certain metrics, e.g. BAT if bankfull never occurs)
    df_reach = df_reach[(df_reach[ecol] >= 0) | (df_reach[vcol] >= 0)]

    # keep only reaches of order <= order_max
    # if value is zero or neg, skip - assume no limit
    
#    if order_max > 0:
#        df_reach = df_reach[df_reach['order'] <= order_max]
    
    # keep only reaches where huc is defined (non-missing)
    df_reach = df_reach[df_reach['HUC10'] > 0]

    # get total stream length of 'included' reaches in each HUC10 
    # (less than order max, defined HUC10, non-missing stat value)
    df_strlen = df_reach[['HUC10','length_m']].groupby('HUC10').sum()
    df_strlen = df_strlen.rename(columns = {'length_m' : 'tot_strlen'})
    df_hucinfo_sub = df_strlen.join(df_hucinfo, how = 'left').fillna(value = 0)

    # the data is boolean type, sum values by HUC10 to count num of reaches (True = 1)
    # tally total stream length of 'true' reaches by HUC10
    data_type = df_reach[ecol].dtype
    if data_type == 'bool':
    
        df_huc_temp = df_hucinfo_sub
        for col in [ecol, vcol]:
            df_huc_temp = sum_bool_by_huc(col, df_reach, df_huc_temp)
            
        df_huc = df_huc_temp
    
    # if it is a number, get mean for the HUC10
    elif data_type in ['int64','float64']:
        df_huc = df_reach[['HUC10',ecol, vcol]].groupby('HUC10').mean()
    
    else:
        raise TypeError('Cannot summarize data by HUC10 for data type', data_type)
        
    return df_huc
    
def sum_bool_by_huc(col, df_reach, df_huc_temp):
    
    # summarize by HUC10 - count # true, and sum reach lengths where true
    df = df_reach[df_reach[col]].groupby('HUC10').sum()[['length_m',col]]
    df = df.rename(columns = {'length_m' : col + '_len_m', col : col + '_count'})
    
    # merge with huc info
    df = df_huc_temp.join(df, how = 'outer').fillna(value = 0)

    # get percent of total number and percent of total stream length
    df[col + '_pernum'] = df[col + '_count']/df['tot_feats'] * 100
    df[col + '_perlen'] = df[col + '_len_m']/df['tot_strlen'] * 100
    
    # set NaN to zero resulting from div by zero
    df = df.fillna(value = 0)
    
    return df
    
    
def eval_stats_reach(df_reach, configs, metric, comp_method = 'matrix'):
    '''
    calculate evaluation statistics on reaches
    method options:
    'matrix' - contingency categories (data must be boolean)
    'diff' - straight difference (data must be integer or float)
    'perdiff' - percent difference (relative to obs) (data must be integer or float)
    add a call to WRES for others (?)
    '''

    # extract the data columns for the defined config and metric
    # ecol (evaluation config header), vcol (verifying config header)
    cols = get_column_headers(configs, metric)

    # get object list
    objects = list(set(df_reach['obj']))

    # IF data type is boolean - can apply contingency categories directly
    # assign contingency category - initialize w all true negatives (both False)
    # where both are True               -- true pos (hit)    
    # where obs = True and fcst = False -- false positive (false alarm)
    # where obs = False and fcst = True -- false negative (miss)    
    
    df_stats = pd.DataFrame()
    if comp_method == 'matrix':
    
        #confirm data type is already boolean
        if df_reach[cols[0]].dtype != 'bool':
            raise TypeError('data type must already be boolean for matrix method')
            
        for obj in objects:
            
            df_obj = df_reach.loc[df_reach['obj'] == obj, cols]
            inds_obj = df_obj.index

            # add integer to represent categories - facilitates plotting later
            # 1 = TP, 2 = FP, 3 = FN, 4 = TN
            df_matrix_obj, stats_obj = contingency(df_obj)
            
            # write values for this object to full dataframes
            df_reach.loc[inds_obj, 'matrix'] = df_matrix_obj
            
            df_stats = df_stats.append(pd.DataFrame([stats_obj.values], 
                                       columns = stats_obj.index, index = [obj]))
            
    #  Add difference and % difference later
    
    return df_reach, df_stats
    
    
    
def eval_stats_huc(df_huc, configs, metric, 
                   comp_method = 'matrix', spatial_agg_method = "str_length",
                   event_thresh = 30, thresh_lim = 'lower'):
    '''
    get evaluation statistics on aggregrated huc-scale metrics, like ROF %
    method options:
    
        'matrix' - contingency categories (data must be boolean or become boolean here)
                ** this method requires defining an dichotomous "event" 
                ** e.g. for ROF this is a threshold on the % of reaches/length
                ** that meet the criteria         
                  
        'diff' - straight difference (data must be integer or float)
    
    event_threshold - value to compare to evaluation metric (e.g. %)
                      to test if the huc is a True (yes event) or 
                      False (no event)
                      
    thresh_lim - apply the threshold as an 'upper' or 'lower' limit            
        
    '''
    
    # get column suffix for selected spatial metric
    # "length" - percent stream length, "count" - percent of the total number of reaches
    if spatial_agg_method == 'str_length':
        suffix = 'perlen'
    else:
        suffix = 'pernum'
        
    # generate column headers to read huc metrics dataframe, add suffix if app.
    cols_read = get_column_headers(configs, metric, suffix = suffix)
    
    # get object list
    objects = list(set(df_huc['obj']))
    
    df_stats = pd.DataFrame()
    if comp_method == 'matrix':
    
        # generate column headers for binary event, add suffix
        cols_write = get_column_headers(configs, suffix = 'event')
        
        # assign binary event based on threshold for all hucs
        for i, col_w in enumerate(cols_write):
            
            col_r = cols_read[i]
        
            if thresh_lim == "lower":
                df_huc.loc[df_huc[col_r] >= event_thresh, col_w] = True 
                df_huc.loc[df_huc[col_r] < event_thresh, col_w] = False 
            else:
                df_huc.loc[df_huc[col_r] > event_thresh, col_w] = False 
                df_huc.loc[df_huc[col_r] <= event_thresh, col_w] = True 
            
        for obj in objects:
            
            df_obj = df_huc.loc[df_huc['obj'] == obj, cols_write]
            inds_obj = df_obj.index

            # add integer to represent categories - facilitates plotting later
            # 1 = TP, 2 = FP, 3 = FN, 4 = TN
            df_matrix_obj, stats_obj = contingency(df_obj)
            
            # write values for this object to full dataframes
            df_huc.loc[inds_obj, 'matrix'] = df_matrix_obj
            
            df_stats = df_stats.append(pd.DataFrame([stats_obj.values], 
                                       columns = stats_obj.index, index = [obj]))     

            df_stats = df_stats.astype({'tot':'int','hits':'int','falarms':'int',
                                        'misses':'int','truenegs':'int'})                                       
                                       
    else:
        # later add other methods
        print('only binary evaluation available at the moment')
        
    cols_read.extend(cols_write)
        
    return df_huc, df_stats, cols_read
    
    
    
def contingency(df_data):
    '''
    given a dataframe of two boolean columns, where the 
    first (left) is the evaluation dataset and the second (right)
    is the verifying dataset, add a column containing the 
    contingency category and calculate POD, FAR, POFD and CSI
    '''
    # add column for contigency matrix categories, initialize with "TN"
    # using integer to represent categories - facilitates plotting later
    # 1 = TP, 2 = FP, 3 = FN, 4 = TN
    df_data['matrix'] = np.full(df_data.shape[0],4)
    df_data.loc[df_data.iloc[:,0] & df_data.iloc[:,1], 'matrix'] = 1
    df_data.loc[df_data.iloc[:,0] & ~df_data.iloc[:,1], 'matrix'] = 2
    df_data.loc[~df_data.iloc[:,0] & df_data.iloc[:,1], 'matrix'] = 3

    nhit = df_data[df_data['matrix'] == 1].count()[0]
    nfa = df_data[df_data['matrix'] == 2].count()[0]
    nmiss = df_data[df_data['matrix'] == 3].count()[0]
    ntn = df_data[df_data['matrix'] == 4].count()[0]
    tot = nhit + nfa + nmiss + ntn
        
    pod = nhit/(nhit + nmiss) if (nhit + nmiss) > 0 else 0
    far = nfa/(nhit + nfa) if (nhit + nfa) > 0 else 0
    csi = nhit/(nhit + nmiss + nfa) if (nhit + nmiss + nfa) > 0 else 0    
    pofd = nfa/(ntn + nfa) if (ntn + nfa) > 0 else 1

    statcols=['tot','hits','falarms','misses','truenegs','POD','FAR','CSI','POFD']
    stats = [tot, nhit, nfa, nmiss, ntn, pod, far, csi, pofd]

    d_stats = pd.Series(stats, index = statcols)
    
    return df_data['matrix'], d_stats
    
######################################################################################
# data processing - objects
######################################################################################

def resolve_objects(df_huc, gdf_huc, configs, metric = "rof", spatial_agg_method = "str_length",
                    conv_radius = 1, mask_thresh = 1, gap_thresh = 0.5, buffer_radius = 0.25):
    '''
    Using concepts from MODE, resolve 'objects' (regions of interest) across the domain
    
    '''
    # get column suffix for selected spatial metric
    # "length" - percent stream length, "count" - percent of the total number of reaches
    if spatial_agg_method == 'str_length':
        suffix = 'perlen'
        totcol = 'tot_strlen'
    else:
        suffix = 'pernum'
        totcol = 'tot_feats'
    
    # generate column headers, add suffix
    ecol, vcol = get_column_headers(configs, metric, suffix = suffix)
    
    # create a dataframe of nonzero hucs, add column that is the max value
    # for the reach (between evaluation config and verifying config),
    # resolve the objects based on the union and max values
    
    # extract data into numpy array for faster comparison
    evals = df_huc[ecol].to_numpy()
    vvals = df_huc[vcol].to_numpy()
    df_huc['max'] = np.where(evals > vvals, evals, vvals) 
    df_huc['diff'] = evals - vvals
    
    # reduce to only needed columns
    df_huc = df_huc.loc[:,['lat','lon', totcol, ecol, vcol, 'diff', 'max']]
    
    # get thresholded masks from union-max (1 = 'in mask', 0 = 'not in mask')
    # df_huc returns with added column 'mask'
    df_huc = get_object_mask(df_huc, conv_radius, mask_thresh)
        
    # extract 'in mask' hucs with lat/lons
    df_mask = df_huc.loc[df_huc['mask'] == 1,['lat','lon']]

    # split and assign object numbers to analyze separately based on separation distance gap_thresh
    df_objs_split = split_objects(df_mask, gap_thresh)
    objects = list(set(df_objs_split['obj']))
    max_obj = max(objects)
    
    # drop objects with fewer than X hucs over Y%
    df_objs_drop = drop_objects(df_objs_split, df_huc, min_cluster = 5, cluster_thresh = 10)    
    
    if df_objs_drop.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    # get the outer boundary around each remaining object, add a buffer
    print('Calculating and buffering object boundary(s)')
    gdf_bounds, gdf_obj_hucs = obj_bound_buff(df_objs_drop, gdf_huc, max_obj, buffer_radius = buffer_radius, alpha = 1.0)

    # add the rest of the info back to huc geodataframe
    gdf_region_hucs = gdf_obj_hucs.merge(df_huc[[totcol, ecol, vcol, 'max','diff']], 
                                        how = 'left', left_index = True, right_index = True).fillna(value = 0)
    df_region_hucs = pd.DataFrame(gdf_region_hucs.drop(columns='geometry'))
    
    return gdf_bounds, gdf_region_hucs, df_region_hucs #df_region, df_objs_split
    
    
def get_object_mask(points, R, T):
    '''
    simple circular filter (moving average) to smooth a datafield
    by calculating the mean of points within a radius, R, then
    define a mask by dropping values below threshold T
    '''
       
    # initialize columns for the smoothed values and mask (1/0)
    points['conv'] = np.zeros(points.shape[0])
    points['mask'] = np.zeros(points.shape[0])
    
    # get the subset of points with values > 0, loop through these points
    df_sub = points[(points['max'] > 0)]    

    for i in range(df_sub.shape[0]):
        
        ind = df_sub.index.values[i]
        lat = df_sub.loc[ind,'lat']
        lon = df_sub.loc[ind,'lon']

        # get subset of huc centroids within radius R of each candidate point (includes zeros)
        df_in = points[(np.sqrt((points['lat'] - lat)**2 + (points['lon'] - lon)**2)) <= R].copy()
        
        # get the mean of all values within the radius, write filtered value to dataframe
        points.loc[ind,'conv'] = df_in['max'].mean()

    # set mask = 1 where filtered value > threshold T
    points.loc[points['conv'] > T,'mask'] = 1

    return points
    
    
def split_objects(points, gap_thresh):
    '''
    
    THIS IS NO LONGER USED
    
    separates points into groups in each dimension (lat/lon), separated by distance gap_thresh
    then finds intersecting regions between to two dimensions, and assigns object numbers to 
    the intersecting areas
    '''
    points['obj'] = np.zeros(points.shape[0]).astype("int64")

    # get object regions in each dimension 
    regions_x = get_regions(points, 'lon', gap_thresh)
    regions_y = get_regions(points, 'lat', gap_thresh)

    # assign object number to points within each unique region - i.e. intersections of x, y regions
    obj = 1
    for rx in regions_x:
        for ry in regions_y:
            # get set of points within the intersecting region
            n_pts = points[(points['lon'] >= rx[0]) & (points['lon'] <= rx[1]) &
                           (points['lat'] >= ry[0]) & (points['lat'] <= ry[1])].shape[0]
            
            # if points exist, assign object number and increment obj, otherwise do nothing
            if n_pts > 0:
                points.loc[(points['lon'] >= rx[0]) & (points['lon'] <= rx[1]) &
                           (points['lat'] >= ry[0]) & (points['lat'] <= ry[1]),'obj'] = obj
                obj += 1
    
    return points
    
    
def get_regions(points, dim, gap_thresh):

    dmin = points[dim].min()
    dmax = points[dim].max()

    # starting region is full range
    regions = []

    d = dmin
    region_min = dmin
    while d < dmax:

        d_next = d + gap_thresh
        subset = points[(points[dim] > d) & (points[dim] <= d_next)]
        n_pts = subset.shape[0]   
        
        #print(d, d_next, n_pts)

        if n_pts == 0:
            # gap found, record region limits prior to gap
            region_max = d
            #print('gap found between ', d, d_next)
            #print('region of points between ', region_min, region_max)
            regions.append([region_min, region_max])
            # get starting point of next region after gap
            
            region_min = points.loc[(points[dim] > d), dim].min()
            d = region_min

        else:
            # next x is the point with max x within this interval
            d = subset[dim].max()

    region_max = dmax
    regions.append([region_min, region_max])

    return regions
    
def obj_bound_buff(df_objs, gdf, max_obj, buffer_radius, alpha = 1.0):
    '''
    get the 'concave hull' around the polygon points making up each object
    and add a fixed buffer
    ''' 
    objects = list(set(df_objs['obj']))

    gdf_bounds = gpd.GeoDataFrame()    

    #set up a geodataframe of hucs in regions
    gdf_hucs = gdf.set_index('HUC10')
    gdf_hucs['obj'] = np.full(len(gdf), -999)        

    for obj_num in objects:

        df_mask_obj = df_objs[df_objs['obj'] == obj_num]
        gdf_mask_obj = gdf[gdf['HUC10'].isin(df_mask_obj.index)]

        points=[]
        coords=[]
        for index, row in gdf_mask_obj.iterrows():
            pt = (row.lon, row.lat)
            points.append(Point(pt))
            coords.append(pt)    

        # get the outer boundary and add buffer
        print('Getting outer boundary of object', obj_num)
        concave_hull, edge_points = alpha_shape(points, alpha=alpha)
        concave_hull_buff = concave_hull.buffer(buffer_radius)
        
        # check if the boundary became a multipolygon, if so, separate into
        # separate polygons and add an object number
        obj_list = [obj_num]    
        if concave_hull_buff.geom_type == 'MultiPolygon':
            print('re-splitting object', obj_num)
            poly_list = list(concave_hull_buff)
            for i in range(1,len(poly_list)):
                max_obj += 1
                obj_list.append(max_obj)
        else:
            poly_list = [concave_hull_buff]

        # add the outer boundary(s) to geodataframe
        new_dict = {'obj' : obj_list, 'geometry': poly_list}
        gdf_new = gpd.GeoDataFrame(new_dict)
        gdf_bounds = gdf_bounds.append(gdf_new)
        
        # rebuild set of hucs within each object
        
        # get min/max lat/lat to subset the huc10s
        x = [p.coords.xy[0] for p in points]
        y = [p.coords.xy[1] for p in points]    
        
        # get a subset of candidate hucs
        sub_gdf = gdf[(gdf['lon'] >= np.min(x)-1) & (gdf['lon'] <= np.max(x)+1) & \
                      (gdf['lat'] >= np.min(y)-1) & (gdf['lat'] <= np.max(y)+1)]    
        
        for i, obj in enumerate(obj_list):
            
            bound = poly_list[i]
            print('Getting set of hucs within the buffered boundary for object ', obj)

            for index, row in sub_gdf.iterrows():
                p = Point(row['lon'], row['lat'])
                if p.within(bound):
                    gdf_hucs.loc[row['HUC10'], 'obj'] = obj

    # drop the hucs with no object assignment
    gdf_hucs = gdf_hucs[gdf_hucs['obj'] > 0]

    # reset the boundary index
    gdf_bounds = gdf_bounds.set_index('obj')
    
    return gdf_bounds, gdf_hucs
    
    
def drop_objects(df_objs, df_huc, min_cluster = 5, cluster_thresh = 10):

    objects = list(set(df_objs['obj']))
    df_objs['max'] = df_huc.loc[df_objs.index, 'max']

    df_keep = pd.DataFrame()

    for obj in objects:
        df = df_objs[df_objs['obj'] == obj]
        n = df[df['max']> cluster_thresh].shape[0]
        if n > min_cluster:
            df_keep = df_keep.append(df)
            print('keeping object', obj, 'with ', n, 'HUCs over 10 %')
        else:
            print('dropping object', obj, 'with ', n, 'HUCs over 10 %')
            
    return df_keep 
    
    
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points
    
    
######################################################################################
# data processing - mean areal precipitation
######################################################################################

def check_projection(domain, shp_file, gdf_poly):
    '''
    check geodataframe projection and write temporary shapefile 
    of region of interest hucs for MAP processing -
    (** check into using a geodataframe with raster stats to avoid writing shapefile)
    '''
    
    print('\nChecking shapefile projection')
    gdf_poly, is_reproj = shp_to_crs(gdf_poly, domain)
    gdf_poly.to_file(shp_file)
    
    return 


    
def huc_mean_areal_precip(version_dir, ref_time,
                          config, filelist, 
                          metric, gdf_poly, #shp_file,
                          shp_header = "HUC10", 
                          shp_tag = "eval_reg",
                          outdir_suffix = "mean",
                          use_existing = True, 
                          all_touched = False):
    '''
    shp_header = column header for IDs in the geodataframe
    shp_tag = region tag - in this case region of interest, not full domain  
    outdir_suffix = suffix added to forcing directory name for MAP storage location  
    read_if_exists = read MAP file if already exists in cache - set to False to override
    '''
           
    # add a tag to the reference time string to identify that
    # this MAP was generated for an evaluation, which (for ana) 
    # dicatates the timesteps included
    ref_tag = metric
    
    # set of polygons to calculate maps - create a copy to maintain orig list, calc list may change below
    calc_poly = gdf_poly.copy()
    
    merge_prior = False

    if calc_poly.index.name != shp_header:
        calc_poly.set_index(shp_header)

    polyids = calc_poly.index.values 
    
    # Build output dir for MAP files
    mapout_dir = build_mapout_dir(version_dir, ref_time, config, outdir_suffix)
    
    # Build MAP output filename, check if it already exists
    map_path, map_exists = get_map_path(mapout_dir, ref_time, config, filelist[0], shp_tag, ref_tag)

    # if skip flag = True, output already exsits and not generating individual step output, skip iteration
    if use_existing:
        if map_exists:
            print('  MAP timeseries output already exists for this ref time - reading the file')
            df_map = pd.read_csv(map_path, index_col = 0)
            
            check = np.isin(polyids, df_map.index)
            
            # check if all needed HUC10s were processed previously, are included in this file
            if not all(check):
                print('  MAPs do not exist for all needed HUC10s')
                df_map_prior = df_map.copy()
                
                calc_poly = calc_poly[~check]
                polyids = calc_poly.index.values
                
                # flag to merge the prior and new MAPs at the end
                merge_prior = True
          
            else:
                print('  MAPs already processed for all HUC10s in the region') 
                return df_map
            
    
    # loop through a filelist and generate MAP timeseries file 
    # (currently csv, need to update to SQLite)
    df_map = map_by_filelist(version_dir, filelist, calc_poly, all_touched)

    # check number of timesteps missing
    nmissing = df_map.isnull().sum(axis=1).max()
    
    # if only 1 timestep missing, calculate the sum, otherwise set sum to missing
    if not df_map.empty and nmissing < 2:
        df_map['sum_ts'] = df_map.iloc[:,1:].sum(axis = 1)   
        
        if merge_prior:
            df_map = df_map_prior.append(df_map).sort_index()
        
        print('Writing time series for this ref time', map_path)
        df_map.to_csv(map_path)    
    else:
        # NaN if more than 1 timestep missing
        df_map['sum_ts'] = np.full(len(polyids), np.nan)
        print('MAP not generated for reference time: ', ref_time)
    
    # losing the index header somewhere
    df_map.index = df_map.index.rename(shp_header)
        
    return df_map
    

def build_mapout_dir(base_dir, ref_time, config, suffix):

    # create directory name from ref time and config
    datedir = ref_time.strftime("nwm.%Y%m%d")
    config_dir = 'forcing_' + config
    
    # datedir = nwm_path.split("\\")[0]    
    #config_dir = nwm_path.split("\\")[1]
    #filename = nwm_path.split("\\")[2]

    # build output path
    mapout_dir = base_dir / datedir / (config_dir + "_" + suffix)

    # check if output directory exists, if not create
    create_dir_if_not_exist(mapout_dir)

    return mapout_dir
    

def get_map_path(mapout_dir, ref_time, config, nwm_path, shp_tag, ref_tag):

    # parse out first 4 sections of nwm filename from full_path
    parts = nwm_path.name.split(".")[0:4]
    
    # make sure the configuration portion of the filename matches variable 'config'
    # needed for the ROF 'latest ana' case (first filename could be either
    # standard or extended AnA)
    parts[2] = config
    
    # replace timestep indentifier (e.g. f001) with "series", 
    #    "conus" with shp_tag, and extension to "csv"
    parts.extend(['series',shp_tag,'csv'])  
        
    ref_hr_str = ref_time.strftime("t%Hz")
    parts[1] = ref_hr_str
    
    if mapout_dir.name == "forcing_realtime_mean":
        parts[2] = ref_tag
    elif ref_tag:
        parts[1] = parts[1] + "." + ref_tag    

    filename = ".".join(parts)

    path = mapout_dir / filename

    # check if file exists, if yes, return flag to skip
    file_exists = False
    if path.exists():   
        file_exists = True 
        
    return path, file_exists


def map_by_filelist(version_dir, filelist, calc_poly, all_touched):#, shp_header):

    df_map = pd.DataFrame()   
    success = True
    npoly = len(calc_poly)
    
    # Loop through files in the timeseries  
    for i, nwm_path in enumerate(filelist):
        
        t_start = time.time()

        grid_path = version_dir / nwm_path

        # add catch if grid wasn't downloaded/doesn't exist, fill with NaN
        if not grid_path.exists():   
            print('grid missing for grid: ', str(nwm_path))
            df_map['mean'] = np.full(npoly,np.nan)
            
        else:

            ############# calculate mean areal values for polygons ##############
            
            print('Calculating mean areal values for grid: ', str(nwm_path))
            #df_zstats = get_grid_stats(grid_path, shp_file, polyids, shp_header)
            df_zstats = get_grid_stats(grid_path, calc_poly, all_touched)#, shp_header)

            # if failed for some reason, e.g. netcdf file is corrupt, fill with NaN
            # but allow process to proceed (MAP plots will be empty)
            if df_zstats.empty:
                print('\n !! GRID PROCESSING FAILED FOR : ', str(nwm_path))
                
                df_zstats = pd.DataFrame({'mean' : np.full(npoly,np.nan)}, index = polyids)
            
            # convert from mm s-1 to mm hr-1
            df_zstats["mean"] = df_zstats["mean"] * 60 * 60  
         
            if i == 0: 
                # first timestep, keep the count, after that only keep mean
                df_map = df_zstats.loc[:,['count','mean']]
                df_map = df_map.rename(columns={"count":"cell_count"})
            else:
                df_map = pd.concat([df_map, df_zstats[['mean']]], axis = 1)

        fileparts = nwm_path.name.split(".")
        df_map = df_map.rename(columns={"mean": fileparts[1] + "-" + fileparts[-3]})
        df_map.index = df_map.index.rename(calc_poly.index.name)

        t_stop = time.time()
        print('Elapsed time this file', t_stop - t_start)   
        
    return df_map
    
        
def geom_to_crs(gdf, domain, **kwargs):

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
    
    if check_crs is None:
        gdf = gdf.set_crs(epsg=4269)
        
    elif check_crs.name == crs_name:
        print('Geometry already in correct projection: ', check_crs.name)
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
    print("Projecting geometry to", nwm_crs.name) 
    gdf_reproj = gdf.to_crs(nwm_crs)
    #print("Reprojecting shapefile to ", gdf_reproj.crs.name)
        
    return gdf_reproj, True
    
    
def affine_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def get_grid_stats(grid_path, calc_poly, all_touched):#, shp_header):

    try:
        ds_sr = xr.open_dataset(grid_path)
    except OSError:
        print('Cannot open file: ', str(grid_path))
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

    zstats = zonal_stats(vectors = calc_poly['geometry'], raster = rain_flip, affine=transform, 
                         nodata = -999, stats="count mean", all_touched = all_touched)

    df_zstats = pd.DataFrame(zstats, index = calc_poly.index)

    return df_zstats
    
    
def rof_fig_text(version, domain, verif_config, ref_time, out_dir, order_max, spatial_agg_method):

    verif_abbrev = pd.Series(['Standard','Extended','Current Avail'],
             index = ["analysis_assim", "analysis_assim_extend", "latest_ana"])
             
    # set up text for overall figure heading
    vtime_start_str = (ref_time + timedelta(hours=1)).strftime("%m-%d %Hz")
    vtime_end_str = (ref_time + timedelta(hours=18)).strftime("%m-%d %Hz")
    fig_title = 'Ref Time: ' + ref_time.strftime("%Y-%m-%d %Hz") + \
                     ' (Valid Times: '+ vtime_start_str + ' to ' + vtime_end_str + ')' + \
                     ' | AnA Source: ' + verif_abbrev.loc[verif_config] + \
                     ' | Order limit: ' + str(order_max) + \
                     ' | Aggregate by: ' + spatial_agg_method
                     
    # define output directory, filename and save PNG file
    if verif_config == 'latest_ana':
        out_dir = out_dir / 'png' / 'LatestAnA'
    elif verif_config == 'analysis_assim_extend':
        out_dir = out_dir / 'png' / 'ExtAnA'   
    else: # "analysis_assim"
        out_dir = out_dir / 'png' / 'StdAnA'  

    # create output dir if it doesnt exist
    create_dir_if_not_exist(out_dir)
        
    if order_max == 0 and spatial_agg_method == 'str_num':
        rof_ver = '_ROFv1'
    elif order_max == 4 and spatial_agg_method == 'str_length':
        rof_ver = '_ROFv2'
    else:    
        rof_ver = '_other'
         
    fig_path = out_dir / ("ROF_9panel_" + domain + ref_time.strftime("_%Y%m%d_%Hz") + ".png")
                 
    return fig_title, fig_path

                     
def nine_panel_conus(df_stats_huc, gdf_region_hucs_eval, gdf_bounds, df_reach,
                     event_thresh, eval_config, verif_config, fig_path, fig_title, gdf_states):
                     
    # show stats only for largest object/region on the plot
    max_obj = df_stats_huc['tot'].max()
    df_show_stats = df_stats_huc[df_stats_huc['tot'] == max_obj]
    gdf_max_bound = gdf_bounds.loc[df_show_stats.index]
    gdf_rest_bound = gdf_bounds[~gdf_bounds.index.isin(df_show_stats.index)]

    # TEMPORARY - column headers to read reach DF to get # ROF reaches and total stream length 'in ROF'
    reach_col = ['','','','srf_rof','exana_rof']
    if verif_config != 'analysis_assim_extend':
        reach_col[4] = 'stana_rof'    
    
    ######################### Set-up labels and text ##############################
    
    # event threshold (defined in configuration definition at top)
    event_text = ' - ' + str(event_thresh) + '% Event Threshold'    

    # abbreviations for plot headings and wording for title
    # based on the eval_config and verif_config 
    abbrev = pd.Series(['srf','mrf','stana','exana'],
             index = ["short_range", "medium_range", "analysis_assim", "analysis_assim_extend"])
    
    verif_abbrev = pd.Series(['Standard','Extended','Current Available'],
             index = ["analysis_assim", "analysis_assim_extend", "latest_ana"])

    eval_label = get_abbrev(eval_config).upper()
    verif_label = get_abbrev(verif_config).upper()

    # set up remainder of the plot heading text
    subtitle_text = [eval_label + ' MAP (QPF)',
                     verif_label + ' MAP (QPE)',
                     'Difference (QPF-QPE)',
                     eval_label + ' ROF %',
                     verif_label + ' ROF %',
                     'Difference (' + eval_label + ' % - ' + verif_label + ' %)', 
                     eval_label + ' Exceeds ' + event_text,
                     verif_label + ' Exceeds ' + event_text,
                     'Contingency Matrix' + event_text]    

    # specify data colummns to plot and convert precip values to inches in the dataframe
    gdf = gdf_region_hucs_eval.copy()
    
    col_labels = ['SRF_MAP','AnA_MAP','MAP_DIFF',
                  'SRF_pct','AnA_pct','perc_diff',
                  'SRF_event','AnA_event','matrix']    
    
    # convert mm to inches for plots
    for col in col_labels[0:3]:
        gdf[col] = gdf[col] / 25.4
        
    cmaps, cmaps_cb, norms, bounds = fig_colors(gdf) 
         
    ################################### set up the figure ###############################
    
    # set up the figure
    xlim = [-126, -60]
    ylim = [24, 53]
    figsize = (30.5, 16)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize, squeeze=0, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.03, wspace=0.02, top=0.95)
    
    # loop through the axes, adding maps/plots
    for i, ax in enumerate(axes.flat):

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)       
        
        ############### HUC polygons ###################
        if i in [0, 1, 2]:
            # if precip was missing, leave MAP plots blank
            if not gdf[col_labels[i]].isnull()[0]:
                gdf.plot(ax=ax, column = col_labels[i], cmap = cmaps[i], norm = norms[i], zorder = 3)
            else:    
                ax.text('MAP data error', np.mean(xlim), np.mean(ylim), ha = 'center', fontsize = 20)
            
        else:
            gdf.plot(ax=ax, column = col_labels[i], cmap = cmaps[i], norm = norms[i], zorder = 3)
        
        ############### colorbars ###################
        # precip use defined bins above
        
        if i < 6:
            cb_pos = [0.96, 0.03, 0.03, 0.94]
            cbaxes = ax.inset_axes(cb_pos)
            cmap = mpl.cm.get_cmap(cmaps_cb[i])
            norm = norms[i]
            
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbaxes, orientation='vertical',ticklocation='left')
                   
            cb.set_ticks(bounds[i])      
            if i < 2:
                cb.set_ticks(bounds[i][:-1])   
                
            if i < 3:
                cb.set_label(label='inches',size = 15)
            else:
                cb.set_label(label='%',size = 17)                            
        
        subtitle_x = xlim[0] + 0.5
        subtitle_y = ylim[1] - 2
        ax.text(subtitle_x, subtitle_y, subtitle_text[i], fontsize = 22)
        ax.tick_params(axis='both', which='major', labelsize=16)

        # annotate bottom row

        # Event yes/no maps
        if i in [6,7]:
        
            nyes = gdf[gdf[col_labels[i]]].count()[0]
            nno = gdf.shape[0] - nyes
        
            leg_label = ['No (' + str(nno) + ')','Yes (' + str(nyes) + ')'] 
            patch_x = subtitle_x + 1.5
            patch_y0 = ylim[0]+1.2
            text_x = patch_x + 1.5
            text_y0 = patch_y0 - 0.5
            dy = 2
            patch_y = [patch_y0, patch_y0 + dy]
            text_y = [text_y0, text_y0 + dy]

            for j in range(2):
                square = ax.scatter(patch_x, patch_y[j], color = cmaps[i](j), marker='s',  edgecolor = 'black', s=300, zorder = 2)            
                ax.text(text_x, text_y[j], leg_label[j], fontsize = 18)
                                      
        # contingency matrix map
        if i == 8:
            leg_label_left = ['True Pos:  ','False Pos:  ', 'False Neg:  ', 'True Neg:  ']
            leg_label_right = ['POD =  ','FAR =  ', 'CSI =  ', 'POFD =  ']  
            patch_x = subtitle_x + 1.5
            patch_y0 = ylim[0]+1.5
            left_text_x = patch_x + 1.5
            right_text_x = xlim[1] - 0.5

            text_y0 = patch_y0 - 0.5
            dy = 1.6
            patch_y = [patch_y0 + 3*dy, patch_y0 + 2*dy, patch_y0 + dy, patch_y0]
            text_y = [text_y0 + 3*dy, text_y0 + 2*dy, text_y0 + dy, text_y0]
            leg_title_y = text_y0 + 3*dy

            vals_left = [str(i) for i in df_show_stats.iloc[:,1:5].values.tolist()[0]] 
            vals_right = [f'{i:.3f}' for i in df_show_stats.iloc[:,5:9].values.tolist()[0]]       
       
            for j in range(3):#(4):
                square = ax.scatter(patch_x, patch_y[j+1], color = cmaps[i](j), edgecolor = 'black', marker='s', s=200, zorder = 2)            
                ax.text(left_text_x, text_y[j+1], leg_label_left[j] + vals_left[j], fontsize = 18)
            
            ax.text(right_text_x, leg_title_y,'-- Stats Region', fontsize = 18, fontweight = 'bold', ha = 'right')
            for j in range(3):
                ax.text(right_text_x, text_y[j+1], leg_label_right[j] + vals_right[j], fontsize = 18, ha = 'right')
                
        # add outer boundary around objects, with thicker black dashed line around selected (largest) object
        gdf_max_bound.plot(ax=ax, facecolor = 'none', edgecolor='black', linewidth = 1.5, linestyle = '--', zorder = 2)
        gdf_rest_bound.plot(ax=ax, facecolor = 'none', edgecolor='white', linewidth = 1.5, zorder = 2)
    
        gdf_states.plot(ax=ax, alpha=0.5, facecolor = 'lightgray', edgecolor='gray', zorder = 1)

    fig.suptitle(fig_title, fontsize=20, y = 0.98)   
    print('writing PNG file: ', fig_path)
    plt.savefig(fig_path, dpi=150, bbox_inches = "tight")
    
    
    
def region_shapefile(dict_map, df_huc_eval, gdf_huc10_dd,
                     eval_config, verif_config, col_heads):
    
    col_labels = ['sum_ts','sum_ts','sum_ts',
                   col_heads[0], col_heads[1], 'diff', 
                   col_heads[2], col_heads[3], 'matrix','obj']   
    
    rename_labels = ['SRF_MAP','AnA_MAP','MAP_DIFF',
                     'SRF_pct','AnA_pct','perc_diff', 
                     'SRF_event','AnA_event', 'matrix','obj']   
    
    gdf_dd = gdf_huc10_dd     
                
    for i in range(len(col_labels)):
 
        if i == 0:
            df = dict_map[eval_config]
        elif i == 1:
            df = dict_map[verif_config]
        elif i == 2:
            df = dict_map[eval_config] - dict_map[verif_config]
        if i in [3, 4, 5, 6, 7, 8]:
            df = df_huc_eval            
            
        # losing the index header somewhere
        df.index = df.index.rename('HUC10')

        gdf_dd = gdf_dd.merge(df[col_labels[i]], on = 'HUC10', how = "inner")
        gdf_dd = gdf_dd.rename(columns = {col_labels[i] : rename_labels[i]}) 
        
    gdf_dd['matrix_x'] = gdf_dd['matrix'].replace([1, 2, 3, 4], ['TP', 'FP', 'FN', 'TN'])    
           
    return gdf_dd


def write_output(ref_time, out_dir, domain, gdf_dd, df_gages, gdf_bounds):

    ref_str = ref_time.strftime("%Y%m%d_t%H")

    # Instantiate output dirs
    shp_out_dir = out_dir / "shp"
    csv_out_dir = out_dir / "csv"

    # Create output dirs if they do not exist
    create_dir_if_not_exist(shp_out_dir)
    create_dir_if_not_exist(csv_out_dir)
    
    # shapefiles - all HUC10s
    shpfile_dd = shp_out_dir / ('region_hucs_' + ref_str + '_' + domain + '_dd.shp')
    gdf_dd.to_file(shpfile_dd)    
    
    # shapefiles - region boundary
    shpfile_bounds_dd = shp_out_dir / ('region_bounds_' + ref_str + '_' + domain + '_dd.shp')
    gdf_bounds = gdf_bounds.set_crs(epsg=4269)
    gdf_bounds.to_file(shpfile_bounds_dd)

    # list of gages in region of interest
    gagelist_fname = csv_out_dir / ('region_gagelist_' + ref_str + '_' + domain + '.csv')
    df_gages_region = df_gages.loc[df_gages['HUC10'].isin(gdf_dd['HUC10']),:].copy()
    df_gages_region['feature_id'] = df_gages_region.index
    df_gages_region = df_gages_region.set_index('gage')
    df_gages_region = df_gages_region[['feature_id','lat','lon','HUC10']]
    df_gages_region.to_csv(gagelist_fname)
    
    return df_gages_region
    
def empty_fig(fig_path, fig_title, gdf_states):
      
    # set up the figure
    xlim = [-126, -66]
    ylim = [24, 52]
    figsize = (36, 20)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize, squeeze=0, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.03, wspace=0.02, top=0.95)
    
    # loop through the axes, adding maps/plots
    for i, ax in enumerate(axes.flat):

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)      
        
        gdf_states.plot(ax=ax, alpha=0.5, facecolor = 'lightgray', edgecolor='gray', zorder = 1)         
         
        ax.text(np.mean(xlim), np.mean(ylim), 'No HUCs meet ROF criteria', ha = 'center', fontsize = 16)

    fig.suptitle(fig_title, fontsize=20, y = 0.98)   

    print('writing PNG file: ', fig_path)
    plt.savefig(fig_path, dpi=150, bbox_inches = "tight")
    
def create_dir_if_not_exist(p: Path):
    if not p.exists():
        print(f"Directory {p} does not exist. Creating {p}")
        p.mkdir(parents=True, exist_ok=True)

def fig_colors(gdf):

    # define color maps and colorbars

    # Precip maps - WPC colormap
    cm_map_list = [[0.,0.,0.,0.],
                   'lime',
                   'limegreen',
                   [0, 0.55, 0.25, 1.], 
                   [0, 0.35, 0.45, 1.],
                   'dodgerblue',
                   'deepskyblue',
                   'cyan',
                   'mediumpurple',
                   'darkorchid',
                   'darkmagenta',
                   'firebrick',
                   'red',
                   'orangered',
                   'darkorange',
                   'darkgoldenrod',
                   'gold',
                   'yellow',
                   'lightpink']
    bounds_map = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 7, 10, 15, 20, 100]
    
    cm_map_cbar_list = cm_map_list.copy()
    cm_map_cbar_list[0] = [0.1,0.1,0.1,0.05]
    
    cmap_map = ListedColormap(cm_map_list)       
    cmap_map_cbar = ListedColormap(cm_map_cbar_list)

    # ROF maps
    cm_rof_list = [[0., 0., 0., 0.],
                   [0.98046875, 0.86328125, 0.42578125, 1.],
                   [0.984375  , 0.69921875, 0.265625, 1.],
                   [0.9453125 , 0.5546875 , 0.359375, 1.],
                   [0.85546875, 0.44921875, 0.453125, 1.],
                   [0.73046875, 0.37890625, 0.59765625, 1.],
                   [0.6015625 , 0.328125  , 0.5703125, 1.],
                   [0.46484375, 0.28125   , 0.5546875, 1.],
                   [0.328125  , 0.28515625, 0.44140625, 1.],
                   [0.24609375, 0.24609375, 0.24609375, 1.]]
                    
    bounds_rof = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    cm_rof_cbar_list = cm_rof_list.copy()
    cm_rof_cbar_list[0] = [0.1,0.1,0.1,0.05]
    
    cmap_rof = ListedColormap(cm_rof_list)       
    cmap_rof_cbar = ListedColormap(cm_rof_cbar_list)

    # Difference maps
    
    getmap = cm.get_cmap('jet_r', 25)
    getarr = getmap(np.linspace(0, 1, 25))
    trans = [0.1, 0.1, 0.1, 0.]
    gray = [0.1, 0.1, 0.1, 0.05]
    
    cmap_map_diff = ListedColormap(np.vstack((getarr[0:9,:], [1., 1., 0., 1.], trans, getarr[15:,:])), name='Diff_map')
    cmap_map_diff_cbar = ListedColormap(np.vstack((getarr[0:9,:], [1., 1., 0., 1.], gray, getarr[15:,:])), name='Diff_map')
    bounds_diff_map = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5,
                      -0.1, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    
    cmap_rof_diff = ListedColormap(np.vstack((getarr[0:9,:], trans, trans, trans, getarr[16:,:])), name='Diff_map')
    cmap_rof_diff_cbar = ListedColormap(np.vstack((getarr[0:9,:], gray, gray, gray, getarr[16:,:])), name='Diff_map')
    bounds_diff_rof = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,-1,
                        1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    #contingency matrix
    colors_mat = ['lime','yellow','red',[0., 0., 0., 0.0]]
    cmap_matrix = ListedColormap(colors_mat)

    colors_event = ['white','black']
    cmap_event = ListedColormap(colors_event)
    
    cmaps = [cmap_map, cmap_map, cmap_map_diff, 
             cmap_rof, cmap_rof, cmap_rof_diff, 
             cmap_event, cmap_event, cmap_matrix]  
    
    cmaps_cb = [cmap_map_cbar, cmap_map_cbar, cmap_map_diff_cbar, 
                cmap_rof_cbar, cmap_rof_cbar, cmap_rof_diff_cbar]  
                
    bounds = [bounds_map, bounds_map, bounds_diff_map, 
              bounds_rof, bounds_rof, bounds_diff_rof]
                         
    # define limits for colormaps/colorbars
    # set precip max (plim) for colormap to either 100 or the max value in the data
    # rof percent limit (rlim) is 100   
    plim = 5
    rlim = 100
    vmins = [0, 0, -plim, 
             0, 0, -rlim, 
             0, 0, 1]
    vmaxs = [plim, plim, plim,
             rlim, rlim, rlim, 
             1, 1, 4]
             
    norms = []         
    for i in range(9):
        if i < 6:
            norm = mpl.colors.BoundaryNorm(bounds[i], cmaps[i].N)
        else:
            norm = mpl.colors.Normalize(vmin=vmins[i], vmax=vmaxs[i])
            
        norms.append(norm)
                
    return cmaps, cmaps_cb, norms, bounds
