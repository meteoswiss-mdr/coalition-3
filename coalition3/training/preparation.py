""" [COALITION3] Functions to prepare generation of training dataset:
    Generate list of datetime objects included in the training period
    and get respective information from TRT text files."""

from __future__ import division
from __future__ import print_function

import os
import datetime
import numpy as np
import pandas as pd

from pandas.api.types import CategoricalDtype
from coalition3.inout.paths import path_creator
from coalition3.inout.readconfig import get_config_info_op
from coalition3.visualisation.TRTcells import print_TRT_cell_histograms, print_TRT_cell_map

sys.path.insert(0, '/opt/users/jmz/monti-pytroll/packages/mpop')
from mpop.satin import swisstrt

## =============================================================================
## FUNCTIONS:

## Create list of datetime objects, at which samples should be generated for the training dataset:
def create_dt_sampling_list(cfg_set_tds):
    """Create list of datetime objects, at which samples should be
    generated for the training dataset.
    """
    print("Creating list of sampling datetime objects")
    
    ## Append further datetime objects accroding to settings in cfg_set_tds:
    dt_temp = datetime.datetime.combine(cfg_set_tds["tds_period_start"],datetime.time(0,0))
    
    ## Insert starting date as first element:
    dt_sampling_list = []
    while dt_temp.date()<=cfg_set_tds["tds_period_end"]:
        dt_sampling_list.append(dt_temp)
        day_temp = dt_temp.day
        dt_temp = dt_temp+datetime.timedelta(minutes=cfg_set_tds["dt_samples"])
        if day_temp < dt_temp.day:
            dt_temp = dt_temp+datetime.timedelta(minutes=cfg_set_tds["dt_daily_shift"])
            day_temp = dt_temp.day
    print("  Number of sampling times points: %s" % len(dt_sampling_list))
    return dt_sampling_list

## Get TRT values from single .trt files on disk and concatenate into dataframe
## WARNING: Quite time consuming (opens all TRT files within time-frame)
def get_TRT_cell_info(dt_sampling_list,cfg_set_tds,cfg_set_input=None,len_ini_df=None):
    """Get information on TRT cells within time period.
    
    Parameters:
    -----------
    
    len_ini_df : uint
        Length of initial dataframe (to setup the dataframe, if number of TRT cells
        exceeds this initial length, additional lines are appended, if there are fewer,
        the exceeding lines are deleted.
    """
    print("Estimate number of samples within training period")
    
    ## Get input data config file
    if cfg_set_input is None:
        cfg_set_input, cfg_var = get_config_info_op()
                               
    ## Create empty DataFrame
    if len_ini_df is None: len_ini_df = len(dt_sampling_list)*3
    ## Old:
    #df_cols = ["traj_ID","date","RANKr","area","lat","lon","iCH","jCH"]
    #samples_df = pd.DataFrame(np.zeros((len_ini_df,len(df_cols)))*np.nan,
    #                          columns=df_cols)
    ## New:
    #samples_df = Nip.df_empty(["traj_ID"]+cfg_set_input["TRT_cols"],[np.object]+cfg_set_input["TRT_dtype"])   
    samples_ls = []
    
    #ind_df = 0; first_append = True; doy_temp = -1
    
    ## Loop over time steps to gather information on TRT cells at specific time step:
    for sampling_time in dt_sampling_list:
        perc_checked = np.round((sampling_time.hour*60+sampling_time.minute)/1440.,2)*100
        print("  Check input data availability of date: %s - %02d%%  " % (sampling_time.strftime("%d.%m.%Y"),perc_checked), end='\r')
        
        ## Update time in config dict:
        cfg_set_input["t0"]     = sampling_time
        t0                      = cfg_set_input["t0"]
        cfg_set_input["t0_doy"] = t0.timetuple().tm_yday
        cfg_set_input["t0_str"] = t0.strftime("%Y%m%d%H%M")
        #if cfg_set_input["t0_doy"]%10==0 and cfg_set_input["t0_doy"]!=doy_temp:
        #    print("   For doy: %s" % cfg_set_input["t0_doy"])
        #    doy_temp = cfg_set_input["t0_doy"]
        
        ## Get file path to respective TRT file of time point sampling_time:
        filepaths, timestamps = path_creator(sampling_time, "TRT", "TRT", cfg_set_input)
        
        ## In case file is not available, look for files just right before and after this timepoint
        ## (e.g. if no file available at 16:35, look at 16:25/16:30/16:40/16:45), otherwise skip this time point.
        if filepaths[0] is None:
            for dt_daily_shift_fac in [-1,1,-2,2]:
                sampling_time_temp = sampling_time + dt_daily_shift_fac*datetime.timedelta(minutes=cfg_set_tds["dt_daily_shift"])
                filepaths_temp, timestamps = path_creator(sampling_time_temp, "TRT", "TRT", cfg_set_input)
                if filepaths_temp[0] is not None:
                    filepaths = filepaths_temp
                    print("       Instead using dataset: %s" % filepaths[0])
                    break
        if filepaths[0] is None: 
            print("       No files found, skip this timepoint")
            continue
            
        ## Read in TRT-info:
        traj_IDs, TRTcells, cell_mask = swisstrt.readRdt(filepaths[0])        
        for traj_ID in traj_IDs:
            ## New:
            dict_cellinfo = {key:value for key, value in TRTcells[traj_ID].__dict__.items() if not key.startswith('__') and not callable(key)}
            #cell_info_df  = pd.DataFrame.from_records([dict_cellinfo], index=[9])
            #samples_df_append = pd.DataFrame([[traj_ID]],columns=["traj_ID"],index=[9]).join(pd.DataFrame.from_records([dict_cellinfo],index=[9]))
            #samples_df = samples_df.append(samples_df_append, ignore_index=True, sort=True)
            samples_ls.append(pd.DataFrame([[traj_ID]],columns=["traj_ID"],index=[9]).join(pd.DataFrame.from_records([dict_cellinfo],index=[9])))
            ## Old:
            """
            cell = TRTcells[traj_ID]
            cell_date = datetime.datetime.strptime(cell.date,"%Y%m%d%H%M")
            if ind_df <= len_ini_df-1:
                samples_df.iloc[ind_df,:] = [traj_ID,cell_date,cell.RANKr,cell.area,
                                             cell.lat,cell.lon,int(cell.iCH),int(cell.jCH)]
            else:            
                if first_append: print("   *** Start appending to dataframe at t = %s ***" % sampling_time)
                first_append = False
                samples_df = samples_df.append(pd.DataFrame([[traj_ID,cell_date,cell.RANKr,cell.area,
                                                              cell.lat,cell.lon,int(cell.iCH),int(cell.jCH)]],
                                               columns=["traj_ID","date","RANKr","area","lat","lon","iCH","jCH"]))
            ind_df += 1
            """
    
    samples_df = pd.concat(samples_ls)
    
    ## Only keep non-nan lines (where there are TRT cells):
    #print("   Lenght of dataframe before dropping of nans: %s" % samples_df.shape[0])
    #print("   Index of dataframe after filling: %s" % ind_df)
    samples_df = samples_df.dropna()
    print("   Lenght of dataframe after dropping of nans: %s" % samples_df.shape[0])
    print("   Number of different TRT cells: %s\n" % len(np.unique(samples_df["traj_ID"])))
    print(samples_df.info(),"\n")
    print(samples_df,"\n")
    samples_df.to_pickle(os.path.join(cfg_set_tds["root_path_tds"],u"TRT_sampling_df_testset.pkl"))
    print("   Dataframe saved in: %s" % os.path.join(cfg_set_tds["root_path_tds"],u"TRT_sampling_df_testset.pkl"))
    return(samples_df)

## Print some basic information on the TRT cells which will be sampled:
def print_basic_info(cfg_set_tds):
    samples_df = pd.read_pickle(os.path.join(cfg_set_tds["root_path_tds"],
                                             u"TRT_sampling_df_testset_enhanced.pkl"))
    print("Basic information on the training dataset:")
    print("   Number of different TRT cells:      %s" % len(np.unique(samples_df["traj_ID"])))
    print("   Number of different time steps:     %s" % len(np.unique(samples_df["date"])))
    print("   Number of TRT cells with rank >= 1: %s\n" % sum(samples_df["RANKr"]>=10))
    
## Change and append some of the TRT cell values or append additional ones:
def change_append_TRT_cell_info(cfg_set_tds):    
    """Correct and append some information to TRT cell info."""
    
    print("Enhance and correct information of TRT cells within time period.")
    samples_df = pd.read_pickle(os.path.join(cfg_set_tds["root_path_tds"],
                                u"TRT_sampling_df_testset.pkl"))
        
    ## Change datatypes
    ## Old:
    #samples_df["lat"]   = samples_df["lat"].astype(np.float32)
    #samples_df["lon"]   = samples_df["lon"].astype(np.float32)
    #samples_df["RANKr"] = samples_df["RANKr"].astype(np.uint8)
    #samples_df["area"]  = samples_df["area"].astype(np.uint32)
    #samples_df["jCH"]   = samples_df["jCH"].astype(np.uint32)
    #samples_df["iCH"]   = samples_df["iCH"].astype(np.uint32)
    
    ## Assign LV03 coordinates
    samples_df["LV03_x"], samples_df["LV03_y"] = lonlat2xy(samples_df["lon"].tolist(),
                                                           samples_df["lat"].tolist())
    samples_df["LV03_x"] = samples_df["LV03_x"].astype(np.float32)
    samples_df["LV03_y"] = samples_df["LV03_y"].astype(np.float32)
    
    ## Assign TRT categories
    samples_df["category"] = "WEAK"                                                       
    samples_df["category"].loc[(samples_df["RANKr"] >= 12) & (samples_df["RANKr"] < 15)]  = "DEVELOPING"                                                       
    samples_df["category"].loc[(samples_df["RANKr"] >= 15) & (samples_df["RANKr"] < 25)]  = "MODERATE"                                                       
    samples_df["category"].loc[(samples_df["RANKr"] >= 25) & (samples_df["RANKr"] < 35)]  = "SEVERE"                                                       
    samples_df["category"].loc[(samples_df["RANKr"] >= 35) & (samples_df["RANKr"] <= 40)] = "VERY SEVERE"          
    category_type = CategoricalDtype(categories=["WEAK","DEVELOPING","MODERATE","SEVERE","VERY SEVERE"],
                                              ordered=True)
    
    ## Save new dataset
    samples_df.to_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"TRT_sampling_df_testset_enhanced.pkl"))
    print("   Dataframe saved in: %s%s" % (cfg_set_tds["root_path_tds"],"TRT_sampling_df_testset_enhanced.pkl"))

## Plot some of the TRT data::
def exploit_TRT_cell_info(cfg_set_tds):
    """Exploit information of TRT cells within time period."""
        
    print("Exploit information of TRT cells within time period.")
    samples_df = pd.read_pickle(os.path.join(cfg_set_tds["root_path_tds"],
                                u"TRT_sampling_df_testset_enhanced.pkl"))
    
    ## Print histograms:
    print_TRT_cell_histograms(samples_df,cfg_set_tds)
    
    ## Print map of cells:
    print_TRT_cell_map(samples_df,cfg_set_tds)
    
## Set up the trainings datasets (DEPRECATED):
def get_empty_tds(cfg_set_tds, cfg_set_input):
    """Create empty xarray field in which to store the statistics (DEPRECATED)."""
    
    ## Check whether file is already available, if so, abort:
    if os.path.isfile("%s%s" % (cfg_set_tds["root_path_tds"],"Training_data_3d.nc")):
        print("File 'Training_data_3d.pkl' exists already and is not overwritten!\n")
        return
    else:
        print("Create xarray dataset for trainings:")
    
    ## Read in samples dataframe:
    samples_df = pd.read_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"TRT_sampling_df_testset_enhanced.pkl"))
    samples_df_subset = samples_df.loc[samples_df["RANKr"] >= cfg_set_input["min_TRT_rank"]]
    
    ## Get values of coordinates:
    statistics_coord = cfg_set_input["stat_list"] #cfg_var.columns[np.where(cfg_var.columns=="SUM")[0][0]:-1]
    pixcount_coord   = cfg_set_input["pixcount_list"] #cfg_var.columns[np.where(cfg_var.columns=="SUM")[0][0]:-1]
    time_delta_coord = np.arange(-cfg_set_input["n_integ"]+1,cfg_set_input["n_integ"],dtype=np.int16)*cfg_set_input["timestep"]
    TRT_ID_coord     = samples_df_subset["traj_ID"].values
    
    ## Decide and date type for count data:
    if (cfg_set_input["stat_sel_form"]=="circle" and cfg_set_input["stat_sel_form_width"]>18) or \
       (cfg_set_input["stat_sel_form"]=="square" and cfg_set_input["stat_sel_form_width"]>15):
        print("  Form for pixel selection: %s with %skm width" % (cfg_set_input["stat_sel_form"],
                                                                  cfg_set_input["stat_sel_form_width"]))
        print(cfg_set_input["stat_sel_form_width"],cfg_set_input["stat_sel_form_width"]>18)
        dtype_pixc = np.uint16
        fill_value = 2**16-1
        print("    use 'np.uint16' for pixel count data type (fill value: %s)" % fill_value)
    else:
        print("  Form for pixel selection: %s with %skm width" % (cfg_set_input["stat_sel_form"],
                                                                  cfg_set_input["stat_sel_form_width"]))
        dtype_pixc = np.uint8
        fill_value = 2**8-1
        print("    use 'np.uint8' for pixel count data type (fill value: %s)" % fill_value)
    
    ## Establish xarray:
    xr_tds = xr.Dataset()
    var_list = cfg_set_input["var_list"] #cfg_var["VARIABLE"][cfg_var["PREDICTOR"]].tolist()
    for var in var_list:
        empty_array_stat = np.zeros((len(TRT_ID_coord),len(time_delta_coord),len(statistics_coord)),dtype=np.float32)*np.nan
        empty_array_pixc = np.zeros((len(TRT_ID_coord),len(time_delta_coord),len(pixcount_coord)),dtype=dtype_pixc)+fill_value
        xr_tds[var+"_stat"] = (('TRT_ID', 'time_delta', 'statistic'), empty_array_stat)
        xr_tds[var+"_pixc"] = (('TRT_ID', 'time_delta', 'pixel_count'), empty_array_pixc)
    xr_tds["time"] = (('TRT_ID'), samples_df_subset["date"].values)
    xr_tds.coords['TRT_ID']        = TRT_ID_coord
    xr_tds.coords['time_delta']    = time_delta_coord
    xr_tds.coords['statistic']     = statistics_coord
    xr_tds.coords['pixel_count']  = pixcount_coord
    
    ## Save array:
    print("   Created the follwoing xarray:\n")
    print(xr_tds)
    xr_tds.to_netcdf("%s%s" % (cfg_set_tds["root_path_tds"],"Training_data_3d.nc"))
    print("\n   xarray saved in: %s%s" % (cfg_set_tds["root_path_tds"],"Training_data_3d.nc"))    










