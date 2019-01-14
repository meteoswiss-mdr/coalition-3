""" Functions for NOSTRADAMUS_0_training_ds.py:

The following functions are assembled here for the generation of the
training dataset for NOSTRADAMUS:
"""

from __future__ import division
from __future__ import print_function

import sys
import ast
import configparser
import datetime
import matplotlib.pylab as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import os
import pysteps as st
import pdb
from netCDF4 import Dataset
import warnings
from scipy import ndimage
from PIL import Image
import shapefile
from glob import glob


sys.path.insert(0, '/opt/users/jmz/monti-pytroll/packages/mpop')
from mpop.satin import swisstrt
import NOSTRADAMUS_1_input_prep_fun as Nip


## ===============================================================================
## FUNCTIONS:

## Get settings from config file and some additional static settings:
def get_config_info(CONFIG_PATH_set_train,CONFIG_FILE_set_train):
    """Get information from configuration file and return in a dictionary "cfg_set".
    Furthermore, get data from variable configuration file (cfg_var.csv) and return
    in a pandas.dataframe "cfg_var"

    Parameters
    ----------
    
    CONFIG_PATH : str
        Path to config file.
        
    CONFIG_FILE_set : str
        Name of settings config file.
        
    Output
    ------

    cfg_set_tds : dict
        Dictionary with the basic variables used throughout the code
     
    """
    
    ## ===== Read the data source/output configuration file: ============
    config = configparser.RawConfigParser()
    config.read("%s%s" % (CONFIG_PATH_set_train,CONFIG_FILE_set_train))

    config_ds = config["datasource"]
    root_path_tds               = config_ds["root_path_tds"]
    CONFIG_PATH_set_input       = config_ds["CONFIG_PATH_set_input"]
    CONFIG_PATH_var_input       = config_ds["CONFIG_PATH_var_input"]
    CONFIG_FILE_set_input       = config_ds["CONFIG_FILE_set_input"]
    CONFIG_FILE_var_input       = config_ds["CONFIG_FILE_var_input"]
    CONFIG_FILE_var_combi_input = config_ds["CONFIG_FILE_var_combi_input"]
    PATH_stat_output            = config_ds["PATH_stat_output"]
    PATH_stdout_output          = config_ds["PATH_stdout_output"]

    ## Read further config information training dataset
    config_bs = config["basicsetting"]
    tds_period_start    = datetime.datetime.strptime(config_bs["tds_period_start"], "%Y%m%d").date()
    tds_period_end      = datetime.datetime.strptime(config_bs["tds_period_end"], "%Y%m%d").date()
    dt_samples          = int(config_bs["dt_samples"])
    dt_daily_shift      = int(config_bs["dt_daily_shift"])
        
    ## Get information on date and time
    tds_period_start_doy  = tds_period_start.timetuple().tm_yday
    tds_period_end_doy    = tds_period_end.timetuple().tm_yday
    
    ## Save key input variables in dictionary
    cfg_set_tds = {
                   "root_path_tds":               root_path_tds,
                   "tds_period_start":            tds_period_start,
                   "tds_period_end":              tds_period_end,
                   "dt_samples":                  dt_samples,
                   "dt_daily_shift":              dt_daily_shift,
                   "tds_period_start_doy":        tds_period_start_doy,
                   "tds_period_end_doy":          tds_period_end_doy,
                   "CONFIG_PATH_set_train":       CONFIG_PATH_set_train,
                   "CONFIG_FILE_set_train":       CONFIG_FILE_set_train,
                   "CONFIG_PATH_set_input":       CONFIG_PATH_set_input,
                   "CONFIG_PATH_var_input":       CONFIG_PATH_var_input,
                   "CONFIG_FILE_set_input":       CONFIG_FILE_set_input,
                   "CONFIG_FILE_var_input":       CONFIG_FILE_var_input,
                   "CONFIG_FILE_var_combi_input": CONFIG_FILE_var_combi_input,
                   "PATH_stat_output":            PATH_stat_output,
                   "PATH_stdout_output":          PATH_stdout_output
                  }
 
    return(cfg_set_tds)  
    
## Print information before running script:
def print_config_info(cfg_set_tds,CONFIG_FILE_set):
    """Print information before running script"""
    print("\n-------------------------------------------------------------------------------------------------------\n")
    print_str = '    Configuration of NOSTRADAMUS training dataset preparation procedure:'+ \
    '\n      Config files:     '+CONFIG_FILE_set+' (Settings)'+ \
    '\n      Date range:       '+cfg_set_tds["tds_period_start"].strftime("%Y-%m-%d")+' to '+cfg_set_tds["tds_period_end"].strftime("%Y-%m-%d")+ \
    '\n      dt of samples:    '+str(cfg_set_tds["dt_samples"])+"min"+ \
    '\n      dt change:        '+str(cfg_set_tds["dt_daily_shift"])+"min per day \n"
    print(print_str)
    print("-------------------------------------------------------------------------------------------------------\n")
    
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

## Check the availability of input datasets during the complete testdata period:
def create_df_missing(cfg_set_tds,cfg_set_input,cfg_var,check_sources):
    """Check the availability of input datasets during the complete testdata period.
    """
    print("Check the availability of input datasets during the complete testdata period")
    if check_sources!=check_sources == ["RADAR","SEVIRI","COSMO_CONV","THX"]:
        raise NotImplementedError("Other check_sources not yet implemented.")
    
    ## Append further datetime objects accroding to settings in cfg_set_tds:
    dt_start = datetime.datetime.combine(cfg_set_tds["tds_period_start"],datetime.time(0,0))
    dt_end   = datetime.datetime.combine(cfg_set_tds["tds_period_end"]+datetime.timedelta(days=1),datetime.time(0,0))
    
    dt_complete_list = pd.date_range(dt_start,dt_end,freq="5min")
    RADAR_vars = cfg_var.loc[cfg_var["SOURCE"]=="RADAR","VARIABLE"].values[:-1]
    columns_list = np.concatenate([RADAR_vars,check_sources[1:]])
    
    bool_array = np.full((len(dt_complete_list), len(columns_list)), False, dtype=np.bool)
    df_missing = pd.DataFrame(bool_array,index=dt_complete_list,columns=columns_list)
    
    cfg_set_input["n_past_frames"]=0
    for sampling_time in dt_complete_list:
        perc_checked = np.round((sampling_time.hour*60+sampling_time.minute)/1440.,2)*100
        print("  Check input data availability of date: %s - %02d%%" % (sampling_time.strftime("%d.%m.%Y"),perc_checked), end='\r')
        
        for RADAR_var in RADAR_vars:
            if Nip.path_creator(sampling_time, RADAR_var, "RADAR", cfg_set_input)[0][0] is None:
                df_missing.loc[sampling_time,RADAR_var] = True
        if Nip.path_creator(sampling_time, "IR_108", "SEVIRI", cfg_set_input)[0][0] is None:
            df_missing.loc[sampling_time,"SEVIRI"] = True
        if not os.path.exists(Nip.path_creator(sampling_time, "POT_VORTIC_70000", "COSMO_CONV", cfg_set_input)[0]):
            df_missing.loc[sampling_time,"COSMO_CONV"] = True
        if Nip.path_creator(sampling_time, "THX_abs", "THX", cfg_set_input)[0][0] is None:
            df_missing.loc[sampling_time,"THX"] = True
    
    df_missing.to_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"Missing_InputData.pkl"))
    print("Save dataframe to %s%s" % (cfg_set_tds["root_path_tds"],"Missing_InputData.pkl"))
    return df_missing

## Analyse missing input data datasets:
def analyse_df_missing(cfg_set_tds,cfg_set_input,cfg_var,check_sources):
    """Analyse missing input data datasets.
    """
    print("Analyse missing input data datasets")
    if check_sources!=check_sources == ["RADAR","SEVIRI","COSMO_CONV","THX"]:
        raise NotImplementedError("Other check_sources not yet implemented.")
    
    RADAR_vars = cfg_var.loc[cfg_var["SOURCE"]=="RADAR","VARIABLE"].values[:-1]
    columns_list = np.concatenate([RADAR_vars,check_sources[1:]])  
    
    print("Number off missing input datasets:")
    df_missing = pd.read_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"Missing_InputData.pkl"))
    for var in columns_list: print("  %s: %s" % (var,np.sum(df_missing[var])))
    
    df_missing_SEVIRI = df_missing.loc[df_missing["SEVIRI"]==True,"SEVIRI"]
    df_missing_COSMO  = df_missing.loc[df_missing["COSMO_CONV"]==True,"SEVIRI"]

    df_missing_SEVIRI.groupby(df_missing_SEVIRI.index.month).count().plot(kind="bar")
    plt.show()
    df_missing_COSMO.groupby(df_missing_COSMO.index.month).count().plot(kind="bar")
    plt.show()
    
    df_missing_SEVIRI["datetime"] = df_missing_SEVIRI.index
    df_missing_COSMO["datetime"]  = df_missing_COSMO.index
    
    df_missing_COSMO_dates = df_missing_COSMO["datetime"].values
    print("Dates with missing COSMO data:")
    print(np.unique([pd.to_datetime(date).replace(hour=0,minute=0) for date in df_missing_COSMO_dates]))
    
    df_missing.loc[df_missing["SEVIRI"]==True,"SEVIRI"].to_csv("%s%s" % (cfg_set_tds["root_path_tds"],"Missing_InputData_SEVIRI.csv"),
                                                               index_label=True,index=True,sep=";")
    df_missing.loc[df_missing["COSMO_CONV"]==True,"COSMO_CONV"].to_csv("%s%s" % (cfg_set_tds["root_path_tds"],"Missing_InputData_COSMO_CONV.csv"),
                                                               index_label=True,index=True,sep=";")
    
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
        cfg_set_input, cfg_var = Nip.get_config_info(cfg_set_tds["CONFIG_PATH_set_input"],
                                                     cfg_set_tds["input_set_cfg"],
                                                     cfg_set_tds["input_var_cfg"],
                                                     dt_sampling_list[0].strftime("%Y%m%d%H%M"))
                               
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
        filepaths, timestamps = Nip.path_creator(sampling_time, "TRT", "TRT", cfg_set_input)
        
        ## In case file is not available, look for files just right before and after this timepoint
        ## (e.g. if no file available at 16:35, look at 16:25/16:30/16:40/16:45), otherwise skip this time point.
        if filepaths[0] is None:
            for dt_daily_shift_fac in [-1,1,-2,2]:
                sampling_time_temp = sampling_time + dt_daily_shift_fac*datetime.timedelta(minutes=cfg_set_tds["dt_daily_shift"])
                filepaths_temp, timestamps = Nip.path_creator(sampling_time_temp, "TRT", "TRT", cfg_set_input)
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
    samples_df.to_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"TRT_sampling_df_testset.pkl"))
    print("   Dataframe saved in: %s%s" % (cfg_set_tds["root_path_tds"],"TRT_sampling_df_testset.pkl"))
    return(samples_df)

## Print some basic information on the TRT cells which will be sampled:
def print_basic_info(cfg_set_tds):
    samples_df = pd.read_pickle("%s%s" % (cfg_set_tds["root_path_tds"],
                                          "TRT_sampling_df_testset_enhanced.pkl"))
    print("Basic information on the training dataset:")
    print("   Number of different TRT cells:      %s" % len(np.unique(samples_df["traj_ID"])))
    print("   Number of different time steps:     %s" % len(np.unique(samples_df["date"])))
    print("   Number of TRT cells with rank >= 1: %s\n" % sum(samples_df["RANKr"]>=10))
    
## Change and append some of the TRT cell values or append additional ones:
def change_append_TRT_cell_info(cfg_set_tds):    
    """Correct and append some information to TRT cell info."""
    from pandas.api.types import CategoricalDtype
    
    print("Enhance and correct information of TRT cells within time period.")
    samples_df = pd.read_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"TRT_sampling_df_testset.pkl"))
        
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

## Make log file which is containing information on training dataset generation:
def setup_log_file(cfg_set_tds,cfg_set_input):    
    """Make log file which is containing information on training dataset generation."""
    
    print("Create file containing processing status of training dataset generation.")
    samples_df = pd.read_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"TRT_sampling_df_testset_enhanced_5min.pkl"))
    samples_df = samples_df.reset_index(drop=True)
        
    ## Only keep RANKs bigger than minimum value:
    samples_df = samples_df.loc[samples_df["RANKr"]>=cfg_set_input["min_TRT_rank"]*10]
    
    ## Add columns indicating whether time step has already been processed, 
    ## is currently being processed, and whether all input data is available.
    samples_df["Processing"]          = False 
    samples_df["Processing_Start"]    = None 
    samples_df["Processing_End"]      = None 
    samples_df["Processed"]           = False 
    samples_df["COSMO_CONV_missing"]  = False 
    samples_df["SEVIRI_missing"]      = False 
    samples_df["RADAR_missing"]       = False 
    samples_df["THX_missing"]         = False 
    samples_df["Border_cell"]         = False 
    samples_df["Non_NAN_stat"]        = False 
    
    ## Check input data availability:
    for counter, date_i in enumerate(np.unique(samples_df["date"])):
        perc_complete = 100*int(counter)/float(len(np.unique(samples_df["date"])))
        str_print = "  Completed %02d%% of dates" % (perc_complete)
        print('\r',str_print,end='') #, end='\r'
        date_i_dt = datetime.datetime.strptime(date_i,"%Y%m%d%H%M")
        #date_i_dt = datetime.datetime.utcfromtimestamp(date_i.astype(int) * 1e-9)
        samples_df, not_avail = check_input_data_availability(samples_df,date_i_dt,cfg_set_input,cfg_set_tds)
        #samples_df, not_avail = check_input_data_availability(samples_df,date_i_dt,None,None,
        #                                                      "/opt/users/jmz/0_training_NOSTRADAMUS_ANN/Missing_InputData.pkl",
        #                                                      10,5)
        if not_avail:
            samples_df.loc[samples_df["date"]==date_i, "Processed"] = True
    
    ## Add TRT/Date ID:
    #samples_df["DATE_TRT_ID"]    =  np.array([date.strftime("%Y%m%d%H%M")+"_"+ID for date, ID in
    #                                          zip(samples_df["date"].astype(datetime).values,samples_df["traj_ID"].values)],
    #                                         dtype=np.object)
    samples_df["DATE_TRT_ID"] =  np.array([date+"_"+ID for date, ID in zip(samples_df["date"].values,samples_df["traj_ID"].values)],
                                          dtype=np.object)
    samples_df["date"]        =  [datetime.datetime.strptime(date,"%Y%m%d%H%M") for date in samples_df["date"].values]
    
    ## Save new dataset
    samples_df.to_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"Training_Dataset_Processing_Status.pkl"))
    print("\n   Dataframe saved in: %s%s" % (cfg_set_tds["root_path_tds"],"Training_Dataset_Processing_Status.pkl"))
        
## Select only dates which have cells with TRT rank larger than threshold:
def select_high_TRTrank_dates(log_file_path, new_filename, TRT_treshold):
    """From log-file, select only time points which contain a TRT cell higher than a specific value"""
    print("Only keep time points with TRT cells above rank %s" % TRT_threshold)
    
    df = pd.read_pickle(log_file_path)
    dates_high_rank = np.unique(df["date"].loc[df["RANKr"]>=TRT_treshold*10.])
    df_high_rank    = df_merge.iloc[np.in1d(df["date"], dates_high_rank)]

    n_processed_TRT = np.sum(df_high_rank["Processed"])
    tot_TRT         = df_high_rank.shape[0]
    print("  Percentage of already processed TRT cells:  %02d%%" % (100.*n_processed_TRT/tot_TRT))

    n_processed_dt  = len(np.unique(df_high_rank["date"].loc[df_high_rank["Processed"]]))
    tot_dt          = len(np.unique(df_high_rank["date"]))
    print("  Percentage of already processed time poins: %02d%%" % (100.*n_processed_dt/tot_dt))

    df_high_rank.to_pickle(new_filename)
    print("  Log-file dataframe with restricted dates saved in: %s" % (new_filename))
    
## Copy entries from old log_file into new log_file:
def copy_log_file_entries(file_old_path, file_new_path, filename_merge):
    """Copy entries from old log_file into new log_file. New log_file dateframe must contain old
    dataframe columns (DATE_TRT_ID, Processing, Processed, Processing_Start, Processing_End,
    COSMO_CONV_missing, SEVIRI_missing, RADAR_missing, THX_missing, Border_cell, Non_NAN_stat)"""
    
    df_old = pd.read_pickle(file_old_path)
    df_new = pd.read_pickle(file_new_path)

    df_old.Processing_Start = df_old.Processing_Start.astype(np.object)
    df_old.Processing_End   = df_old.Processing_End.astype(np.object)

    column_ls = ["Processing", "Processed", "Processing_Start", "Processing_End", \
                 "COSMO_CONV_missing", "SEVIRI_missing", "RADAR_missing", "THX_missing", "Border_cell", "Non_NAN_stat"]

    df_old_red = df_old[column_ls+["DATE_TRT_ID"]]
    df_new_red = df_new.drop(column_ls,axis=1)
    df_merge   = pd.merge(df_new_red,df_old_red,on="DATE_TRT_ID",how="left")
    for column_name in ["Processing", "Processed", "COSMO_CONV_missing", "SEVIRI_missing", \
                        "RADAR_missing", "THX_missing", "Border_cell", "Non_NAN_stat"]:
        df_merge[column_name].fillna(False,inplace=True)

    df_merge.to_pickle(filename_merge)
    print("   Merged log-file dataframe saved in: %s" % (filename_merge))
    
    """    
    for DATE_TRT_ID in df_old["DATE_TRT_ID"].values:
        #print_str = "Working on DATE_TRT_ID: %s" % DATE_TRT_ID
        print_str = "Processed %02d%%" % (100.*np.where(df_old["DATE_TRT_ID"].values==DATE_TRT_ID)[0][0]/len(df_old["DATE_TRT_ID"].values))
        print(print_str, end="\r")
        
        
        if DATE_TRT_ID not in df_new["DATE_TRT_ID"].values:
            #raise ValueError("DATE_TRT_ID not found in new dataframe")
            missing_DATE_TRT_ID_ls.append(DATE_TRT_ID)
        else:
            for column in column_ls:
                df_new[column].loc[df_new["DATE_TRT_ID"]==DATE_TRT_ID] = df_old[column].loc[df_old["DATE_TRT_ID"]==DATE_TRT_ID].values
    """

## Function which checks whether the input data at the respective time point is available:
def check_input_data_availability(samples_df,time_point,cfg_set_input,cfg_set_tds,
                                  missing_date_df_path=None,n_integ=None,timestep=None):
                                  
    if missing_date_df_path is None:
        missing_date_df_path = "%s%s" % (cfg_set_tds["root_path_tds"],"Missing_InputData.pkl")
    if n_integ is None:
        n_integ = cfg_set_input["n_integ"]
    if timestep is None:
        timestep = cfg_set_input["timestep"]
    
    dates_of_needed_input_data = pd.date_range(time_point-n_integ*datetime.timedelta(minutes=timestep),
                                               time_point+n_integ*datetime.timedelta(minutes=timestep),
                                               freq=str(timestep)+'Min')
    with open(missing_date_df_path, "rb") as path: missing_date_df = pickle.load(path)
    #missing_date_df = pd.read_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"Missing_InputData.pkl"))    
    t_ind = (missing_date_df.index >= dates_of_needed_input_data[0]) & \
            (missing_date_df.index <= dates_of_needed_input_data[-1])

    missing_COSMO_CONV = np.any(missing_date_df.loc[t_ind,"COSMO_CONV"])
    missing_SEVIRI     = np.any(missing_date_df.loc[t_ind,"SEVIRI"])
    missing_THX        = np.any(missing_date_df.loc[t_ind,"THX"])
    missing_RADAR      = np.any(missing_date_df.loc[t_ind,["RZC","BZC","LZC","MZC","EZC15","EZC20","EZC45","EZC50","CZC"]])  
            
    samples_df.loc[samples_df["date"]==time_point, "COSMO_CONV_missing"]  = missing_COSMO_CONV
    samples_df.loc[samples_df["date"]==time_point, "SEVIRI_missing"]      = missing_SEVIRI
    samples_df.loc[samples_df["date"]==time_point, "THX_missing"]         = missing_THX
    samples_df.loc[samples_df["date"]==time_point, "RADAR_missing"]       = missing_RADAR
    
    not_all_input_data_available = np.any([missing_COSMO_CONV,
                                           missing_SEVIRI,
                                           missing_THX,
                                           missing_RADAR])
    return(samples_df, not_all_input_data_available)

    
## Read and edit log file which is containing information on training dataset generation:
def read_edit_log_file(cfg_set_tds,cfg_set_input,process_point,t0_object=None,log_file=None,
                       samples_df=None,check_input_data=True):    
    """Read and edit log file which is containing information on training dataset generation."""
    
    print("Reading training dataset generation log file at %s of input generation process" % process_point)
    if log_file is None and samples_df is None:
        log_file_path = "%s%s" % (cfg_set_tds["root_path_tds"],"Training_Dataset_Processing_Status.pkl")
        with open(log_file_path, "rb") as path: samples_df = pickle.load(path)
    elif samples_df is None: 
        log_file_path = log_file
        with open(log_file_path, "rb") as path: samples_df = pickle.load(path)
    #samples_df = pd.read_pickle(log_file_path)
    
    if process_point=="start":
        ## Check all time steps which have not been processed yet and are not currently processed:
        samples_df_subset = samples_df.loc[(samples_df["Processed"].values==False) & \
                                           (samples_df["Processing"].values==False)]
        
        ## (Unfinished) Read list of time-points which are currently processed:
        #try:
        #    file_dates_currently_processing = "%s%s" % (cfg_set_tds["root_path_tds"],"Training_Dataset_Processing_Dates.npy")
        #    np.load(file_dates_currently_processing)
        #except IOError:
                    
        ## Pick random time step:
        if len(np.unique(samples_df_subset["date"]))==0:
            return_value = 200
            return(return_value)
        chosen_date = np.random.choice(np.unique(samples_df_subset["date"]))
        chosen_date = datetime.datetime.utcfromtimestamp(chosen_date.astype(int) * 1e-9)

        ## Set process starting time to current time:
        samples_df.loc[samples_df["date"]==chosen_date, "Processing_Start"] = datetime.datetime.now()
                                                   
        ## Check whether SEVIRI and COSMO input data is available:
        if check_input_data:
            samples_df, not_avail = check_input_data_availability(samples_df,chosen_date,cfg_set_input,cfg_set_tds)

        if not_avail:
            ## If any input data is missing, skip this time point and set 'Processed' to True:
            samples_df.loc[samples_df["date"]==chosen_date, "Processing_End"] = datetime.datetime.now()
            samples_df.loc[samples_df["date"]==chosen_date, "Processed"] = True
            #samples_df.to_pickle(log_file_path) 
            #with open(log_file_path, "wb") as output_file: pickle.dump(samples_df, output_file, protocol=-1)
            print("   For timepoint %s not all input data exists, skip this timepoint\n" % chosen_date.strftime("%d.%m.%Y %H:%M"))         
            return(None, samples_df)
        else:
            ## If all input data is available, actually start process (setting 'Processing' to True)
            ## by returning the respective time point:
            samples_df.loc[samples_df["date"]==chosen_date, "Processing"] = True
            #samples_df.to_pickle(log_file_path)
            with open(log_file_path, "wb") as output_file: pickle.dump(samples_df, output_file, protocol=-1)
            print("   Determine statistics for timepoint %s\n" % chosen_date.strftime("%d.%m.%Y %H:%M")) 
            return(chosen_date, None)
    
    elif process_point=="end":
        ## Note end-time of process, set 'Processing' back to False but 'Processed' to True:
        samples_df.loc[samples_df["date"]==t0_object, "Processing_End"] = datetime.datetime.now()
        samples_df.loc[samples_df["date"]==t0_object, "Processed"]      = True
        samples_df.loc[samples_df["date"]==t0_object, "Processing"]     = False
        print("\n",samples_df.loc[samples_df["date"]==t0_object],"\n")
        
        ## Delete files in /tmp directory of time steps where processing started more than 2h ago:
        dates_old = samples_df.loc[samples_df["Processing_End"]<datetime.datetime.now()-datetime.timedelta(hours=2),["Processing_Start"]]
        dates_old_unique_str = np.unique(dates_old.iloc[:,0].dt.strftime('%Y%m%d%H%M'))
        for date_str in dates_old_unique_str[:-1]: clean_disparr_vararr_tmp(cfg_set_input,fix_t0_str=date_str)
        
        ## Save the log file as pickle and as csv:
        with open(log_file_path, "wb") as output_file: pickle.dump(samples_df, output_file, protocol=-1)
        #samples_df.to_pickle(log_file_path)
        samples_df.to_csv(log_file_path[:-3]+"csv")

        ## Return percentage of processed files plus 100:
        print("   Finished process for timepoint %s." % t0_object.strftime("%d.%m.%Y %H:%M"))
        return_value = int(100.*np.sum(samples_df["Processed"])/len(samples_df["Processed"]))+100
        return(return_value)
    else: raise ValueError("process_point argument must either be 'start' or 'end'")
    
    
## Plot some of the TRT data::
def exploit_TRT_cell_info(cfg_set_tds):
    """Exploit information of TRT cells within time period."""
        
    print("Exploit information of TRT cells within time period.")
    samples_df = pd.read_pickle("%s%s" % (cfg_set_tds["root_path_tds"],"TRT_sampling_df_testset_enhanced.pkl"))
    
    ## Print histograms:
    print_TRT_cell_histograms(samples_df,cfg_set_tds)
    
    ## Print map of cells:
    print_TRT_cell_map(samples_df,cfg_set_tds)
    
## Print map of TRT cells:
def print_TRT_cell_map(samples_df,cfg_set_tds):
    """Print map of TRT cells."""
    ## Load DEM and Swiss borders
    
    shp_path = "%s%s" % (cfg_set_tds["CONFIG_PATH_set_train"],"Shapefile_and_DTM/CHE_adm0.shp")
    #shp_path = "%s%s" % (cfg_set_tds["CONFIG_PATH_set_train"],"Shapefile_and_DTM/CCS4_merged_proj_clip_G05_countries.shp")
    dem_path = "%s%s" % (cfg_set_tds["CONFIG_PATH_set_train"],"Shapefile_and_DTM/ccs4.png")

    dem = Image.open(dem_path)
    dem = np.array(dem.convert('P'))

    sf = shapefile.Reader(shp_path)
    # for shape in sf.shapeRecords():
        # x = [i[0] for i in shape.shape.points[:]]
        # y = [i[1] for i in shape.shape.points[:]]
        # plt.plot(x,y)
       
    fig_map, axes = plt.subplots(1, 1)
    fig_map.set_size_inches(12, 12)
    axes.imshow(dem, extent=(255000,965000,-160000,480000), cmap='gray')
        
    ## Plot in swiss coordinates (radar CCS4 in LV03 coordinates)
    for shape in sf.shapeRecords():
        lon = [i[0] for i in shape.shape.points[:]]
        lat = [i[1] for i in shape.shape.points[:]]
        
        ## Convert to swiss coordinates
        x,y = lonlat2xy(lon, lat)
        #x = lon
        #y = lat
        axes.plot(x,y,color='b',linewidth=1)
        
    ## Convert lat/lon to Swiss coordinates:
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "DEVELOPING"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "DEVELOPING"],c='w',edgecolor=(.7,.7,.7),s=18)
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "MODERATE"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "MODERATE"],c='g',edgecolor=(.7,.7,.7),s=22)
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "SEVERE"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "SEVERE"],c='y',edgecolor=(.7,.7,.7),s=26)
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "VERY SEVERE"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "VERY SEVERE"],c='r',edgecolor=(.7,.7,.7),s=30)
    axes.set_xlim([255000,965000])
    axes.set_ylim([-160000,480000])
    fig_map.savefig("%s%s" % (cfg_set_tds["CONFIG_PATH_set_train"],"Map.pdf"))

## Convert lat/lon-values in decimals to values in seconds:
def dec2sec(angles):
    """Convert lat/lon-values in decimals to values in seconds.
    
    Parameters
    ----------
    
    angles : list of floats
        Location coordinates in decimals.   
    """
    angles_ = np.zeros_like(angles)
    for i in range(len(angles)):
        angle = angles[i]
        ## Extract dms
        deg = float(str(angle).split(".")[0])
        min = float(str((angle - deg)*60.).split(".")[0])
        sec = (((angle - deg)*60.) - min)*60.
        angles_[i] = sec + min*60. + deg*3600.
    return angles_
    
## Convert lat/lon-values (in seconds) into LV03 coordinates:
def lonlat2xy(s_lon, s_lat): # x: easting, y: northing
    """Convert lat/lon-values (in seconds) into LV03 coordinates.
    
    Parameters
    ----------
    
    s_lon, s_lat : float
        Lat/Lon locations in seconds (not decimals!).   
    """
    # convert decimals to seconds...
    s_lon = dec2sec(s_lon)
    s_lat = dec2sec(s_lat)

    ## Auxiliary values 
    # i.e. differences of latitude and longitude relative to Bern in the unit [10000'']
    s_lng_aux = (s_lon - 26782.5)/10000.
    s_lat_aux = (s_lat - 169028.66)/10000.
    
    # easting
    s_x =   (600072.37 
        +  211455.93*s_lng_aux 
        -   10938.51*s_lng_aux*s_lat_aux 
        -       0.36*s_lng_aux*(s_lat_aux**2)  
        -      44.54*(s_lng_aux**3))
    
    # northing
    s_y =   (200147.07 
        + 308807.95*s_lat_aux 
        +   3745.25*(s_lng_aux**2) 
        +     76.63*(s_lat_aux**2) 
        -    194.56*(s_lng_aux**2)*s_lat_aux 
        +    119.79*(s_lat_aux**3))

    return s_x, s_y

## Print histogram of TRT cell values:
def print_TRT_cell_histograms(samples_df,cfg_set_tds):
    """Print histograms of TRT cell information."""
    
    fig_hist, axes = plt.subplots(3, 2)
    fig_hist.set_size_inches(12, 15)

    ## Analyse distribution of ranks
    nw = np.sum(np.logical_and(samples_df["RANKr"]>=12, samples_df["RANKr"]<15))
    ng = np.sum(np.logical_and(samples_df["RANKr"]>=15, samples_df["RANKr"]<25))
    ny = np.sum(np.logical_and(samples_df["RANKr"]>=25, samples_df["RANKr"]<35))
    nr = np.sum(np.logical_and(samples_df["RANKr"]>=35, samples_df["RANKr"]<=40))
    print("  The number of Cells with TRT Rank w is: %s" % nw)
    print("  The number of Cells with TRT Rank g is: %s" % ng)
    print("  The number of Cells with TRT Rank y is: %s" % ny)
    print("  The number of Cells with TRT Rank r is: %s" % nr)
    samples_df["RANKr"] = samples_df["RANKr"]/10.
    pw = patches.Rectangle((1.2, 65000), 0.3, 10000, facecolor='w')
    pg = patches.Rectangle((1.5, 65000),   1, 10000, facecolor='g')
    py = patches.Rectangle((2.5, 65000),   1, 10000, facecolor='y')
    pr = patches.Rectangle((3.5, 65000), 0.5, 10000, facecolor='r')
    axes[0,0].add_patch(pw); axes[0,0].add_patch(pg); axes[0,0].add_patch(py); axes[0,0].add_patch(pr)
    axes[0,0].annotate(str(nw),(1.35,70000),(1.25,90500),ha='center',va='center',color='k',arrowprops={'arrowstyle':'->'}) #,arrowprops={arrowstyle='simple'}
    axes[0,0].annotate(str(ng),(2,70000),ha='center',va='center',color='w') 
    axes[0,0].annotate(str(ny),(3,70000),ha='center',va='center',color='w')
    axes[0,0].annotate(str(nr),(3.75,70000),ha='center',va='center',color='w') 
    samples_df["RANKr"].hist(ax=axes[0,0],bins=np.arange(0,4.25,0.25),facecolor=(.7,.7,.7),alpha=0.75,grid=True)
    axes[0,0].set_xlabel("TRT rank")
    axes[0,0].set_title("TRT Rank Distribution")
    
    samples_df["area"].hist(ax=axes[0,1],bins=np.arange(0,650,50),facecolor=(.7,.7,.7),alpha=0.75,grid=True)
    axes[0,1].set_xlabel("Cell Area [km$^2$]")
    axes[0,1].set_title("Cell Size Distribution")
    
    samples_df["date"] = samples_df["date"].astype(np.datetime64)
    
    samples_df["date"].groupby(samples_df["date"].dt.month).count().plot(kind="bar",ax=axes[1,0],facecolor=(.7,.7,.7),
                                                                         alpha=0.75,grid=True)
    #axes[1,0].set_xlabel("Months")
    axes[1,0].set_xlabel("")
    axes[1,0].set_xticklabels(["Apr","May","Jun","Jul","Aug","Sep"],rotation=45)
    axes[1,0].set_title("Monthly Number of Cells")

    samples_df["date"].groupby([samples_df["date"].dt.month,
                                samples_df["date"].dt.day]).count().plot(kind="bar",
                                ax=axes[1,1],facecolor=(.7,.7,.7),alpha=0.75,edgecolor=(.7,.7,.7),grid=True)
    axes[1,1].get_xaxis().set_ticks([])
    axes[1,1].set_xlabel("Days over period")
    axes[1,1].set_title("Daily Number of Cells")
    
    samples_df["date"].groupby(samples_df["date"]).count().hist(ax=axes[2,0],bins=np.arange(0,150,10),
                                                                facecolor=(.7,.7,.7),alpha=0.75,grid=True)
    axes[2,0].set_xlabel("Number of cells")
    axes[2,0].set_title("Number of cells per time step")
    
    #samples_df["date"].loc[samples_df["RANKr"]>=1].groupby(samples_df["date"]).count().hist(ax=axes[2,1],bins=np.arange(0,65,5),
    #                                                            facecolor=(.7,.7,.7),alpha=0.75,grid=True)
    #axes[2,1].set_xlabel("Number of cells")
    #axes[2,1].set_title("Number of cells (TRT Rank >= 1)\n per time step")
    axes[2,1].axis('off')
    
    fig_hist.savefig("%s%s" % (cfg_set_tds["CONFIG_PATH_set_train"],"Histogram.pdf"))

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

## Displace variables to current time step:
def displace_variables(cfg_set_input,cfg_var,reverse):
    """Displace past or future variables to current t0.

    Parameters
    ----------
    
    reverse : boolean
        Boolean value stating whether fields should be displaced from
        past to t0 (False) or from future to t0 (True).        
    """
    
    ## Change boolean cfg_set_input["future_disp_reverse"]:
    cfg_set = cfg_set_input.copy()
    print_reverse = "future" if reverse else "past"
    cfg_set["future_disp_reverse"] = True if reverse else False
    cfg_set["time_change_factor"]  = -1 if reverse else 1
    print("Displace %s observations to time step: %s" % (print_reverse,cfg_set["t0"].strftime("%Y-%m-%d %H:%M")))
    cfg_set["verbose"] = False
    
    ## Check whether precalculated displacement array is already existent, otherwise create it.
    ## If existent, replace with newest displacement
    if cfg_set["precalc_UV_disparr"]:
        t1 = datetime.datetime.now()
        Nip.check_create_precalc_disparray(cfg_set)
        t2 = datetime.datetime.now()
        print("  Elapsed time for creation of precalculated displacement array: "+str(t2-t1)+"\n")

    ## Check whether Displacement array is already existent, otherwise create it.
    ## If existent, replace with newest displacement
    t1 = datetime.datetime.now()
    Nip.check_create_disparray(cfg_set)
    t2 = datetime.datetime.now()
    print("  Elapsed time for creation of displacement array: "+str(t2-t1)+"\n")

    ## Create numpy arrays of variables for quick access
    t1 = datetime.datetime.now()
    Nip.create_new_vararray(cfg_set,cfg_var)
    t2 = datetime.datetime.now()
    print("  Elapsed time for creation of variable arrays: "+str(t2-t1)+"\n")
    
    ## Displace past fields onto current position, according to displacement array
    t1 = datetime.datetime.now()
    Nip.displace_fields(cfg_set)
    t2 = datetime.datetime.now()
    print("  Elapsed time for the displacement: "+str(t2-t1)+"\n")

    ## Correction of residual movements:
    if cfg_set["resid_disp"]:
        t1 = datetime.datetime.now()
        Nip.residual_disp(cfg_set)
        t2 = datetime.datetime.now()
        print("  Elapsed time for the correction of residual movements: "+str(t2-t1)+"\n")
    
    t1 = datetime.datetime.now()
    Nip.read_TRT_area_indices(cfg_set,cfg_set["future_disp_reverse"])
    t2 = datetime.datetime.now()
    print("  Elapsed time for reading of indices of the TRT domains: "+str(t2-t1)+"\n")

## Move statistics to collection directory (on non-temporary disk):
def move_statistics(cfg_set_input,cfg_set_tds,path_addon=""):
    import shutil
    path_in  = "%stmp/" % (cfg_set_input["root_path"])
    path_out = "%s%s" % (cfg_set_tds["PATH_stat_output"],path_addon)
    print("Move file with statistics to directory %s" % path_out)
    for file in os.listdir(path_in):
        if file.startswith(cfg_set_input["t0_str"]+"_stat_pixcount") and file.endswith(".pkl"):
            shutil.move(os.path.join(path_in, file),os.path.join(path_out, file))
        if file.startswith(cfg_set_input["t0_str"]+"_RZC_stat") and file.endswith(".pdf"):
            shutil.move(os.path.join(path_in, file),os.path.join(path_out, file))
    
## Repeat index and statistics reading for a different form width:
def change_form_width_statistics(factor,cfg_set_input,cfg_var,cfg_var_combi):
    """Repeat index and statistics reading for a different form width.

    Parameters
    ----------
    
    factor : float
        Factor of width change.       
    """
    
    ## Get new form width and write it to config file:
    new_form_width = np.round(cfg_set_input["stat_sel_form_width"]*factor)
    print("Change form width from %02dkm to %02dkm and repeat reading in statistics:" %
          (cfg_set_input["stat_sel_form_width"],new_form_width))
    cfg_set_input["stat_sel_form_width"] = new_form_width
    cfg_set_input["stat_sel_form_size"] = Nip.form_size(cfg_set_input["stat_sel_form_width"],
                                                        cfg_set_input["stat_sel_form"])
    
    ## Get new indices from which to read the statistics:
    t1 = datetime.datetime.now()
    Nip.read_TRT_area_indices(cfg_set_input,reverse=False)
    Nip.read_TRT_area_indices(cfg_set_input,reverse=True)
    t2 = datetime.datetime.now()
    print("  (Changed diameter) Elapsed time for reading of indices of the TRT domains: "+str(t2-t1)+"\n")

    ## Get the new statistics (at the new indices):
    t1 = datetime.datetime.now()
    Nip.append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=False)
    Nip.append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=True)
    t2 = datetime.datetime.now()
    print("  (Changed diameter) Elapsed time for reading the statistics / pixel counts: "+str(t2-t1)+"\n")

## Clean up displacement arrays in tmp directory:
def clean_disparr_vararr_tmp(cfg_set,fix_t0_str=None):
    path = "%stmp/" % (cfg_set["root_path"])
    #print("Remove all displacement files from temporary directory %s" % path)
    check_for_npz = cfg_set["save_type"]=="npy"
    
    t0_str = cfg_set["t0_str"] if fix_t0_str is None else fix_t0_str
    for file in os.listdir(path):
        ## Delete .nc/.npy files of the displacement
        if file.startswith(t0_str) and file.endswith(cfg_set["save_type"]):
            os.remove(os.path.join(path, file))
        ## Delete .pkl files (with TRT information)
        if file.startswith(t0_str+"_TRT_df") and file.endswith(".pkl"):
            os.remove(os.path.join(path, file))
        ## Delete .npz files
        if check_for_npz:
            if file.startswith(t0_str) and file.endswith("npz"):
                os.remove(os.path.join(path, file))

def get_stat_path(user_argv_path):
    """Function returning the path where the stat_pixc pickle files are stored"""
    if user_argv_path is not None:
        print("\nYou provided the following path to the '<Datetime>_stat_pixcount*.pkl' files:\n  %s" % user_argv_path)
        correct_path = raw_input("  Is this path correct? [y,n] ")
        while (correct_path!="n" and correct_path!="y"): correct_path = raw_input("  Is this path correct? [y,n] ")
        if correct_path=="y":
            if os.path.exists(user_argv_path):
                #print("It is assumed that in this directory only "+ \
                #      "'<Datetime>_stat_pixcount*.pkl' files are stored:\n  %s\n" % user_argv_path)
                return(os.path.normpath(user_argv_path))
            else: print("This path does not exist!")
            
    user_path = raw_input("\nPlease provide the path to the '<Datetime>_stat_pixcount*.pkl' files:\n  ")
    while not os.path.exists(user_path):
        print("This path does not exist: %s" % user_path)
        user_path = raw_input("\nPlease provide the path to the '<Datetime>_stat_pixcount*.pkl' files:\n  ")
    #print("It is assumed that in this directory only "+ \
    #      "'<Datetime>_stat_pixcount*.pkl' files are stored:\n  %s\n" % user_path)
    return(os.path.normpath(user_path))
                
def get_log_path(user_argv_path):
    """Function returning the path where the stat_pixc pickle files are stored"""
    if user_argv_path is not None:
        print("\nYou provided the following path to the 'Training_Dataset_Processing_Status.pkl' file:\n  %s" % user_argv_path)
        correct_path = raw_input("  Is this path correct? [y,n] ")
        while (correct_path!="n" and correct_path!="y"): correct_path = raw_input("  Is this path correct? [y,n] ")
        if correct_path=="y":
            if os.path.exists(user_argv_path):
                #print("It is assumed that in this directory only "+ \
                #      "'<Datetime>_stat_pixcount*.pkl' files are stored:\n  %s\n" % user_argv_path)
                return(os.path.normpath(user_argv_path))
            else: print("This path does not exist!")
            
    user_path = raw_input("\nPlease provide the path to the 'Training_Dataset_Processing_Status.pkl' file:\n  ")
    while not os.path.exists(user_path):
        print("This path does not exist: %s" % user_path)
        user_path = raw_input("\nPlease provide the path to the 'Training_Dataset_Processing_Status.pkl' file:\n  ")
    #print("It is assumed that in this directory only "+ \
    #      "'<Datetime>_stat_pixcount*.pkl' files are stored:\n  %s\n" % user_path)
    return(os.path.normpath(user_path))
                
def pickle2nc(pkl_pathfile,nc_path):
    nc_filename = os.path.join(nc_path,os.path.basename(pkl_pathfile)[:-3]+"nc")
    if not os.path.exists(nc_filename):
        try:
            with open(pkl_pathfile, "rb") as path: xr_stat_pixcount = pickle.load(path)
        except EOFError:
            print("\n   *** Warning: File could not be converted due to EOF Error - skip this file *** \n        %s" % os.path.basename(pkl_pathfile))
        else:
            xr_stat_pixcount.date.values = np.array([date_str[:12] for date_str in xr_stat_pixcount.date.values],dtype=np.object)
            xr_stat_pixcount.to_netcdf(nc_filename)
            path.close()
    
def read_pickle_file(filename):
    try:
        with open(filename, "rb") as path: xr_stat_pixcount = pickle.load(path)
    except EOFError:
        print("\n   *** Warning: File could not be converted due to EOF Error - skip this file *** \n        %s" % os.path.basename(pkl_pathfile))
    else:
        return xr_stat_pixcount
        
def read_nc_file(filename):
    xr_stat_pixcount = xr.open_dataset(filename,autoclose=True) #,decode_cf=False,autoclose=True,chunks={}
    return xr_stat_pixcount
    
def concat_stat_files(pkl_path, reverse, show_ctrl_plots=False):
    ## Read list of files which should be concatenated:
    import psutil
    fileending = "pixcount.pkl" if not reverse else "pixcount_rev.pkl"
    fileending_out = "pixcount_past.pkl" if not reverse else "pixcount_future.pkl"
    files = sorted(glob(os.path.join(pkl_path,"[0-9]*"+fileending)))
    print("\nStart concatenating the following files:\n   %s, ..." % ', '.join([os.path.basename(file_x) for file_x in files[:3]]))
    
    ## Initate array amongst which the dimensions are saved to check that no entries go missin/appear double:
    #datasets = [read_pickle_file(path) for path in files]
    #datasets = [read_nc_file(path) for path in files]
    #ls_datasets = []
    ls_DATE_TRT_ID = np.array([])

    ## Choose dimension along which the files should be concatenated:
    combined_ds = read_pickle_file(files[0])
    print("These are the dimensions of the xarray object:\n  %s" % combined_ds.dims)
    #print(combined_ds.info())
    dim = raw_input("Along which dimension should the objects be concatenated (e.g. 'DATE_TRT_ID'): ")
    while dim not in combined_ds.dims: dim = raw_input("Please choose out of the list above: ")
    files_KeyError = []
    
    ## Loop over all single files to be concatenated:
    print("Start reading in the data")
    for n, path in enumerate(files[1:],2):
        perc_complete = 100*int(n)/float(len(files))
        str_print = "  Completed %02d%% (working on file: %s)" % (perc_complete,os.path.basename(path))
        print('\r',str_print,end='') #, end='\r'
        #ls_datasets.append(read_nc_file(path))
        #new_ds = read_nc_file(path)
        new_ds = read_pickle_file(path)
        try:
            combined_ds = xr.concat([combined_ds,new_ds], dim)
        except KeyError:
            files_KeyError.append(new_ds.DATE_TRT_ID.values[0][:12])
            print("\n   *** Warning: File %s could not be appended due to KeyError\n" % os.path.basename(path))
        #ls_DATE_TRT_ID = np.append(ls_DATE_TRT_ID,ls_datasets[-1]["DATE_TRT_ID"].values)
        ls_DATE_TRT_ID = np.append(ls_DATE_TRT_ID,new_ds[dim].values)
        if len(np.unique(ls_DATE_TRT_ID))!=len(ls_DATE_TRT_ID):
            print("\n   *** Warning: Dimension lengths start to diverge ***\n")
            pdb.set_trace()
        if n%100==0:
            memoryUse = psutil.virtual_memory()
            memoryUse_MB = memoryUse.used >> 20; memoryUse_perc = memoryUse.percent
            print('\n   Current memory use: %sMB (%s%%)' % (memoryUse_MB,memoryUse_perc))
        if show_ctrl_plots and n%100==0:
            print(combined_ds)
            variable_rand = np.random.choice(np.array(["RZC_stat","IR_108_stat","THX_densCG_stat","CD2_stat","POT_VORTIC_30000_stat"]))
            statistic_rand = np.random.choice(np.array(["MEAN","PERC05","MAX","SUM","PERC99"]))
            #ls_datasets[-1][variable_rand].sel(statistic=statistic_rand).plot.line(x="time_delta", hue="DATE_TRT_ID", add_legend=False, alpha=0.6)
            new_ds[variable_rand].sel(statistic=statistic_rand).plot.line(x="time_delta", hue="DATE_TRT_ID", add_legend=False, alpha=0.6)
            plt.pause(1)
            plt.clf()
    plt.close(); print("\n")
    #print("\nStart concatenating the data along the dimension '%s'" % dim)
    #combined_ds = xr.concat(ls_datasets, dim)
    
    ## Check files which could not be appended due to KeyError:
    if len(files_KeyError)>0:
        print("Files which could not be appended:\n  %s" % files_KeyError)
        pdb.set_trace()
    
    ## Save Pickle:
    pkl_pathfile = os.path.join(pkl_path,"Combined_stat_"+fileending_out)
    with open(pkl_pathfile, "wb") as output_file: pickle.dump(combined_ds, output_file, protocol=-1)
    if os.path.exists(pkl_pathfile): print("Pickle file has been created in %s" % pkl_pathfile)
    else: raise FileNotFoundError("Pickle file was not created")

    ## Save NetCDF:
    nc_path = os.path.join(pkl_path,"nc/")
    if not os.path.exists(nc_path): os.makedirs(nc_path)
    
    nc_pathfile = os.path.join(nc_path,"Combined_stat_"+fileending_out[:-4]+".nc")
    combined_ds.to_netcdf(nc_pathfile)
    if os.path.exists(nc_pathfile): print("NetCDF file has been created in %s" % nc_pathfile)
    else: raise FileNotFoundError("NetCDF file was not created")

def convert_stat_files(path):
    bool_convert = raw_input("\nShould pickle files be converted to NetCDF? [y/n] ")
    while (bool_convert!="n" and bool_convert!="y"): bool_convert = raw_input("Should pickle files be converted to NetCDF? [y/n] ")
    
    if bool_convert=="n":
        print("  Files are not converted\n")
        return
    else:
        files = sorted(glob(os.path.join(path,"*.pkl")))
        print("  Files to be converted: \n    %s, ..." % ', '.join([os.path.basename(file_x) for file_x in files[:3]]))
        print("  Number of files to convert: %s" % len(files))
        nc_path = os.path.join(path,"nc/")
        if not os.path.exists(nc_path): os.makedirs(nc_path)
        
        for n, pkl_file in enumerate(files,1):
            perc_complete = 100*int(n)/float(len(files))
            str_print = "  Completed %02d%% (working on file: %s)" % (perc_complete,os.path.basename(pkl_file))
            print('\r',str_print,end='') #, end='\r'
            pickle2nc(pkl_file,nc_path)
        print("\nFinished converting NetCDF files to\n  %s\n" % nc_path)

def concat_future_past_concat_stat_files(pkl_path):
    print("\nConcatenate the concatenations of past and future statistics into one big file")
    import psutil
    
    ## Read file with future (t0 + n min) statistics:
    file_future = os.path.join(pkl_path,"Combined_stat_pixcount_future.pkl")
    expected_memory_need = float(os.path.getsize(file_future))/psutil.virtual_memory().available*100
    if expected_memory_need > 20:
        print("  *** Warning: File 'Combined_stat_pixcount_future.pkl' is expected ***\n"+\
              "      to use %02d%% of the memory, thus open NetCDF version as mfdataset" % \
              (expected_memory_need))
        file_future = os.path.join(pkl_path,"nc/Combined_stat_pixcount_future.nc")
        xr_future = xr.open_mfdataset(file_future,chunks={"DATE_TRT_ID":1000})
    else:
        with open(file_future, "rb") as path: xr_future = pickle.load(path)
    
    ## Read file with past (t0 - n min) statistics:
    if expected_memory_need > 20:
        file_past = os.path.join(pkl_path,"nc/Combined_stat_pixcount_past.nc")
        xr_past = xr.open_mfdataset(file_past,chunks={"DATE_TRT_ID":1000})
    else:
        file_past = os.path.join(pkl_path,"Combined_stat_pixcount_past.pkl")
        with open(file_past, "rb") as path: xr_past = pickle.load(path)
    xr_past = xr_past.where(xr_past["time_delta"]<0,drop=True)

    ## Concatenate to one file (first bringing 'time_delta' in xr_past in ascending order):
    xr_new = xr.concat([xr_past.sortby("time_delta"),xr_future],"time_delta")
    del(xr_past); del(xr_future)

    ## Remove 'time_delta' coordinate in TRT variables (which are only available for t0):
    ds_keys  = np.array(xr_new.keys())
    keys_TRT = ds_keys[np.where(["stat" not in key_ele and "pixc" not in key_ele for key_ele in ds_keys])[0]]
    keys_TRT_timedelta = xr_new.coords.keys()+["TRT_domain_indices","TRT_cellcentre_indices","CZC_lt57dBZ"]
    keys_TRT = list(set(keys_TRT) - set(keys_TRT_timedelta))
    for key_TRT in keys_TRT:
        if key_TRT in keys_TRT_timedelta: continue
        xr_new[key_TRT] = xr_new[key_TRT].sel(time_delta=0).drop("time_delta")
    
    ## Change all stat-variables to float32:
    keys_stat = ds_keys[np.where(["_stat" in key_ele for key_ele in ds_keys])[0]]
    for key_stat in keys_stat: xr_new[key_stat] = xr_new[key_stat].astype(np.float32)
    
    ## Change all pixc-variables to uint16:
    dtype_pixc = np.uint16 if xr_new.TRT_domain_indices.shape[2]<2**16-1 else np.uint32
    keys_pixc = ds_keys[np.where(["_pixc" in key_ele for key_ele in ds_keys])[0]]
    for key_pixc in keys_pixc: xr_new[key_pixc] = xr_new[key_pixc].astype(dtype_pixc)
    xr_new["CZC_lt57dBZ"] = xr_new["CZC_lt57dBZ"].astype(dtype_pixc)
    xr_new["TRT_cellcentre_indices"] = xr_new["TRT_cellcentre_indices"].astype(np.uint32)
    
    ## In case the dates have bin concatenated, only keep the twelve initial characters (Date+Time):
    if len(xr_new["DATE_TRT_ID"].values[0])>12:
        xr_new["date"].values = np.array([date_i[:12] for date_i in xr_new["date"].values])
    
    ## Save NetCDF:
    file_new = os.path.join(pkl_path,"nc/Combined_stat_pixcount.nc")
    xr_new.to_netcdf(file_new)
    
    ## Save Pickle:
    file_new = os.path.join(pkl_path,"Combined_stat_pixcount.pkl")
    with open(file_new, "wb") as output_file: pickle.dump(xr_new, output_file, protocol=-1)
    
    del(xr_new)

def collection_of_plotting_functions(pkl_path):
    file_path = os.path.join(pkl_path,"Combined_stat_pixcount.pkl")
    with open(file_path, "rb") as path: xr_new = pickle.load(path)
    
    ## RZC Time Series (can be replaced by other statistics/variables):
    xr_new["RZC_stat"].sel(statistic="MEAN").plot.line(x="time_delta", hue="DATE_TRT_ID", add_legend=False, alpha=0.1)
    plt.xlim([-45,45])
    plt.show()
    
    ## Strong Lightning event:
    xr_new["THX_dens_stat"].sel(statistic="SUM").where(xr_new["THX_dens_stat"].sel(statistic="SUM")>500,drop=True)
    THX_max_lat = xr_new["lat"].sel(DATE_TRT_ID="201807201655_2018072016000184").values
    THX_max_lon = xr_new["lon"].sel(DATE_TRT_ID="201807201655_2018072016000184").values
    print("Location of maximum lightning event: %s / %s" % (THX_max_lat[0],THX_max_lon[0]))
    
    ## Plot how well TRT and NOSTRADAMUS information agrees:
    ind_nan = np.isnan(xr_new["LZC_stat"].sel(statistic="PERC95",time_delta=-5)) | np.isnan(xr_new["RZC_stat"].sel(statistic="PERC95",time_delta=-5))
    plt.plot(xr_new["LZC_stat"].sel(statistic="PERC95",time_delta=-5).values[np.where(~ind_nan)],xr_new["RZC_stat"].sel(statistic="PERC95",time_delta=-5).values[np.where(~ind_nan)],'x')
    corr_val = np.corrcoef(xr_new["LZC_stat"].sel(statistic="PERC95",time_delta=-5).values[np.where(~ind_nan)],xr_new["RZC_stat"].sel(statistic="PERC95",time_delta=-5).values[np.where(~ind_nan)])
   
    print("Correlation between VIL and Rain Rate: %s" % corr_val[0,1])
    plt.show()

    ind_nan = np.isnan(xr_new["LZC_stat"].sel(statistic="PERC95")) | np.isnan(xr_new["RZC_stat"].sel(statistic="PERC95"))
    np.corrcoef(xr_new["LZC_stat"].sel(statistic="PERC95").values[np.where(~ind_nan)],xr_new["RZC_stat"].sel(statistic="PERC95").values[np.where(~ind_nan)])

    plt.plot([0,0],[16,16],'k-')
    plt.plot(xr_new["EZC45_stat"].sel(statistic="MAX",time_delta=0),xr_new["ET45"],'x',alpha=0.8)
    plt.xlabel("Circle")
    plt.ylabel("TRT-Cell")
    plt.title("Median EchoTop 45dBZ")
    perc_equal = np.sum((xr_new["EZC45_stat"].sel(statistic="MAX",time_delta=0)==xr_new["ET45"]).values)/len(xr_new["ET45"])*100.
    print(len(xr_new["ET45"]), len(xr_new["EZC45_stat"].sel(statistic="MAX",time_delta=0)))
    plt_text = "%02d%% equal" % perc_equal
    plt.text(3,10,plt_text)
    plt.show()


    plt.plot([0,0],[16,16],'k-')
    plt.plot(xr_new["EZC15_stat"].sel(statistic="MAX",time_delta=0),xr_new["ET15"],'x',alpha=0.8)
    plt.xlabel("Circle")
    plt.ylabel("TRT-Cell")
    plt.title("Median EchoTop 15dBZ")
    plt.xlim([4,20])
    plt.ylim([4,20])
    perc_equal = np.sum((xr_new["EZC15_stat"].sel(statistic="MAX",time_delta=0)==xr_new["ET15"]).values)/len(xr_new["ET15"])*100.
    plt_text = "%02d%% equal" % perc_equal
    plt.text(8,14,plt_text)
    plt.show()

## Wrapper function for adding additional auxiliary static variables to dataset (in training dataset creation environment):    
def wrapper_fun_add_aux_static_variables(CONFIG_PATH_set_tds,CONFIG_FILE_set_tds,pkl_path):
    bool_continue = raw_input(" WARNING: The config path might be hard coded, is the following path correct:\n   %s\n If yes, continue [y,n] " %
                              (CONFIG_PATH_set_tds+CONFIG_FILE_set_tds)) == "y"
    if bool_continue:
        cfg_set_tds   = get_config_info(CONFIG_PATH_set_tds,CONFIG_FILE_set_tds)
        cfg_set_input, cfg_var, cfg_var_combi = Nip.get_config_info(
                                                    cfg_set_tds["CONFIG_PATH_set_input"],
                                                    cfg_set_tds["CONFIG_FILE_set_input"],
                                                    cfg_set_tds["CONFIG_PATH_var_input"],
                                                    cfg_set_tds["CONFIG_FILE_var_input"],
                                                    cfg_set_tds["CONFIG_FILE_var_combi_input"],
                                                    "200001010001")
        file_path = os.path.join(pkl_path,"Combined_stat_pixcount.pkl")
        print(" Adding auxiliary variables to xarray object in file:\n   %s" % file_path)
        
        with open(file_path, "rb") as path: ds = pickle.load(path)
        ds = Nip.add_aux_static_variables(ds, cfg_set_input)
        with open(file_path, "wb") as output_file: pickle.dump(ds, output_file, protocol=-1)
        print(" Saved pickle file with added auxiliary variables")
        file_new = os.path.join(pkl_path,"nc/Combined_stat_pixcount.nc")
        ds.to_netcdf(file_new)
        print(" Saved NetCDF file with added auxiliary variables")
    else: raise IOError(" Skip adding auxiliary variables, change directory in script 'NOSTRADAMUS_0_training_ds_concatenation.py'")
    
## Exploratory data analysis
def EDA_wrapper(pkl_path, cfg_set):
    file_path = os.path.join(pkl_path,"Combined_stat_pixcount.pkl")
    print("Perform exploratory data analysis on the dataset:\n   %s" % file_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    