""" [COALITION3] Check for missing input data and list those in csv files."""

from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
import numpy as np
import pandas as pd
import pickle
import matplotlib.pylab as plt

from coalition3.inout.paths import path_creator

## =============================================================================
## FUNCTIONS:

## Check the availability of input datasets during the complete testdata period:
def create_df_missing(cfg_set_tds,cfg_set_input,cfg_var,check_sources):
    """Check the availability of input datasets during the complete testdata period.
    """
    print("Check the availability of input datasets during the complete testdata period")
    if check_sources!=["RADAR","SEVIRI","COSMO_CONV","THX"]:
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
    t_start = datetime.datetime.now()
    t_exp   = "(calculating)"
    for counter, sampling_time in enumerate(dt_complete_list):
        perc_checked_total = float(counter)/len(dt_complete_list)
        perc_checked = np.round((sampling_time.hour*60+sampling_time.minute)/1440.,2)        
        if counter%100==0 and counter > 10:
            t_exp = (datetime.datetime.now() + \
                     (datetime.datetime.now() - t_start)*int((1-perc_checked_total)/perc_checked_total)).strftime("%d.%m.%Y %H:%M")
        print("  Check input data availability of date: %s - %3d%% | Expected finishing time: %s" % \
              (sampling_time.strftime("%d.%m.%Y"),perc_checked*100,t_exp), end='\r')
        sys.stdout.flush()

        for RADAR_var in RADAR_vars:
            if path_creator(sampling_time, RADAR_var, "RADAR", cfg_set_input)[0][0] is None:
                df_missing.loc[sampling_time,RADAR_var] = True
        if path_creator(sampling_time, "IR_108", "SEVIRI", cfg_set_input)[0][0] is None:
            df_missing.loc[sampling_time,"SEVIRI"] = True
        if not os.path.exists(path_creator(sampling_time, "POT_VORTIC_70000", "COSMO_CONV", cfg_set_input)[0]):
            df_missing.loc[sampling_time,"COSMO_CONV"] = True
        if path_creator(sampling_time, "THX_abs", "THX", cfg_set_input)[0][0] is None:
            df_missing.loc[sampling_time,"THX"] = True
    
    
    df_missing.to_pickle(os.path.join(cfg_set_tds["root_path_tds"],u"MissingInputData.pkl"))
    print("Save dataframe to %s" % (os.path.join(cfg_set_tds["root_path_tds"],u"MissingInputData.pkl")))
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
    df_missing = pd.read_pickle(os.path.join(cfg_set_tds["root_path_tds"],u"MissingInputData.pkl"))
    for var in columns_list: print("  %s: %s" % (var,np.sum(df_missing[var])))
    
    df_missing_SEVIRI = df_missing.loc[df_missing["SEVIRI"]==True,"SEVIRI"]
    df_missing_COSMO  = df_missing.loc[df_missing["COSMO_CONV"]==True,"SEVIRI"]

    if len(df_missing_SEVIRI.index.month)>0:
        df_missing_SEVIRI.groupby(df_missing_SEVIRI.index.month).count().plot(kind="bar")
        plt.title("Number of missing\nSEVIRI files per month")
        #plt.pause(4)
        plt.show()
    plt.close()
    
    if len(df_missing_COSMO.index.month)>0:
        df_missing_COSMO.groupby(df_missing_COSMO.index.month).count().plot(kind="bar")
        plt.title("Number of missing\nCOSMO files per month")
        #plt.pause(4)
        plt.show()
    plt.close()
    
    df_missing_SEVIRI["datetime"] = df_missing_SEVIRI.index
    df_missing_COSMO["datetime"]  = df_missing_COSMO.index
    
    df_missing_COSMO_dates = df_missing_COSMO["datetime"].values
    print("Dates with missing COSMO data:")
    print(np.unique([pd.to_datetime(date).replace(hour=0,minute=0) for date in df_missing_COSMO_dates]))
    
    df_missing.loc[df_missing["SEVIRI"]==True,"SEVIRI"].to_csv(os.path.join(cfg_set_tds["root_path_tds"],u"MissingInputData_SEVIRI.csv"),
                                                               index_label=True,index=True,sep=";")
    df_missing.loc[df_missing["COSMO_CONV"]==True,"COSMO_CONV"].to_csv(os.path.join(cfg_set_tds["root_path_tds"],u"MissingInputData_COSMO_CONV.csv"),
                                                               index_label=True,index=True,sep=";")
    




