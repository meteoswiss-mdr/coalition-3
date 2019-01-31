"""[COALITION3] Script that returns all the time points sampled during the 
   training period, depending on the training data length, the sampling 
   frequency, and the actual existance of TRT cells at these time points.
   Besides two plots showing the TRT properties of the cells sampled, 
   the main output is a log file which is used to keep track of the 
   processed time points and the collection of cell-specific information."""

# ===============================================================================
# Import packages and functions

from __future__ import division
from __future__ import print_function

import sys
import datetime

import coalition3.inout.readconfig as cfg
import coalition3.training.preparation as prep
import coalition3.training.logfile as log
import coalition3.training.missing as miss

# ===============================================================================
# Execute functions:

## Get config settings (training and operational):
cfg_set_tds = cfg.get_config_info_tds()
cfg_set_input, cfg_var, cfg_var_combi = cfg.get_config_info_op()
cfg.print_config_info_tds(cfg_set_tds)

## Set up list with sampling datetime objects 
t1 = datetime.datetime.now()
dt_sampling_list = prep.create_dt_sampling_list(cfg_set_tds)
t2 = datetime.datetime.now()
print("  Elapsed time for creation of sampling timepoint list: "+str(t2-t1)+"\n")

## Get information on TRT cells for all  these time steps:
## WARNING: Quite time consuming (opens all TRT files within time-frame)
t1 = datetime.datetime.now()
samples_df = prep.get_TRT_cell_info(dt_sampling_list,cfg_set_tds,cfg_set_input,len_ini_df=100000)
t2 = datetime.datetime.now()
print("  Elapsed time for creation of sample dataframe: "+str(t2-t1)+"\n")

## Change and append some of the TRT cell values or append additional columns:
prep.change_append_TRT_cell_info(cfg_set_tds)

## Plot histograms and map of TRT cell values:
prep.exploit_TRT_cell_info(cfg_set_tds)

## Print primarily basic information on the training dataset (without reading the data!):
prep.print_basic_info(cfg_set_tds)

## Check for missing input data and save these in pandas dataframe
## WARNING: Quite time consuming (loops over 5min time steps to check existence of files)
t1 = datetime.datetime.now()
check_sources = ["RADAR","SEVIRI","COSMO_CONV","THX"]
df_missing = miss.create_df_missing(cfg_set_tds,cfg_set_input,cfg_var,check_sources)
miss.analyse_df_missing(cfg_set_tds,cfg_set_input,cfg_var,check_sources)
t2 = datetime.datetime.now()
print("  Elapsed time for creation of sampling timepoint list: "+str(t2-t1)+"\n")

## Make file which is containing information on training dataset generation
## which is filled up during the training dataset generation:
log.setup_log_file(cfg_set_tds,cfg_set_input)

## Set up the trainings dataset (DEPRECATED):
# Nds.get_empty_tds(cfg_set_tds,cfg_set_input)

    
    
    
    
    
    
    
