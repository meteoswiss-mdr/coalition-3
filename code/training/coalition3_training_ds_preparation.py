#!/opt/users/common/packages/anaconda2//bin

""" Setup of training dataset for NOSTRADAMUS ANN"""

# ===============================================================================
# Import packages and functions

from __future__ import division
from __future__ import print_function

import sys
import datetime
import pandas as pd
import pdb

import NOSTRADAMUS_0_training_ds_fun as Nds
import NOSTRADAMUS_1_input_prep_fun as Nip

## ===============================================================================
## Make static setting

## Define path to config file
CONFIG_PATH_set_tds   = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset_5min_16km_23km_RADAR_20181220/"
#CONFIG_PATH_set_tds   = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/"
CONFIG_FILE_set_tds   = "NOSTRADAMUS_0_training_ds_20181220.cfg"
#CONFIG_FILE_set_tds   = "NOSTRADAMUS_0_training_ds.cfg"

cfg_set_tds   = Nds.get_config_info(CONFIG_PATH_set_tds,CONFIG_FILE_set_tds)
Nds.print_config_info(cfg_set_tds,CONFIG_FILE_set_tds)

## ===============================================================================
## Set up list with sampling datetime objects 
t1 = datetime.datetime.now()
dt_sampling_list = Nds.create_dt_sampling_list(cfg_set_tds)
t2 = datetime.datetime.now()
print("  Elapsed time for creation of sampling timepoint list: "+str(t2-t1)+"\n")

## Get input config settings:
random_date = datetime.datetime(2000,1,1,0,0)
cfg_set_input, cfg_var, cfg_var_combi = Nip.get_config_info(
                                            cfg_set_tds["CONFIG_PATH_set_input"],
                                            cfg_set_tds["CONFIG_FILE_set_input"],
                                            cfg_set_tds["CONFIG_PATH_var_input"],
                                            cfg_set_tds["CONFIG_FILE_var_input"],
                                            cfg_set_tds["CONFIG_FILE_var_combi_input"],
                                            random_date.strftime("%Y%m%d%H%M"))

## Get information on TRT cells for all  these time steps:
## WARNING: Quite time consuming (opens all TRT files within time-frame)
"""
t1 = datetime.datetime.now()
samples_df = Nds.get_TRT_cell_info(dt_sampling_list,cfg_set_tds,cfg_set_input,len_ini_df=100000)
t2 = datetime.datetime.now()
print("  Elapsed time for creation of sample dataframe: "+str(t2-t1)+"\n")

## Change and append some of the TRT cell values or append additional oness:
Nds.change_append_TRT_cell_info(cfg_set_tds)

## Plot histograms and map of TRT cell values:
Nds.exploit_TRT_cell_info(cfg_set_tds)

## Print primarily basic information on the training dataset (without reading the data!):
Nds.print_basic_info(cfg_set_tds)
"""  
## Make file which is containing information on training dataset generation
## which is filled up during the training dataset generation:
Nds.setup_log_file(cfg_set_tds,cfg_set_input)

## Set up the trainings dataset (DEPRECATED):
# Nds.get_empty_tds(cfg_set_tds,cfg_set_input)

    
    
    
    
    
    
    
