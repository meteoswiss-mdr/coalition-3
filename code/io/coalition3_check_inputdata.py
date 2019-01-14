#!/opt/users/common/packages/anaconda2//bin

""" Setup of training dataset for NOSTRADAMUS ANN"""

# ===============================================================================
# Import packages and functions

from __future__ import division
from __future__ import print_function

import sys
import datetime

import NOSTRADAMUS_0_training_ds_fun as Nds
import NOSTRADAMUS_1_input_prep_fun as Nip

## ===============================================================================
## Make static setting

## Define path to config file
CONFIG_PATH_set_tds   = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/"
CONFIG_FILE_set_tds   = "NOSTRADAMUS_0_training_ds.cfg"

cfg_set_tds   = Nds.get_config_info(CONFIG_PATH_set_tds,CONFIG_FILE_set_tds)
Nds.print_config_info(cfg_set_tds,CONFIG_FILE_set_tds)

## ===============================================================================
## Get input config settings:
start_date = datetime.datetime.combine(cfg_set_tds["tds_period_start"],datetime.time(0,0))
cfg_set_input, cfg_var, cfg_var_combi = Nip.get_config_info(
                                            cfg_set_tds["CONFIG_PATH_set_input"],
                                            cfg_set_tds["CONFIG_FILE_set_input"],
                                            cfg_set_tds["CONFIG_PATH_var_input"],
                                            cfg_set_tds["CONFIG_FILE_var_input"],
                                            cfg_set_tds["CONFIG_FILE_var_combi_input"],
                                            start_date.strftime("%Y%m%d%H%M"))

## Create dataframe with datetime objects as index 
check_sources = ["RADAR","SEVIRI","COSMO_CONV","THX"]

t1 = datetime.datetime.now()
df_missing = Nds.create_df_missing(cfg_set_tds,cfg_set_input,cfg_var,check_sources)
t2 = datetime.datetime.now()
print("  Elapsed time for creation of sampling timepoint list: "+str(t2-t1)+"\n")


Nds.analyse_df_missing(cfg_set_tds,cfg_set_input,cfg_var,check_sources)
                                        
                                        
                                        