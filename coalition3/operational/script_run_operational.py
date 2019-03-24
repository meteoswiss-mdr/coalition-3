# coding: utf-8
""" [COALITION3] This script summarises to the current extent,
    how an operational workflow could look like. But so far,
    it only covers the steps up until the generation of the 
    statistics/pixel counts, and no prediction using statistical
    models."""

## ===============================================================================
## Import packages and functions

from __future__ import division
from __future__ import print_function

import os
import sys
import datetime

import coalition3.inout.readconfig as cfg
import coalition3.operational.update as upd
import coalition3.operational.convertds as cds
import coalition3.operational.processing as prc
import coalition3.operational.statistics as stat
import coalition3.operational.prediction as pred

## ===============================================================================
## Get initial time for timing the function:
t1 = datetime.datetime.now()

## Get time point to start with:
t0_str = sys.argv[1] if len(sys.argv)==2 else IOError("Need time user argv time point <%Y%m%d%H%M>")

## Get config info:
cfg_set, cfg_var, cfg_var_combi = cfg.get_config_info_op()
cfg_set = cfg.cfg_set_append_t0(cfg_set,t0_str)
cfg.print_config_info_op(cfg_set)

## Read in variables to .nc/.npy files and displace TRT cell centres:
prc.displace_variables(cfg_set,cfg_var,reverse=False)

## Read in statistics and pixel counts
stat.append_statistics_pixcount(cfg_set,cfg_var,cfg_var_combi,reverse=False)

## Add auxiliary and derived variables
stat.add_auxiliary_derived_variables(cfg_set)

## Convert statistics (xarray) dataset into (pandas) dataframe:
cds.convert_stat_ds_to_df(cfg_set)

## Convert statistics (xarray) dataset into (pandas) dataframe:
pred.predict_TRT_Rank(cfg_set)

## Delete all .nc files (vararr and disparr files)
#prc.clean_disparr_vararr_tmp(cfg_set)


## OR: 

## Update to next time step (NOT UPDATED - SEE MODULE update.py):
#t0_str_new = cfg_set["t0"] + datetime.timedelta(minutes=cfg_set["timestep"])
#cfg_set    = cfg.cfg_set_append_t0(cfg_set,t0_str_new.strftime("%Y%m%d%H%M"))
#upd.update_fields(cfg_set,verbose=False)
        
## Read in statistics and pixel counts
#stat.append_statistics_pixcount(cfg_set,cfg_var,cfg_var_combi,reverse=False)

## Add auxiliary and derived variables
#stat.add_auxiliary_derived_variables(cfg_set)

t2 = datetime.datetime.now()
line_str = 103*"-"
print("\n%s\n  Elapsed time for processing: " % line_str +str(t2-t1)+"\n%s" % line_str)
        
        






