""" [COALITION3] Create output for EUMETSAT Conference poster """

# ===============================================================================
# Import packages and functions

from __future__ import division
from __future__ import print_function

import os
import pdb
import sys
import datetime
import configparser
import numpy as np

import coalition3.inout.readconfig as cfg
import coalition3_casestudy_fun as ccs

sys.path.insert(0, '/data/COALITION2/database/radar/ccs4/python')
import metranet

## ===============================================================================
## Make static setting

## Input basics
#t_start   = datetime.datetime(2015, 7, 7, 16, 15)
#t_end     = datetime.datetime(2015, 7, 7, 18, 30)
t_start   = datetime.datetime(2015, 7, 7, 11, 45)
t_end     = datetime.datetime(2015, 7, 7, 15, 45)
t_end_alt = datetime.datetime(2015, 7, 7, 18, 30)

#x_range   = (300,355)#(300,370)
#y_range   = (160,220)#(160,230)
#x_range   = (350,450)# <- This is used for the poster
#y_range   = (200,300)# <- This is used for the poster
x_range   = (300,500)#(300,370)
y_range   = (150,350)#(160,230)

t_end_str = datetime.datetime.strftime(t_end, "%Y%m%d%H%M")
cfg_set, cfg_var, cfg_var_combi = cfg.get_config_info_op()
cfg_set = ccs.update_cfg_set(cfg_set,t_start,t_end,x_range,y_range,t_end_alt)
      
print('\n\nStart NOSTRADAMUS case study with:'+
'\n  Starttime:      '+str(datetime.datetime.strftime(t_start, "%d.%m.%Y %H:%M"))+
'\n  Endtime:        '+str(datetime.datetime.strftime(t_end,   "%d.%m.%Y %H:%M"))+
'\n  Timestep:       '+str(cfg_set["timestep"])+"min"+
'\n  X range:        '+str(x_range[0])+"000km to "+str(x_range[1])+'000km E'+
'\n  Y range:        '+str(y_range[0])+"000km to "+str(y_range[1])+'000km N\n')

## ===============================================================================
## Get data (orig and displaced) of the respective time frame
ccs.subset_disp_data(cfg_set, resid=True)

## Plot temporal statistics of data subset (variable-specific)
#ccs.subset_stats(cfg_set)

## Plot some displacement fields
#ccs.disp_field_plot(cfg_set)















