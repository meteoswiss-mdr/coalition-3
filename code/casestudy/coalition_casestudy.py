#!/opt/users/common/packages/anaconda2//bin

""" Input generation for NOSTRADAMUS artificial neural network:

Input fields for NOSTRADAMUS are spatially displaced by the motion 
fields derived from an optical flow algorithm, using radar rain
rates (RZC) to derive the motion vectors.
"""

# ===============================================================================
# Import packages and functions

from __future__ import division
from __future__ import print_function

import sys
import configparser
import datetime
import os
import pdb #pdb.set_trace()
import numpy as np

import pysteps as st
import NOSTRADAMUS_1_input_prep_fun as Nip
import NOSTRADAMUS_1_casestudy_fun as Ncs
sys.path.insert(0, '/data/COALITION2/database/radar/ccs4/python')
import metranet

## ===============================================================================
## Make static setting

## Define path to config file
CONFIG_PATH = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN"
CONFIG_FILE_set = "NOSTRADAMUS_1_input_prep_longrun.cfg"
CONFIG_FILE_var = "cfg_var.csv"

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
cfg_set, cfg_var = Nip.get_config_info(CONFIG_PATH,CONFIG_FILE_set,CONFIG_FILE_var,t_end_str)
cfg_set = Ncs.update_cfg_set(cfg_set,t_start,t_end,x_range,y_range,t_end_alt)
      
print('\n\nStart NOSTRADAMUS case study with:'+
'\n  Starttime:      '+str(datetime.datetime.strftime(t_start, "%d.%m.%Y %H:%M"))+
'\n  Endtime:        '+str(datetime.datetime.strftime(t_end,   "%d.%m.%Y %H:%M"))+
'\n  Timestep:       '+str(cfg_set["timestep"])+"min"+
'\n  X range:        '+str(x_range[0])+"000km to "+str(x_range[1])+'000km E'+
'\n  Y range:        '+str(y_range[0])+"000km to "+str(y_range[1])+'000km N\n')

## ===============================================================================
## Get data (orig and displaced) of the respective time frame
Ncs.subset_disp_data(cfg_set, resid=True)

## Plot temporal statistics of data subset (variable-specific)
#Ncs.subset_stats(cfg_set)

## Plot some displacement fields
#Ncs.disp_field_plot(cfg_set)















