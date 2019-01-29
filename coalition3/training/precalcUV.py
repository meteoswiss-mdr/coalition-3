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
import datetime

import NOSTRADAMUS_1_input_prep_fun as Nip

## ===============================================================================
## Make static setting

## Define path to config file
CONFIG_PATH = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/"
CONFIG_FILE_set = "NOSTRADAMUS_1_input_prep.cfg"
CONFIG_FILE_var = "cfg_var_test.csv"
CONFIG_FILE_var_combi = "cfg_var_combi.csv"

## Input basics
#t0_str = "201804010000"
t0_str = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d").strftime("%Y%m%d%H%M")

cfg_set, cfg_var, cfg_var_combi = Nip.get_config_info(CONFIG_PATH,CONFIG_FILE_set,CONFIG_FILE_var,CONFIG_FILE_var_combi,t0_str)
Nip.print_config_info(cfg_set,CONFIG_FILE_set,CONFIG_FILE_var)
cfg_set["verbose"] = False

## ===============================================================================
## Check whether Displacement array is already existent, otherwise create it.
## If existent, replace with newest displacement
print("  Precalcualte displacement field for %s" % cfg_set["t0"].strftime("%Y-%m-%d %H:%M"))
t1 = datetime.datetime.now()
Nip.check_create_precalc_disparray(cfg_set)
t2 = datetime.datetime.now()
print("  Elapsed time for creating precalculated displacement array: "+str(t2-t1)+"\n")
sys.exit()

## Update the data for the next 5min step:
while cfg_set["t0"].month < 10:
    t1 = datetime.datetime.now()
    
    cfg_set["t0"]     = cfg_set["t0"] + datetime.timedelta(days=1)
    t0                = cfg_set["t0"]
    cfg_set["t0_doy"] = t0.timetuple().tm_yday
    cfg_set["t0_str"] = t0.strftime("%Y%m%d%H%M")
    if cfg_set["t0"].month == 10:
        print("   Finished precalculating displacement fields")
        break
    else:
        print("  Precalcualte displacement field for %s" % t0.strftime("%Y-%m-%d %H:%M"))
        Nip.check_create_precalc_disparray(cfg_set)
    t2 = datetime.datetime.now()
    print("  Elapsed time for creating precalculated displacement array: "+str(t2-t1)+"\n")
        
        
        
        






