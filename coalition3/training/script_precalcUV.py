""" [COALITION3] Precalculating the UV fields for the training period:

    This script is repeatedly called by the shell script precalcUV_loop.sh,
    where it generates the UV field for the respective date.

    example call:
    python script_precalcUV.py 2019-03-01 training
    """

# ===============================================================================
# Import packages and functions

from __future__ import division
from __future__ import print_function

import sys
import datetime

import coalition3.operational.lagrangian as lag
import coalition3.inout.readconfig as cfg

## ===============================================================================
## Make static setting

## Interpreting command line arguments
# first argument is date in %Y-%m-%d
t0_str = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d").strftime("%Y%m%d%H%M")
# second [optional] arguement: "training" do another run with revered time (foreward tracking)
type   = sys.argv[2]

## Generate config dictionary
cfg_set, cfg_var, cfg_var_combi = cfg.get_config_info_op()
cfg_set = cfg.cfg_set_append_t0(cfg_set,t0_str)
cfg.print_config_info_op(cfg_set)
cfg_set["verbose"] = False

## Set settings such that the UV fields are calculated from past to present:
cfg_set["future_disp_reverse"] = False
cfg_set["time_change_factor"]  = 1

## Check whether Displacement array is already existent, otherwise create it.
## If existent, replace with newest displacement
print("  Pre-calculate displacement field for %s" % cfg_set["t0"].strftime("%Y-%m-%d %H:%M"))
t1 = datetime.datetime.now()
lag.check_create_precalc_disparray(cfg_set)
t2 = datetime.datetime.now()
print("  Elapsed time for creating precalculated displacement array: "+str(t2-t1)+"\n")

## In case the type is set to training, also pre-calculate reversed flow field:
if type == "training":
    cfg_set["future_disp_reverse"] = True
    cfg_set["time_change_factor"]  = -1
    print("  Pre-calculate reversed displacement field for %s" % cfg_set["t0"].strftime("%Y-%m-%d %H:%M"))
    t1 = datetime.datetime.now()
    lag.check_create_precalc_disparray(cfg_set)
    t2 = datetime.datetime.now()
    print("  Elapsed time for creating precalculated displacement array: "+str(t2-t1)+"\n")
    
"""
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
"""
        
        
        






