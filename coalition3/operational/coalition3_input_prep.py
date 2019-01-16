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
CONFIG_FILE_set       = "NOSTRADAMUS_1_input_prep.cfg" # "NOSTRADAMUS_1_input_prep_shorttest.cfg"
CONFIG_FILE_var       = "cfg_var_test.csv"
CONFIG_FILE_var_combi = "cfg_var_combi.csv"

## Input basics
t0_str = "201807041710"
#t0_str = "201807220055"    
#t0_str = "201507071430"
#t0_str = "201805302300"
n_updates = 0

cfg_set, cfg_var, cfg_var_combi = Nip.get_config_info(CONFIG_PATH,CONFIG_FILE_set,CONFIG_PATH,CONFIG_FILE_var,
                                                      CONFIG_FILE_var_combi,t0_str)
Nip.print_config_info(cfg_set,CONFIG_FILE_set,CONFIG_FILE_var)

## ===============================================================================
## Set-up the basic structure

## Check whether temporary subdirectories are present, otherwise create those
#Nip.check_create_tmpdir(cfg_set)

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

## Plot moving and displaced fields next to each other
for var in cfg_set["var_list"]:
    if var in ["Wind","Conv"]: continue
    #Nip.plot_displaced_fileds(var,cfg_set)

## Correction of residual movements:
if cfg_set["resid_disp"]:
    t1 = datetime.datetime.now()
    Nip.residual_disp(cfg_set)
    t2 = datetime.datetime.now()
    print("  Elapsed time for the correction of residual movements: "+str(t2-t1)+"\n")

## Update the data for the next 5min step:
for _ in range(n_updates):
    t1 = datetime.datetime.now()
    cfg_set["t0"]     = cfg_set["t0"] + cfg_set["time_change_factor"]*datetime.timedelta(minutes=cfg_set["timestep"])
    t0                = cfg_set["t0"]
    cfg_set["t0_doy"] = t0.timetuple().tm_yday
    cfg_set["t0_str"] = t0.strftime("%Y%m%d%H%M")
    
    Nip.update_fields(cfg_set,verbose=False)

    t2 = datetime.datetime.now()
    print("  Elapsed time for updating the displacement: "+str(t2-t1)+"\n")
#sys.exit()

## Analyse skill scores if verification should be made:
if cfg_set["verify_disp"]:
    for var in cfg_set["var_list"]:
        if var in ["Wind","Conv"]: continue
        Nip.analyse_skillscores(cfg_set,var)
        Nip.compare_skillscores_help(cfg_set,var,["Onestep","Twostep","None"]) #[1.0,8.0,95.0,99.0,99.9] [0.5,1.0,2.0,4.0,8.0]
sys.exit()

        

        
        






