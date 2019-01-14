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
CONFIG_PATH_set_tds   = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset_5min_16km_23km_RADAR_20181220/"
CONFIG_FILE_set_tds   = "NOSTRADAMUS_0_training_ds_20181220.cfg"

cfg_set_tds   = Nds.get_config_info(CONFIG_PATH_set_tds,CONFIG_FILE_set_tds)
Nds.print_config_info(cfg_set_tds,CONFIG_FILE_set_tds)

## ===============================================================================
## Get input config settings (random date only needed to load config settings 'cfg_set'):
random_date = datetime.datetime(2000,1,1,0,0)
cfg_set_input, cfg_var, cfg_var_combi = Nip.get_config_info(
                                            cfg_set_tds["CONFIG_PATH_set_input"],
                                            cfg_set_tds["CONFIG_FILE_set_input"],
                                            cfg_set_tds["CONFIG_PATH_var_input"],
                                            cfg_set_tds["CONFIG_FILE_var_input"],
                                            cfg_set_tds["CONFIG_FILE_var_combi_input"],
                                            random_date.strftime("%Y%m%d%H%M"))
                                        
## Get time point to be processed:
""" SET MANUALLY: """
log_file_pkl = "/opt/users/jmz/0_training_NOSTRADAMUS_ANN/Training_Dataset_Processing_Status_RADAR_nonmin.pkl"

time_point, samples_df = Nds.read_edit_log_file(cfg_set_tds,cfg_set_input,"start",log_file=log_file_pkl)
#time_point = datetime.datetime(2018,5,9,17,35)    
if time_point==200:
    sys.exit(time_point)
while time_point is None:
    time_point, samples_df = Nds.read_edit_log_file(cfg_set_tds,cfg_set_input,"start",samples_df=samples_df)

## Write all stdout ouput to file:
orig_stdout = sys.stdout
log_file_path = "%s%s.txt" % (cfg_set_tds["PATH_stdout_output"],time_point.strftime("%Y%m%d%H%M"))
log_file = open(log_file_path, 'w')
sys.stdout = log_file

t0 = datetime.datetime.now()
## Change t0 to timestep:
cfg_set_input["t0"]     = time_point
cfg_set_input["t0_doy"] = cfg_set_input["t0"].timetuple().tm_yday
cfg_set_input["t0_str"] = cfg_set_input["t0"].strftime("%Y%m%d%H%M")

## Read in variables to .nc/.npy files and displace TRT cell centres:
t1 = datetime.datetime.now()
print(" =========== Start displacement algorithm =========== \n")
Nds.displace_variables(cfg_set_input,cfg_var,reverse=False)
print(" ============ Change temporal direction ============= \n")
Nds.displace_variables(cfg_set_input,cfg_var,reverse=True)
t2 = datetime.datetime.now()
print("  Elapsed time for the displacement: "+str(t2-t1))
print(" ============ End displacement algorithm ============\n\n")

## Read in statistics and pixel counts
print(" ======================= Start reading statistics =======================\n")
t1 = datetime.datetime.now()
Nip.append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=False)
Nip.append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=True)

## Move statistics out of /tmp directory to collection directory:
Nds.move_statistics(cfg_set_input,cfg_set_tds,"diam_16km/")
t2 = datetime.datetime.now()
print("\n  Elapsed time for reading the statistics / pixel counts: "+str(t2-t1))
print(" ======================== End reading statistics ========================\n\n")


## Change diameter of circle of interest to double the area:
print(" =========== Start process with change form width =========== \n")
t1 = datetime.datetime.now()
Nds.change_form_width_statistics(1.414213,cfg_set_input,cfg_var,cfg_var_combi)
Nds.move_statistics(cfg_set_input,cfg_set_tds,"diam_23km/")
t2 = datetime.datetime.now()
print("\n  Elapsed time for reading the statistics with changed form: "+str(t2-t1))
print(" ============ End process with change form width ============\n\n")

## Delete all .nc files (vararr and disparr files)
Nds.clean_disparr_vararr_tmp(cfg_set_input)
    
## State that the processing is finished
exit_code = Nds.read_edit_log_file(cfg_set_tds,cfg_set_input,"end",time_point,log_file=log_file_pkl)                                        
if exit_code==100: print("\n\n ALL TIMEPOINTS HAVE BEEN PROCESSED")

tend = datetime.datetime.now()
print("\n  Elapsed time for one timestep: "+str(tend-t0)+"\n")

## Close log-file
sys.stdout = orig_stdout
log_file.close()

sys.exit(exit_code)

    
    
    
