""" [COALITION3] This script is called repeatedly by the shell script 
    'script_tds_processing_loop.sh', in order to loop through all the
    time points summarised in the log file which was created with
    the script 'script_tds_preparation'. First it checks which time
    step to process, then it loads the input variables and the UV
    displacement fields into the /tmp directory, moves the TRT
    cell centres, reads the statistics and pixel counts, and finally
    removes the respective files from the /tmp directory."""

## ===============================================================================
## Import packages and functions

from __future__ import division
from __future__ import print_function

import os
import sys
import datetime

import coalition3.inout.readconfig as cfg
import coalition3.training.logfile as log
import coalition3.operational.processing as prc
import coalition3.operational.statistics as stat

## ===============================================================================
## Make static setting

## Define path to config file
cfg_set_tds   = cfg.get_config_info_tds()
cfg.print_config_info_tds(cfg_set_tds)
cfg_set_input, cfg_var, cfg_var_combi = cfg.get_config_info_op()
        
## Get time point to be processed:
time_point, samples_df = log.read_edit_log_file(cfg_set_tds,cfg_set_input,"start")

## In case all time points are processed,
## exit the processing and return exit code 200.
if time_point==200:
    sys.exit(time_point)
    
## In case not all input data is available at time point chosen (returns None)
## try another time point:
while time_point is None:
    time_point, samples_df = log.read_edit_log_file(cfg_set_tds,cfg_set_input,
                                                    "start",samples_df=samples_df)
## Write all stdout ouput to file:
if len(sys.argv)>1 and sys.argv[1]=="std2file":
    orig_stdout = sys.stdout
    log_file_path = os.path.join(cfg_set_tds["stdout_output_path"],
                                 "%s.txt" % time_point.strftime("%Y%m%d%H%M"))
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file

t0 = datetime.datetime.now()
## [OPERATIONAL] Change t0 to timestep:
cfg_set_input["t0"]     = time_point
cfg_set_input["t0_doy"] = cfg_set_input["t0"].timetuple().tm_yday
cfg_set_input["t0_str"] = cfg_set_input["t0"].strftime("%Y%m%d%H%M")

## [OPERATIONAL] Read in variables to .nc/.npy files and displace TRT cell centres:
t1 = datetime.datetime.now()
print(" =========== Start displacement algorithm =========== \n")
prc.displace_variables(cfg_set_input,cfg_var,reverse=False)
print(" ============ Change temporal direction ============= \n")
prc.displace_variables(cfg_set_input,cfg_var,reverse=True)
t2 = datetime.datetime.now()
print("  Elapsed time for the displacement: "+str(t2-t1))
print(" ============ End displacement algorithm ============\n\n")

## [OPERATIONAL] Read in statistics and pixel counts
print(" ======================= Start reading statistics =======================\n")
t1 = datetime.datetime.now()
stat.append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=False)
stat.append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=True)

## [OPERATIONAL] Move statistics out of /tmp directory to collection directory:
prc.move_statistics(cfg_set_input,cfg_set_tds,"diam_16km/")
t2 = datetime.datetime.now()
print("\n  Elapsed time for reading the statistics / pixel counts: "+str(t2-t1))
print(" ======================== End reading statistics ========================\n\n")

## Change diameter of circle of interest to double the area:
print(" =========== Start process with change form width =========== \n")
t1 = datetime.datetime.now()
prc.change_form_width_statistics(1.414213,cfg_set_input,cfg_var,cfg_var_combi)
prc.move_statistics(cfg_set_input,cfg_set_tds,"diam_23km/")
t2 = datetime.datetime.now()
print("\n  Elapsed time for reading the statistics with changed form: "+str(t2-t1))
print(" ============ End process with change form width ============\n\n")

## [OPERATIONAL] Delete all .nc files (vararr and disparr files)
prc.clean_disparr_vararr_tmp(cfg_set_input)
    
## State that the processing is finished
exit_code = log.read_edit_log_file(cfg_set_tds,cfg_set_input,"end",time_point)                                        
if exit_code==200: print("\n\n ALL TIMEPOINTS HAVE BEEN PROCESSED")

tend = datetime.datetime.now()
print("\n  Elapsed time for one timestep: "+str(tend-t0)+"\n")

## Close log-file
if len(sys.argv)>1 and sys.argv[1]=="std2file":
    sys.stdout = orig_stdout
    log_file.close()

sys.exit(exit_code)

    
    
    
