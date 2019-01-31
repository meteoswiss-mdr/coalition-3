""" [COALITION3] Functions to process a single time step, from reading the
    UV fields, loading the input data to the /tmp directory, displacing
    the TRT cell centres, reading the statistics and pixel counts and 
    cleaning up the temporary directory."""

from __future__ import division
from __future__ import print_function

import os
import datetime
import pickle
import shutil
import numpy as np
import pandas as pd

from coalition3.inout.readconfig import form_size
import coalition3.operational.lagrangian as lag
import coalition3.operational.inputdata as ipt
import coalition3.operational.statistics as sta

## =============================================================================
## FUNCTIONS:

## Read UV fields, read input variable arrays, displace TRT cell centres,
## and read the TRT indices:
def displace_variables(cfg_set_input,cfg_var,reverse):
    """Displace past or future variables to current t0.

    Parameters
    ----------
    
    reverse : boolean
        Boolean value stating whether fields should be displaced from
        past to t0 (False) or from future to t0 (True).        
    """
    
    ## Change boolean cfg_set_input["future_disp_reverse"]:
    cfg_set = cfg_set_input.copy()
    print_reverse = "future" if reverse else "past"
    cfg_set["future_disp_reverse"] = True if reverse else False
    cfg_set["time_change_factor"]  = -1 if reverse else 1
    
    print("Displace %s observations to time step: %s" % (print_reverse,cfg_set["t0"].strftime("%Y-%m-%d %H:%M")))
    cfg_set["verbose"] = False
    
    ## Check whether precalculated displacement array is already existent, otherwise create it.
    ## If existent, replace with newest displacement
    if cfg_set["precalc_UV_disparr"]:
        t1 = datetime.datetime.now()
        lag.check_create_precalc_disparray(cfg_set)
        t2 = datetime.datetime.now()
        print("  Elapsed time for creation of precalculated displacement array: "+str(t2-t1)+"\n")

    ## Check whether Displacement array is already existent, otherwise create it.
    ## If existent, replace with newest displacement
    t1 = datetime.datetime.now()
    lag.check_create_disparray(cfg_set)
    t2 = datetime.datetime.now()
    print("  Elapsed time for creation of displacement array: "+str(t2-t1)+"\n")

    ## Create numpy arrays of variables for quick access
    t1 = datetime.datetime.now()
    ipt.create_new_vararray(cfg_set,cfg_var)
    t2 = datetime.datetime.now()
    print("  Elapsed time for creation of variable arrays: "+str(t2-t1)+"\n")
    
    ## Displace past fields onto current position, according to displacement array
    t1 = datetime.datetime.now()
    lag.displace_fields(cfg_set)
    t2 = datetime.datetime.now()
    print("  Elapsed time for the displacement: "+str(t2-t1)+"\n")

    ## Correction of residual movements:
    if cfg_set["resid_disp"]:
        t1 = datetime.datetime.now()
        lag.residual_disp(cfg_set)
        t2 = datetime.datetime.now()
        print("  Elapsed time for the correction of residual movements: "+str(t2-t1)+"\n")
    
    t1 = datetime.datetime.now()
    sta.read_TRT_area_indices(cfg_set,cfg_set["future_disp_reverse"])
    t2 = datetime.datetime.now()
    print("  Elapsed time for reading of indices of the TRT domains: "+str(t2-t1)+"\n")

## Move statistics to collection directory (on non-temporary disk):
def move_statistics(cfg_set_input,cfg_set_tds,path_addon=""):
    path_in  = cfg_set_input["tmp_output_path"]
    path_out = os.path.join(cfg_set_tds["stat_output_path"],path_addon)
    
    print("Move file with statistics to directory %s" % path_out)
    if not os.path.exists(path_out):
        print("   *** Warning: Output path\n          %s\n       created" % path_out)
        os.makedirs(path_out)
    
    for file in os.listdir(path_in):
        if file.startswith(cfg_set_input["t0_str"]+"_stat_pixcount") and file.endswith(".pkl"):
            shutil.move(os.path.join(path_in, file),os.path.join(path_out, file))
        if file.startswith(cfg_set_input["t0_str"]+"_RZC_stat") and file.endswith(".pdf"):
            shutil.move(os.path.join(path_in, file),os.path.join(path_out, file))
    
## Repeat index and statistics reading for a different form width:
def change_form_width_statistics(factor,cfg_set_input,cfg_var,cfg_var_combi):
    """Repeat index and statistics reading for a different form width.

    Parameters
    ----------
    
    factor : float
        Factor of width change.       
    """
    
    ## Get new form width and write it to config file:
    new_form_width = np.round(cfg_set_input["stat_sel_form_width"]*factor)
    print("Change form width from %02dkm to %02dkm and repeat reading in statistics:" %
          (cfg_set_input["stat_sel_form_width"],new_form_width))
    cfg_set_input["stat_sel_form_width"] = new_form_width
    cfg_set_input["stat_sel_form_size"]  = form_size(cfg_set_input["stat_sel_form_width"],
                                                     cfg_set_input["stat_sel_form"])
    
    ## Get new indices from which to read the statistics:
    t1 = datetime.datetime.now()
    sta.read_TRT_area_indices(cfg_set_input,reverse=False)
    sta.read_TRT_area_indices(cfg_set_input,reverse=True)
    t2 = datetime.datetime.now()
    print("  (Changed diameter) Elapsed time for reading of indices of the TRT domains: "+str(t2-t1)+"\n")

    ## Get the new statistics (at the new indices):
    t1 = datetime.datetime.now()
    sta.append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=False)
    sta.append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=True)
    t2 = datetime.datetime.now()
    print("  (Changed diameter) Elapsed time for reading the statistics / pixel counts: "+str(t2-t1)+"\n")

## Clean up displacement arrays in /tmp directory:
def clean_disparr_vararr_tmp(cfg_set,fix_t0_str=None):
    path = cfg_set["tmp_output_path"]
    #print("Remove all displacement files from temporary directory %s" % path)
    check_for_npz = cfg_set["save_type"]=="npy"
    
    t0_str = cfg_set["t0_str"] if fix_t0_str is None else fix_t0_str
    for file in os.listdir(path):
        ## Delete .nc/.npy files of the displacement
        if file.startswith(t0_str) and file.endswith(cfg_set["save_type"]):
            os.remove(os.path.join(path, file))
        ## Delete .pkl files (with TRT information)
        if file.startswith(t0_str+"_TRT_df") and file.endswith(".pkl"):
            os.remove(os.path.join(path, file))
        ## Delete .npz files
        if check_for_npz:
            if file.startswith(t0_str) and file.endswith("npz"):
                os.remove(os.path.join(path, file))














