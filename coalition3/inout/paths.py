""" [COALITION3] Creating names of temporary files and CCS4 input data"""

from __future__ import division
from __future__ import print_function

import os
import configparser
import datetime
import numpy as np
import pysteps as st

## =============================================================================
## FUNCTIONS:

## Provide path to variable array:
def path_creator_vararr(type,var,cfg_set,
                        path=None,t0=None,
                        oflow_source=None,save_type=None,
                        disp_reverse=None):
    """Provide path to variable array."""
    
    type_dict = {"standard":"",
                 "orig":"_orig","_orig":"_orig","disp":"_disp","_disp":"_disp",
                 "disp_resid":"_disp_resid","_disp_resid":"_disp_resid",
                 "disp_resid_combi":"_disp_resid_combi","_disp_resid_combi":"_disp_resid_combi"}
    
    ## Check that name-type is in dictionary:
    if type not in type_dict:
        raise ValueError("Only the following types are known:\n'orig', \
                         '_orig','disp','_disp','disp_resid','disp_resid_combi'")
    
    ## Set string values to be concatenated:    
    path_str          = path if path is not None else cfg_set["tmp_output_path"]
    t0_str            = t0 if t0 is not None else cfg_set["t0"].strftime("%Y%m%d%H%M")
    save_type_str     = save_type  if save_type is not None else cfg_set["save_type"]
    file_ext_verif    = cfg_set["file_ext_verif"]
    if disp_reverse is not None:
        disp_reverse_str = disp_reverse 
    else:
        disp_reverse_str  = "" if not cfg_set["future_disp_reverse"] else "_rev"
    
    ## Concatenate file name:
    filename = "%s%s_%s%s%s%s.%s" % (path_str,
                                     t0_str,
                                     var,
                                     type_dict[type],
                                     file_ext_verif,
                                     disp_reverse_str,
                                     save_type_str)
    return filename           
    
## Provide path to UV displacement array:
def path_creator_UV_disparr(type,cfg_set,path=None,t0=None,
                            oflow_source=None,save_type=None,
                            disp_reverse=None):
    """Provide path to UV displacement array."""
    
    type_dict = {"standard":"","resid":"_resid","resid_combi":"_resid_combi",
                 "vec":"_vec","vec_resid":"_vec_resid","":"",
                 "_resid":"_resid","_resid_combi":"_resid_combi",
                 "_vec":"_vec","_vec_resid":"_vec_resid"}
    
    ## Check that name-type is in dictionary:
    if type not in type_dict:
        raise ValueError("Only the following types are known:\n'standard', \
                         'resid','resid_combi','vec','vec_resid'")
    
    ## Set string values to be concatenated:    
    path_str          = path if path is not None else cfg_set["tmp_output_path"]
    t0_str            = t0 if t0 is not None else cfg_set["t0"].strftime("%Y%m%d%H%M")
    oflow_source_str  = oflow_source if oflow_source is not None else cfg_set["oflow_source"]
    file_ext_vefif    = cfg_set["file_ext_verif"]
    if disp_reverse is not None:
        disp_reverse_str = disp_reverse 
    else:
        disp_reverse_str  = "" if not cfg_set["future_disp_reverse"] else "_rev"
    if save_type is not None:
        save_type_str = save_type 
    else:
        save_type_str = cfg_set["save_type"]
        if save_type_str=="npy": save_type_str = "npz"
    
    ## Concatenate file name:
    filename = "%s%s_%s_disparr_UV%s%s%s.%s" % (path_str, t0_str,
                                                oflow_source_str,
                                                type_dict[type],
                                                file_ext_vefif,
                                                disp_reverse_str,
                                                save_type_str)
    return filename
    
## Provide path to different data sources:
def path_creator(t, var, source, cfg_set):
    """Provide path to different data sources."""
    config = configparser.RawConfigParser()
    config.read(os.path.join(cfg_set["CONFIG_PATH"],"input_data.cfg"))
    var_path_str = var+"_path"
    
    if source == "RADAR":
        config_ds = config["radar_read"]
        var_path_str    = var[:3]+"_path"
        var_path        = config_ds[var_path_str]
        path_fmt        = config_ds["path_fmt"]
        fn_pattern      = config_ds["fn_pattern"]
        fn_ext          = config_ds["fn_ext"]
        etop_EZC        = config_ds["etop_EZC"]
                
        fn_pattern = var[:3]+fn_pattern
        if var=="EZC":
            fn_ext = "8"+etop_EZC
        elif var in ["EZC15","EZC20","EZC45","EZC50",]:
            fn_ext = "8"+var[3:]
        elif var=="BZC":
            fn_ext = "845"
        elif var=="MZC":
            fn_ext = "850"
            
        ## Get file paths from the future if reverse flow field should
        ## calculated:
        if not cfg_set["future_disp_reverse"]:
            file_paths = st.io.find_by_date(t, var_path, path_fmt, fn_pattern, 
                                            fn_ext, cfg_set["timestep"],
                                            num_prev_files=cfg_set["n_past_frames"])
        else: 
            filenames, timestamps = st.io.find_by_date(t, var_path, path_fmt, fn_pattern, 
                                            fn_ext, cfg_set["timestep"],
                                            num_next_files=cfg_set["n_past_frames"])
            file_paths = (filenames[::-1],timestamps[::-1]) #(filenames,timestamps)
        return file_paths
        
    elif source == "THX":
        config_ds = config["light_read"]
        var_path        = config_ds["THX_path"]
        path_fmt        = config_ds["path_fmt"]
        fn_pattern      = config_ds["fn_pattern"]
        fn_ext          = config_ds["fn_ext"]
        
        if t.hour==0 and t.minute==0: t = t-datetime.timedelta(minutes=cfg_set["timestep"])
        file_paths = st.io.find_by_date(t, var_path, path_fmt, fn_pattern, 
                                        fn_ext, cfg_set["timestep"], 0)
        return file_paths
        
    elif source == "TRT":
        config_ds = config["TRT_read"]
        var_path        = config_ds["TRT_path"]
        path_fmt        = config_ds["path_fmt"]
        fn_pattern      = config_ds["fn_pattern"]
        fn_ext          = config_ds["fn_ext"]
        file_paths = st.io.find_by_date(t, var_path, path_fmt, fn_pattern, 
                                        fn_ext, cfg_set["timestep"], 0)
        return file_paths
        
    elif source == "COSMO_WIND":
        config_ds = config["wind_read"]
        var_path        = config_ds["Wind_path"]
        path_fmt        = config_ds["path_fmt"]
        fn_pattern      = config_ds["fn_pattern"]
        fn_ext          = config_ds["fn_ext"]
        
        t_last_model_run = '%02d' % (int(t.hour/3)*3)
        t_fcst_model     = '%02d' % (t.hour%3)
        file_path = str(datetime.datetime.strftime(t, var_path+path_fmt+'/'+fn_pattern) + t_last_model_run + '_' +
                        t_fcst_model + '_cosmo-1_UV_swissXXL.' + fn_ext)
        
        timestamp = t.replace(minute=0)
        return (file_path, timestamp)
        
    elif source == "COSMO_CONV":
        config_ds = config["conv_read"]
        var_path        = config_ds["Conv_path"]
        path_fmt        = config_ds["path_fmt"]
        fn_pattern      = config_ds["fn_pattern"]
        fn_ext          = config_ds["fn_ext"]
        
        t_last_model_run = '%02d' % (int(t.hour/3)*3)
        t_fcst_model     = '%02d' % (t.hour%3)
        if t_last_model_run=="0": t_last_model_run="00"
        
        ## Do not use analysis at SYNOP times (not available for 50min),
        ## but the forecast from three hours ago.
        if t_fcst_model=="00":
            t_last_model_run = '%02d' % (int((t.hour-1)/3)*3)
            t_fcst_model     = "03"
            
        file_path = str(datetime.datetime.strftime(t, var_path+path_fmt+'/'+fn_pattern) + t_last_model_run + '_' +
                        t_fcst_model + '_cosmo-1_convection_swiss.' + fn_ext)

        timestamp = t.replace(minute=0)
        return (file_path, timestamp)
        
    elif source == "SEVIRI":
        config_ds = config["sat_read"]
        var_path        = config_ds["SAT_path"]
        path_fmt        = config_ds["path_fmt"]
        fn_pattern      = config_ds["fn_pattern"]
        fn_ext          = config_ds["fn_ext"]
        
        ## Input 2 after MSG due to satellite change before March 20, 2018
        if t < datetime.datetime(2018,03,20): fn_pattern = fn_pattern[:3]+"2"+fn_pattern[4:]
        
        file_paths = st.io.find_by_date(t, var_path, path_fmt, fn_pattern,
                                       fn_ext, cfg_set["timestep"], 0)
        return file_paths
        
    else:
        raise NotImplementedError("So far path_creator implemented for radar products (RZC, BZC...), SEVIRI, THX, and COSMO Conv")
      
## Function returning the path where the stat_pixc pickle files are stored:
def get_stat_path(user_argv_path):
    """Function returning the path where the stat_pixc pickle files are stored"""
    if user_argv_path is not None:
        print("\nYou provided the following path to the '<Datetime>_stat_pixcount*.pkl' files:\n  %s" % user_argv_path)
        correct_path = raw_input("  Is this path correct? [y,n] ")
        while (correct_path!="n" and correct_path!="y"): correct_path = raw_input("  Is this path correct? [y,n] ")
        if correct_path=="y":
            if os.path.exists(user_argv_path):
                #print("It is assumed that in this directory only "+ \
                #      "'<Datetime>_stat_pixcount*.pkl' files are stored:\n  %s\n" % user_argv_path)
                return(os.path.normpath(user_argv_path))
            else: print("This path does not exist!")
            
    user_path = raw_input("\nPlease provide the path to the '<Datetime>_stat_pixcount*.pkl' files:\n  ")
    while not os.path.exists(user_path):
        print("This path does not exist: %s" % user_path)
        user_path = raw_input("\nPlease provide the path to the '<Datetime>_stat_pixcount*.pkl' files:\n  ")
    #print("It is assumed that in this directory only "+ \
    #      "'<Datetime>_stat_pixcount*.pkl' files are stored:\n  %s\n" % user_path)
    return(os.path.normpath(user_path))
    
## Function returning the path where the pickle file with the logging information is stored:
def get_log_path(user_argv_path):
    """Function returning the path where the stat_pixc pickle files are stored"""
    if user_argv_path is not None:
        print("\nYou provided the following path to the 'Training_Dataset_Processing_Status.pkl' file:\n  %s" % user_argv_path)
        correct_path = raw_input("  Is this path correct? [y,n] ")
        while (correct_path!="n" and correct_path!="y"): correct_path = raw_input("  Is this path correct? [y,n] ")
        if correct_path=="y":
            if os.path.exists(user_argv_path):
                #print("It is assumed that in this directory only "+ \
                #      "'<Datetime>_stat_pixcount*.pkl' files are stored:\n  %s\n" % user_argv_path)
                return(os.path.normpath(user_argv_path))
            else: print("This path does not exist!")
            
    user_path = raw_input("\nPlease provide the path to the 'Training_Dataset_Processing_Status.pkl' file:\n  ")
    while not os.path.exists(user_path):
        print("This path does not exist: %s" % user_path)
        user_path = raw_input("\nPlease provide the path to the 'Training_Dataset_Processing_Status.pkl' file:\n  ")
    #print("It is assumed that in this directory only "+ \
    #      "'<Datetime>_stat_pixcount*.pkl' files are stored:\n  %s\n" % user_path)
    return(os.path.normpath(user_path))
      