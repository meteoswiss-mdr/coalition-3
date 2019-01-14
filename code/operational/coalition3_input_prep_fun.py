""" Functions for NOSTRADAMUS_1_input_prep.py:

The following functions are assembled here for the input data generation
of NOSTRADAMUS:
"""

from __future__ import division
from __future__ import print_function

import sys
import os
import configparser
import datetime
import pdb

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import pysteps as st
import ephem
from netCDF4 import Dataset
from scipy import ndimage, signal, interpolate, spatial

from joblib import Parallel, delayed
import multiprocessing

#from contextlib import contextmanager
#import warnings
#import ast

sys.path.insert(0, '/data/COALITION2/database/radar/ccs4/python')
import metranet
sys.path.insert(0, '/opt/users/jmz/monti-pytroll/packages/mpop')
from mpop.satin import swisslightning_jmz, swisstrt, swissradar

## ===============================================================================
## FUNCTIONS:

## Get settings from config file and some additional static settings:
def get_config_info(CONFIG_PATH,CONFIG_FILE_set,CONFIG_PATH_set,CONFIG_FILE_var,CONFIG_FILE_var_combi,t0_str):
    """Get information from configuration file and return in a dictionary "cfg_set".
    Furthermore, get data from variable configuration file (cfg_var.csv) and return
    in a pandas.dataframe "cfg_var"

    Parameters
    ----------
    
    CONFIG_PATH : str
        Path to config file.
        
    CONFIG_FILE_set : str
        Name of settings config file.
        
    CONFIG_FILE_var : str
        Name of variable config file (.csv).
        
    t0_str : str
        String of the time t0 (displacement target time)
        
    Output
    ------

    cfg_set : dict
        Dictionary with the basic variables used throughout the code:

    Dictionary elements
    -------------------
    
    See Manual.docx
     
    """
    
    ## ===== Make static settings: ======================================
    oflow_source  = "RZC"
    square_domain = False
    oflow_method_name = "lucaskanade" # lucaskanade or DARTS
    adv_method        = "semilagrangian"
    #n_past_frames     = 1 if oflow_method_name == "lucaskanade" else 9

    ## Get information on date and time
    t0      = datetime.datetime.strptime(t0_str, "%Y%m%d%H%M")
    t0_orig = datetime.datetime.strptime(t0_str, "%Y%m%d%H%M")
    t0_doy  = t0.timetuple().tm_yday

    ## Visualization parameters
    colorscale      = "MeteoSwiss" # MeteoSwiss or STEPS-BE
    motion_plot     = "quiver"     # streamplot or quiver
    
    
    ## ===== Import configuration information on variables: =============
    dtype_cfg_var = {'N_CATEGORIES': 'uint8', 'MIN_VALUE': 'float32'}
    dtype_cfg_var.update(dict.fromkeys(['PREDICTOR', 'LAGRANGIAN', 'PAST_OBS', 'SMOOTH', 'SUM', 'MEAN', 'STDDEV','IQR', \
                                        'PERC01', 'PERC05', 'PERC30','PERC50', 'PERC80', 'PERC95','PERC99', 'MIN', 'MAX', \
                                        'PC_NONNAN', 'PC_NONZERO', 'PC_LT55','PC_CAT','READ','READ_IF','VARIABILITY'], 'bool'))
    dtype_cfg_var_combi = {'dt': 'bool','PREDICTOR': 'bool'}
    cfg_var       = pd.read_csv("%s/%s" % (CONFIG_PATH_set,CONFIG_FILE_var),delimiter=";",dtype=dtype_cfg_var)
    cfg_var_combi = pd.read_csv("%s/%s" % (CONFIG_PATH_set,CONFIG_FILE_var_combi),delimiter=";",dtype=dtype_cfg_var_combi)

    ## ===== Read the data source/output configuration file: ============
    config = configparser.RawConfigParser()
    config.read("%s%s" % (CONFIG_PATH,CONFIG_FILE_set))

    config_ds = config["datasource"]
    root_path       = config_ds["root_path"]
    output_path     = config_ds["output_path"]
    RZC_path        = config_ds["RZC_path"]
    #wind_path       = config_ds["wind_path"]
    #conv_path       = config_ds["conv_path"]
    if oflow_source == "RZC":
        oflow_source_path = RZC_path 
    else:
        raise NotImplementedError("So far displacement array retrieval only implemented for RZC")

    ## Read further config information on displacement
    config_bs = config["basicsetting"]
    verbose             = config_bs["verbose"]=="True"
    timestep            = int(config_bs["timestep"])
    n_integ             = int(config_bs["n_integ"])
    n_past_frames       = int(config_bs["n_past_frames"])
    n_past_frames_resid = int(config_bs["n_past_frames_resid"])
    xy_ext              = (int(config_bs["xy_ext_x"]), int(config_bs["xy_ext_y"]))
    use_precalc_XYW     = config_bs["use_precalc_XYW"]=="True"
    R_threshold         = float(config_bs["R_threshold"])
    R_thresh_meth       = config_bs["R_thresh_meth"]


    instant_resid_corr  = config_bs["instant_resid_corr"] == 'True'
    resid_method        = config_bs["resid_method"] if not instant_resid_corr else "None"
    resid_disp          = False if resid_method=="None" else True
    #resid_disp          = config_bs["resid_disp"]=='True'
    resid_disp_onestep  = True if resid_method=="Onestep" and resid_disp else False
        
    generating_train_ds = config_bs["generating_train_ds"]=='True'
    precalc_UV_disparr  = config_bs["precalc_UV_disparr"]=='True' and generating_train_ds
    future_disp_reverse = config_bs["future_disp_reverse"]=='True' and generating_train_ds
    #timestep_future     = int(config_bs["timestep_future"])
    #n_integ_future      = int(config_bs["n_integ_future"])
    UV_precalc_path     = config_ds["UV_precalc_path"]
    time_change_factor  = -1 if future_disp_reverse else 1
    
    displ_TRT_cellcent  = config_bs["displ_TRT_cellcent"]=='True'
    
    #resid_disp_onestep  = config_bs["resid_disp_onestep"]=='True'
    R_threshold_resid   = float(config_bs["R_threshold_resid"])
    R_thresh_meth_resid = config_bs["R_thresh_meth_resid"]
    UV_inter            = config_bs["UV_inter"]=='True'
    decl_grid_resid     = int(config_bs["decl_grid_resid"])
    zero_interpol       = config_bs["zero_interpol"]=='True'
    inter_fun_resid     = config_bs["inter_fun_resid"]
    epsilon_resid       = int(config_bs["epsilon_resid"])

    ## Read further config information, which variables should be displaced
    #config_vr = config["variables"]
    #var_list        = config_vr["var_list"].split(',')
    var_list        = cfg_var["VARIABLE"][cfg_var["READ_IF"]].tolist()
    var_combi_list  = cfg_var_combi["VARIABLE"][cfg_var_combi["PREDICTOR"]].tolist()

    ## Read further config information on displacement
    config_sc = config["statistics_calculation"]
    stat_sel_form       = config_sc["stat_sel_form"]
    stat_sel_form_width = int(config_sc["stat_sel_form_width"])
    min_TRT_rank        = float(config_sc["min_TRT_rank"])
    save_TRT_domain_map = config_sc["save_TRT_domain_map"]=='True'
    save_stat_ctrl_imag = config_sc["save_stat_ctrl_imag"]=='True'
    stat_list       = cfg_var.columns[np.where(cfg_var.columns=="SUM")[0][0]:np.where(cfg_var.columns=="MAX")[0][0]+1]
    pixcount_list   = cfg_var.columns[np.where(cfg_var.columns=="PC_NONNAN")[0][0]:]
    
    ## Get size of domain of interest:
    stat_sel_form_size = form_size(stat_sel_form_width,stat_sel_form)
    
    ## Read further config information, which variables should be displaced
    config_cr = config["conv_read"]
    smooth_conv        = config_cr["smooth_conv"]=="True"
    smooth_sig         = float(config_cr["smooth_sig"])

    ## Read further config information on updates
    config_up = config["file_handling"]
    delete_prec     = config_up["delete_prec"]=='True'
    save_type       = config_up["save_type"]

    ## Read further config information on verification and skill score calculation
    config_ve = config["verification"]
    verify_disp     = config_ve["verify_disp"]=='True'
    verif_param     = config_ve["verif_param"]
    scores_list     = config_ve["scores_list"].split(',')
    n_stat          = config_ve["n_stat"]

    ## Dictionaries of SEVIRI channel names and units
    #sat_long_names = {0.6:'VIS006', 0.8:'VIS008', 1.6:'IR_016', 3.9:'IR_039', 6.2:'WV_062', 7.3:'WV_073',\
    #                  8.7:'IR_087', 9.7:'IR_097', 10.8:'IR_108', 12.0:'IR_120', 13.4:'IR_134',\
    #                  'VIS006':'VIS006', 'VIS008':'VIS008', 'IR_016':'IR_016', 'IR_039':'IR_039', 'WV_062':'WV_062', 'WV_073':'WV_073',\
    #                  'IR_087':'IR_087', 'IR_097':'IR_097', 'IR_108':'IR_108', 'IR_120':'IR_120', 'IR_134':'IR_134', 'HRV':'HRV'}
    #sat_units      = {0.6:'percent', 0.8:'percent', 1.6:'percent', 3.9:'K', 6.2:'K', 7.3:'K',\
    #                  8.7:'K', 9.7:'K', 10.8:'K', 12.0:'K', 13.4:'K',\
    #                  'VIS006':'percent', 'VIS008':'percent', 'IR_016':'percent', 'IR_039':'K', 'WV_062':'K', 'WV_073':'K',\
    #                  'IR_087':'K', 'IR_097':'K', 'IR_108':'K', 'IR_120':'K', 'IR_134':'K', 'HRV':'percent'}

    
    abbrev_dict = {"csi": "CSI", "sedi": "SEDI", "hk": "HK"} #,
    type_dict = {}; unit_dict = {}; source_dict = {}; minval_dict = {}
    for i in range(len(var_list)):
        abbrev_dict.update({var_list[i]: cfg_var["DESCRITPION"][cfg_var["READ_IF"]].tolist()[i]})
        type_dict.update({var_list[i]:   cfg_var["DATA_TYPE"][cfg_var["READ_IF"]].tolist()[i]})
        unit_dict.update({var_list[i]:   cfg_var["UNIT"][cfg_var["READ_IF"]].tolist()[i]})
        source_dict.update({var_list[i]: cfg_var["SOURCE"][cfg_var["READ_IF"]].tolist()[i]})
        minval_dict.update({var_list[i]: cfg_var["MIN_VALUE"][cfg_var["READ_IF"]].tolist()[i]})
    if displ_TRT_cellcent:
        abbrev_dict.update({"TRT": "TRT Cell Centre"})
        type_dict.update({"TRT": "int32"})
        unit_dict.update({"TRT": "Cell ID"})
        source_dict.update({"TRT": "TRT"})
        minval_dict.update({"TRT": None})

    type_dict_combi = {}; unit_dict_combi = {}; source_dict_combi = {}; abbrev_dict_combi = {}
    for i in range(len(var_combi_list)):
        abbrev_dict_combi.update({var_combi_list[i]: return_var_combi_information(var_combi_list[i],cfg_var_combi,abbrev_dict,"abbrev")})
        type_dict_combi.update({var_combi_list[i]:   return_var_combi_information(var_combi_list[i],cfg_var_combi,type_dict,"data_type")})
        unit_dict_combi.update({var_combi_list[i]:   return_var_combi_information(var_combi_list[i],cfg_var_combi,unit_dict,"unit")})
        source_dict_combi.update({var_combi_list[i]: return_var_combi_information(var_combi_list[i],cfg_var_combi,source_dict,"source")})

        
    TRT_cols = ['date', 'lon', 'lat', 'iCH', 'jCH', 'ell_L', 'ell_S', 'area', 'vel_x', 'vel_y', 'det',
                'RANKr', 'CG', 'CG_plus', 'CG_minus', 'perc_CG_plus', 'ET45', 'ET45m', 'ET15', 'ET15m', 'VIL',
                'maxH', 'maxHm', 'POH', 'RANK', 'Dvel_x', 'Dvel_y', 'angle']
    TRT_dtype = [np.object, np.float32, np.float32, np.int16, np.int16, np.float32, np.float32, np.int32, np.float32, np.float32, np.int16,
                 np.int16, np.int16, np.int16, np.int16, np.int16, np.float32, np.float32, np.float32, np.float32, np.float32,
                 np.float32, np.float32, np.float32, np.int16, np.float32, np.float32, np.float32]
    type_dict_TRT = {}
    for i in range(len(TRT_cols)): type_dict_TRT.update({TRT_cols[i]: TRT_dtype[i]})
    ## Dictionaries of abbreviations and plotting names
    #abbrev_dict = {"csi": "CSI", "sedi": "SEDI", "hk": "HK",
    #               "RZC": "Rain Rate", "BZC": "POH", "LZC": "VIL", "MZC": "MESHS", "EZC": "EchoTop", "EZC15": "EchoTop 15dBZ", "EZC45": "EchoTop 45dBZ",
    #               "HRV": "HRV", "VIS006": "VIS 0.6", "VIS008": "VIS 0.8", "IR_108": "IR 10.8", "IR_120": "IR 12.0", "WV_062": "WV 6.2",
    #               "THX_dens": "Lightning density", "THX_densCG": "Lightning density (Cloud-to-ground)", "THX_densIC": "Lightning density (Intra/Inter-cloud)",
    #               "THX_curr_abs": "Absolute lightning current", "THX_curr_pos": "Positive lightning current", "THX_curr_neg": "Negative lightning current"}
                   
    ## Dictionaries of data types
    #type_dict = {"RZC": np.float32,"BZC": np.float32,"LZC": np.float32,"MZC": np.float32,"EZC": np.float32,"EZC15": np.float32,"EZC45": np.float32,
    #             "HRV": np.float32,"VIS006": np.float32,"VIS008": np.float32,"IR_108": np.float32,"IR_120": np.float32,"WV_062": np.float32,"THX": np.int8,
    #             "TWATER": np.float32,"CAPE_MU": np.float32,"CAPE_ML": np.float32,"CIN_MU": np.float32,"CIN_ML": np.float32,"SLI": np.float32}
                   
    ## Save key input variables in dictionary
    cfg_set = {
                  "t0":                  t0,
                  "t0_str":              t0_str,
                  "t0_doy":              t0_doy,
                  "t0_orig":             t0_orig,
                  "oflow_method_name":   oflow_method_name,
                  "adv_method":          adv_method,
                  "R_threshold":         R_threshold,
                  "R_thresh_meth":       R_thresh_meth,
                  "colorscale":          colorscale,
                  "motion_plot":         motion_plot,
                  "root_path":           root_path,
                  "output_path":         output_path,
                  #"wind_path":           wind_path,
                  #"conv_path":           conv_path,
                  "oflow_source":        oflow_source,
                  "oflow_source_path":   oflow_source_path,
                  "square_domain":       square_domain,
                  "verbose":             verbose,
                  "timestep":            timestep,
                  "n_integ":             n_integ,
                  "n_past_frames":       n_past_frames,
                  "n_past_frames_resid": n_past_frames_resid,
                  "xy_ext":              xy_ext,
                  "use_precalc_XYW":     use_precalc_XYW,
                  "generating_train_ds": generating_train_ds,
                  "future_disp_reverse": future_disp_reverse,
                  "time_change_factor":  time_change_factor,
                  #"timestep_future":     timestep_future,
                  #"n_integ_future":      n_integ_future,
                  "precalc_UV_disparr":  precalc_UV_disparr,
                  "UV_precalc_path":     UV_precalc_path,
                  "var_list":            var_list,
                  "var_combi_list":      var_combi_list,
                  "stat_sel_form":       stat_sel_form,
                  "stat_sel_form_width": stat_sel_form_width,
                  "stat_sel_form_size":  stat_sel_form_size,
                  "save_TRT_domain_map": save_TRT_domain_map,
                  "save_stat_ctrl_imag": save_stat_ctrl_imag,
                  "min_TRT_rank":        min_TRT_rank,
                  "stat_list":           stat_list,
                  "pixcount_list":       pixcount_list,
                  "displ_TRT_cellcent":  displ_TRT_cellcent,
                  "CONFIG_PATH":         CONFIG_PATH,
                  "CONFIG_FILE_set":     CONFIG_FILE_set,
                  "CONFIG_FILE_var":     CONFIG_FILE_var,
                  "delete_prec":         delete_prec,
                  "save_type":           save_type,
                  "verify_disp":         verify_disp,
                  "verif_param":         verif_param,
                  "scores_list":         scores_list,
                  "n_stat":              n_stat,
                  #"sat_long_names":      sat_long_names,
                  #"sat_units":           sat_units,
                  "abbrev_dict":         abbrev_dict,
                  "type_dict":           type_dict,
                  "unit_dict":           unit_dict,
                  "source_dict":         source_dict,
                  "minval_dict":         minval_dict,
                  "abbrev_dict_combi":   abbrev_dict_combi,
                  "type_dict_combi":     type_dict_combi,
                  "unit_dict_combi":     unit_dict_combi,
                  "source_dict_combi":   source_dict_combi,
                  "instant_resid_corr":  instant_resid_corr,
                  "resid_method":        resid_method,
                  "resid_disp":          resid_disp,
                  "resid_disp_onestep":  resid_disp_onestep,
                  "R_threshold_resid":   R_threshold_resid,
                  "R_thresh_meth_resid": R_thresh_meth_resid,
                  "UV_inter":            UV_inter,
                  "decl_grid_resid":     decl_grid_resid,
                  "inter_fun_resid":     inter_fun_resid,
                  "epsilon_resid":       epsilon_resid,
                  "zero_interpol":       zero_interpol,
                  "smooth_conv":         smooth_conv,
                  "smooth_sig":          smooth_sig,
                  "TRT_cols":            TRT_cols,
                  "TRT_dtype":           TRT_dtype,
                  "type_dict_TRT":       type_dict_TRT
                 }

    ## Add extention to filenames if verification is performed:
    file_ext_verif  = "_"+str(cfg_set[cfg_set["verif_param"]]) if cfg_set["verify_disp"] else ""
    cfg_set["file_ext_verif"] = file_ext_verif
    
    ## Make sanity checks:
    if 60%timestep!=0:
        raise ValueError("60min modulo Timestep = %smin is not zero, choose a different one" % timestep)
        
    return(cfg_set, cfg_var, cfg_var_combi)  

"""    
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
"""
    
def return_var_combi_information(var_combi,cfg_var_combi,var_dict,type):
    if type=="abbrev":
        if cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"OPERATION"].values=="diff":
            return_val = "Difference between %s and %s" % \
                         (var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_1"].values[0]],
                          var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_2"].values[0]])
        elif cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"OPERATION"].values=="sum_2diff":
            return_val = "Difference between %s and %s, plus two times %s" % \
                         (var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_1"].values[0]],
                          var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_2"].values[0]],
                          var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_3"].values[0]])
        elif cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"OPERATION"].values=="none":
            return_val = var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_1"].values[0]]
        else: raise ValueError("OPERATION not yet implemented")
    if type=="unit" or type=="data_type" or type=="source":
        if cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"OPERATION"].values=="diff":
            unit_list = [var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_1"].values[0]],
                         var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_2"].values[0]]]
        elif cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"OPERATION"].values=="sum_2diff":
            unit_list = [var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_1"].values[0]],
                         var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_2"].values[0]],
                         var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_3"].values[0]]]
        elif cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"OPERATION"].values=="none":
            unit_list = [var_dict[cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"VARIABLE_1"].values[0]]]
        else: raise ValueError("OPERATION not yet implemented")
        if len(set(unit_list))>1:
            raise ValueError("Combination of more than two units/data types/sources, not possible for such operations")
        else:
            return_val = unit_list[0]
        if cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var_combi,"dt"].values[0]:
            raise ValueError("dt channel combinations not yet implemented")
    return return_val
        
    
def form_size(stat_sel_form_width,stat_sel_form):
    if stat_sel_form == "square":
        size = int(stat_sel_form_width**2)
        return size
    elif stat_sel_form == "circle":
        X, Y = np.meshgrid(np.arange(0,stat_sel_form_width*2),
                           np.arange(0,stat_sel_form_width*2))
        interior = ((X-int(stat_sel_form_width))**2 + 
                    (Y-int(stat_sel_form_width))**2) < (stat_sel_form_width/2.)**2
        size = int(np.sum(interior))       
        return size
    else: raise ValueError("stat_sel_form can only be 'square' or 'circle'")
    
## Print Nostradamus Logo:
def print_logo():
    logo_str = "" + \
    "     _.-.___ \n"+ \
    "    ( (  )  )_-_-_- - \n"+ \
    "     ( (   )- - -       'For forty years the rainbow will not be seen.\n"+ \
    "      ( ( ) )            For forty years it will be seen every day.\n"+ \
    "     (_(___)_)           The dry earth will grow more parched,\n"+ \
    "        _<               and there will be great floods when it is seen.'\n"+ \
    "       / /\                                    Nostradamus\n"+ \
    "    _____\__________ \n"
    print(logo_str)
    
## Print information before running script:
def print_config_info(cfg_set,CONFIG_FILE_set,CONFIG_FILE_var):
    """Print information before running script"""
    print("\n-------------------------------------------------------------------------------------------------------\n")
    print_logo()
    if cfg_set["verif_param"] == "R_threshold":
        unit_verifparam = " mm/h" if cfg_set["R_thresh_meth"] == "fix" else "% quantile"
    else: unit_verifparam = ""
    print_str = '    Configuration of NOSTRADAMUS input date preparation procedure:'+ \
    '\n      Date:             '+cfg_set["t0"].strftime("%Y-%m-%d %H:%M")+ \
    '\n      Config files:     '+CONFIG_FILE_set+' (Settings) & '+CONFIG_FILE_var+' (Variables)'+ \
    '\n      Reverse mode:     '+str(cfg_set["future_disp_reverse"])+ \
    '\n      Variables displ.: '+str(cfg_set["var_list"])+ \
    '\n      Save file type:   '+'.'+str(cfg_set["save_type"])+' file'+ \
    '\n      Oflow data:       '+cfg_set["oflow_source"]+ \
    '\n      Oflow method:     '+cfg_set["oflow_method_name"]+ \
    '\n      Adv method:       '+cfg_set["adv_method"]+ \
    '\n      Timestep:         '+str(cfg_set["timestep"])+"min"+ \
    '\n      Integr. steps:    '+str(cfg_set["n_integ"])+ \
    '\n      Use precalc. UV:  '+str(cfg_set["precalc_UV_disparr"])+ \
    '\n      Instant. corr:    '+str(cfg_set["instant_resid_corr"])+ \
    '\n      Verification:     '+str(cfg_set["verify_disp"])
    #'\n  Square domain:    '+str(cfg_set["square_domain"])+ \
    if cfg_set["verify_disp"]:
        print_str = print_str+'\n      Verif. param:     '+str(cfg_set["verif_param"])+'='+ \
                              str(cfg_set[cfg_set["verif_param"]])+unit_verifparam
    #'\n  Verif. param:     '+str(cfg_set["verif_param"])+'='+str(cfg_set[cfg_set["verif_param"]])+unit_verifparam+'\n'      
    print_str = print_str+'\n'
    print(print_str)
    
    #if cfg_set["future_disp_reverse"]:
    #    print_str = "    ==============================================================================\n" + \
    #                "    ==========                                                          ==========\n" + \
    #                "    ==========  RUN IN REVERSE MODE (for creation of training dataset)  ==========\n" + \
    #                "    ==========                                                          ==========\n" + \
    #                "    ==============================================================================\n"
    #    print(print_str)
    print("-------------------------------------------------------------------------------------------------------\n")
    
## Check whether temporary subdirectories are present, otherwise create those:
def check_create_tmpdir(cfg_set):
    """Check tmp directory whether sub-directories for n_integ timesteps exist,
    if not, create those.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
    """
    
    ## Calculate n_integ time deltas:
    t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=cfg_set["timestep"])
    
    ## Check that for each delta, a subdirectory exists.
    for t_d in t_delta:
        t = cfg_set["t0"] - t_d
        path_name = u"%s/tmp/%s" % (cfg_set["root_path"], t.strftime("%Y%m%d%H%M"))
        if not os.path.exists(path_name):
            os.makedirs(path_name); print("  Created directory: %s" % path_name)

            			
## Check whether precalculated displacement array for training is already existent, otherwise create it.
def check_create_precalc_disparray(cfg_set):
    """Check whether precalculated displacement array for training is already existent, otherwise create it.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py         
    """
    ## Change values in cfg_set to fit the requirements for the precalculation
    ## 288 integration steps for 24h coverage
    cfg_set_precalc = cfg_set.copy()
    cfg_set_precalc["n_integ"] = int(24*60/cfg_set["timestep"])
    if not cfg_set["future_disp_reverse"]:
        cfg_set_precalc["t0"] = datetime.datetime(cfg_set["t0"].year,cfg_set["t0"].month,cfg_set["t0"].day,0,0)+datetime.timedelta(days=1)
    else:
        cfg_set_precalc["t0"] = datetime.datetime(cfg_set["t0"].year,cfg_set["t0"].month,cfg_set["t0"].day,0,0)
        
    ## Check path for precalculated displacement array, and its dimensions:
    path_name = path_creator_UV_disparr("standard",cfg_set,
                                        path=cfg_set["UV_precalc_path"],
                                        t0=cfg_set["t0"].strftime("%Y%m%d"))

    ## If not existent, create new one, else check dimensions,
    ## if correct, update, else, create new one.
    if os.path.isfile(path_name):
        #disparr = np.load(path_name)
        disparr = load_file(path_name,"Vx")
        if disparr.ndim!=3 or disparr.shape[0]!=(cfg_set_precalc["n_integ"]):
            print("   Dimensions of existent precalculated displacement array do not match current settings.\n   Ndim: "+
                  str(disparr.ndim)+"\n   Shape: "+str(disparr.shape[0])+" instead of "+str(cfg_set_precalc["n_integ"])+"\n   A new one is created")
            create_new_disparray(cfg_set_precalc,precalc=True)
        else:
            print("Found precalculated disparray file")
        #if cfg_set["precalc_UV_disparr"] and cfg_set["generating_train_ds"]:
        #    get_the_disp_array for specific t0
    else:
        print("Found no precalculated disparray file, create such.")
        create_new_disparray(cfg_set_precalc,precalc=True)
                
## Get datetime objects of integration steps:
def datetime_integ_steps(cfg_set):
    """Get datetime objects of integration steps"""
    t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=cfg_set["timestep"])
    t_integ = cfg_set["t0"] - cfg_set["time_change_factor"]*t_delta
    return t_integ

## Check whether Displacement array is already existent, otherwise create it.
## If existent, replace with newest displacement:
def check_create_disparray(cfg_set):
    """Check whether Displacement array is already existent, otherwise create it.
    If existent, replace with newest displacement.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py         
    """
    
    ## Check path for displacement array, and its dimensions:
    path_name = path_creator_UV_disparr("standard",cfg_set)                                     

    ## In case more than one day is within integration period, check whether all precalculated files are available:
    all_precalc_files_existent = True
    t_integ_ls = datetime_integ_steps(cfg_set)
    if len(np.unique([t_integ.day for t_integ in t_integ_ls]))>1:
        for t_integ in t_integ_ls:
            path_name_precalc = path_creator_UV_disparr("standard",cfg_set,path=cfg_set["UV_precalc_path"],
                                                        t0=t_integ.strftime("%Y%m%d"))
            if not os.path.isfile(path_name_precalc): all_precalc_files_existent = False
        
    ## If not existent, create new one, else check dimensions,
    ## if correct, update, else, create new one.
    if os.path.isfile(path_name):
        #disparr = np.load(path_name)
        disparr = load_file(path_name,"Vx")
        ## Check that existing dimensions match the required dimensions:
        if disparr.ndim!=3 or disparr.shape[0]!=(cfg_set["n_integ"]):
            print("   Dimensions of existent displacement array do not match current settings.\n   Ndim: "+
                  str(disparr.ndim)+"\n   Shape: "+str(disparr.shape[0])+"\n   A new one is created")
            if cfg_set["precalc_UV_disparr"] and all_precalc_files_existent:
                read_disparray_from_precalc(cfg_set)
            else:
                create_new_disparray(cfg_set)
    else:
        if cfg_set["precalc_UV_disparr"] and all_precalc_files_existent:
            read_disparray_from_precalc(cfg_set)
        else:
            create_new_disparray(cfg_set)

## Read UV and Displacement arrays from precalculated array:
def read_disparray_from_precalc(cfg_set):
    """Read UV and Displacement arrays from precalculated array. This function
    is only executed, if the respective precalculated UV- and displacement array
    is available, and t0 is not too close to midnight.      
    """

    t_integ_ls = datetime_integ_steps(cfg_set)
    if len(np.unique([t_integ.day for t_integ in t_integ_ls]))<=1:    
        path_name = path_creator_UV_disparr("standard",cfg_set,path=cfg_set["UV_precalc_path"],
                                            t0=cfg_set["t0"].strftime("%Y%m%d"))
        #path_name = "%s/%s_%s_disparr_UV%s.%s" % (cfg_set["UV_precalc_path"], cfg_set["t0"].strftime("%Y%m%d"), #(cfg_set["t0"]-datetime.timedelta(days=1)).strftime("%Y%m%d"),
        #                                          cfg_set["oflow_source"],cfg_set["file_ext_verif"],
        #                                          cfg_set["save_type"])
        print("   Read from precalculated UV-disparray...\n      %s" % path_name)

        ## Get indices to timesteps of interest:
        if not cfg_set["future_disp_reverse"]:
            t2_ind = int((24*60-cfg_set["t0"].hour*60-cfg_set["t0"].minute+cfg_set["n_integ"]*cfg_set["timestep"])/cfg_set["timestep"])
            t1_ind = int((24*60-cfg_set["t0"].hour*60-cfg_set["t0"].minute)/cfg_set["timestep"])
        else:
            t2_ind = int((cfg_set["t0"].hour*60+cfg_set["t0"].minute+cfg_set["n_integ"]*cfg_set["timestep"])/cfg_set["timestep"])
            t1_ind = int((cfg_set["t0"].hour*60+cfg_set["t0"].minute)/cfg_set["timestep"])
        t_ind = range(t1_ind,t2_ind)
    
        ## Get subset of precalculated UV displacement array:
        UVdisparr = load_file(path_name)
        Vx = UVdisparr["Vx"][t_ind,:,:]; Vy = UVdisparr["Vy"][t_ind,:,:]
        Dx = UVdisparr["Dx"][t_ind,:,:]; Dy = UVdisparr["Dy"][t_ind,:,:]
    else:
        Dx = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
        Dy = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
        Vx = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
        Vy = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
        path_name_file_t0 = path_creator_UV_disparr("standard",cfg_set,path=cfg_set["UV_precalc_path"],
                                                    t0=t_integ_ls[0].strftime("%Y%m%d"))
        UVdisparr = load_file(path_name_file_t0)
        print("   Read from precalculated UV-disparray...\n      %s" % path_name_file_t0)
        t_integ_0_day = t_integ_ls[0].day
        DV_index = 0        

        ## Go through all time steps to read in respective UV disparr field
        for t_integ in t_integ_ls:
            ## In case of non-reverse case, subtract five minutes (as 00:05 is first (!) element in precalc array)
            if not cfg_set["future_disp_reverse"]:
                t_integ_corr = t_integ-datetime.timedelta(minutes=cfg_set["timestep"])
                #t_integ_corr = t_integ-cfg_set["time_change_factor"]*datetime.timedelta(minutes=cfg_set["timestep"])
            else:
                t_integ_corr = t_integ
            
            ## If day changes, read in new daily array
            if t_integ_corr.day!=t_integ_0_day:
                path_name_file_tinteg = path_creator_UV_disparr("standard",cfg_set,path=cfg_set["UV_precalc_path"],
                                                                t0=t_integ_corr.strftime("%Y%m%d"))
                UVdisparr = load_file(path_name_file_tinteg)
                print("   Read from precalculated UV-disparray...\n      %s" % path_name_file_tinteg)
                t_integ_0_day = t_integ_corr.day
                
            ## Calculate index to be read:
            if not cfg_set["future_disp_reverse"]:
                UVdisparr_index = int((24*60-t_integ_corr.hour*60-t_integ_corr.minute)/cfg_set["timestep"])-1
            else:
                UVdisparr_index = int((t_integ_corr.hour*60+t_integ_corr.minute)/cfg_set["timestep"])
            
            ## Read in the data:
            Vx[DV_index,:,:] = UVdisparr["Vx"][UVdisparr_index,:,:]; Vy[DV_index,:,:] = UVdisparr["Vy"][UVdisparr_index,:,:]
            Dx[DV_index,:,:] = UVdisparr["Dx"][UVdisparr_index,:,:]; Dy[DV_index,:,:] = UVdisparr["Dy"][UVdisparr_index,:,:] 
            DV_index += 1            
    
    filename = path_creator_UV_disparr("standard",cfg_set)
    #filename = "%stmp/%s_%s_disparr_UV%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
    #                                            cfg_set["oflow_source"],cfg_set["file_ext_verif"],
    #                                            cfg_set["save_type"])
    print("   ... saving UV-disparray subset in\n      %s" % filename)
    save_file(filename, data_arr=[Dx,Dy,Vx,Vy],var_name=["Dx","Dy","Vx","Vy"],
              cfg_set=cfg_set)

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
    path_str          = path if path is not None else cfg_set["root_path"]+"tmp/"
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
    path_str          = path if path is not None else cfg_set["root_path"]+"tmp/"
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
    config.read(u"%s/%s" % (cfg_set["CONFIG_PATH"],cfg_set["CONFIG_FILE_set"]))
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
        raise NotImplementedError("So far path_creator implemented for radar products (RZC, BZC...), THX, and COSMO Wind")
    
## Read input to calculate optical flow displacement array for specific time-step
def calc_disparr(t_current, cfg_set, resid=False):
    """Get 2-dim displacement array for flow between timesteps t_current and t_current - n_past_frames*timestep.
   
    Parameters
    ----------
    
    t_current : datetime object
        Current time for which to calculate displacement array.
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
        
    UV_inter : bool
        If UV_inter is true, the calculated UV and sparsened UV vectors are returned as well.
    
    See function check_create_disparray(t0, timestep, n_integ, root_path, t0_str, oflow_source, oflow_source_path)
    """
    
    if resid:
        ## Read in current displaced oflow_source file:
        filename = path_creator_vararr("disp",cfg_set["oflow_source"],cfg_set)
        #filename = "%stmp/%s_%s_disp%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                       cfg_set["oflow_source"], cfg_set["file_ext_verif"], cfg_set["save_type"])
        t_diff = cfg_set["t0"] - t_current
        t_diff_ind = int((t_diff.seconds/60)/cfg_set["timestep"])
        #oflow_source_data = np.load(filename)[t_diff_ind:t_diff_ind+cfg_set["n_past_frames"]+1,:,:]
        #oflow_source_data = np.load(filename)[t_diff_ind+cfg_set["n_past_frames_resid"]::-1,:,:][:cfg_set["n_past_frames"]+1]
        oflow_source_data = load_file(filename,cfg_set["oflow_source"])[t_diff_ind+cfg_set["n_past_frames_resid"]::-1,:,:][:cfg_set["n_past_frames"]+1]
        if oflow_source_data.shape[0]==1:
            UV = R = np.zeros((2,oflow_source_data.shape[1],oflow_source_data.shape[2]))
            if not cfg_set["UV_inter"]: return UV, R
            else: return UV, R, np.zeros(4)*np.nan, np.zeros(4)*np.nan
        if np.all(np.array_equal(oflow_source_data[0,:,:],oflow_source_data[1,:,:])):
            print("Input data equal")
            sys.exit()
    else:
        ## Read in current oflow_source file:
        filenames, timestamps = path_creator(t_current, cfg_set["oflow_source"], cfg_set["source_dict"][cfg_set["oflow_source"]], cfg_set)
        ret = metranet.read_file(filenames[0], physic_value=True)
        oflow_source_data = np.atleast_3d(ret.data)
        for filename in filenames[1:]:
            ret_d_t = metranet.read_file(filename, physic_value=True)
            oflow_source_data_d_t = np.atleast_3d(ret_d_t.data)
            oflow_source_data = np.append(oflow_source_data,oflow_source_data_d_t, axis=2)
        
        oflow_source_data = np.moveaxis(oflow_source_data,2,0)
        #oflow_source_data_masked = np.ma.masked_invalid(oflow_source_data)
        #oflow_source_data_masked = np.ma.masked_where(oflow_source_data_masked==0,oflow_source_data_masked)    
        

    ## Check whether there are non-nan entries:
    if np.any(np.isnan(oflow_source_data).all(axis=(1,2))):
        print("   *** Warning: Input oflow source field is all NAN!\n                Returning NAN fields.***")
        nan_arr = oflow_source_data[0,:,:]*np.nan
        D  = np.array([nan_arr,nan_arr])
        UV = np.array([nan_arr,nan_arr])
        UV_vec  = []; UV_vec_sp = []
        if not cfg_set["UV_inter"]:
            return D, UV
        else: return D, UV, UV_vec, UV_vec_sp
        
    ## Convert linear rainrates to logarithimc dBR units
    if not cfg_set["oflow_source"]=="RZC":
        raise NotImplementedError("So far displacement array retrieval only implemented for RZC")
    else:
        ## Get threshold method:
        if not resid:
            R_thresh_meth = cfg_set["R_thresh_meth"]
            R_threshold = cfg_set["R_threshold"]
        else:
            R_thresh_meth = cfg_set["R_thresh_meth_resid"]
            R_threshold = cfg_set["R_threshold_resid"]
        
        ## Get threshold value:
        if R_thresh_meth == "fix":
            R_thresh = R_threshold
        elif R_thresh_meth == "perc":
            R_thresh = np.min([np.nanpercentile(oflow_source_data[0,:,:],R_threshold),
                              np.nanpercentile(oflow_source_data[1,:,:],R_threshold)])
        else: raise ValueError("R_thresh_meth must either be set to 'fix' or 'perc'")
                
        ## Convert to dBR
        dBR, dBRmin = st.utils.mmhr2dBR(oflow_source_data, R_thresh)
        dBR[~np.isfinite(dBR)] = dBRmin
        #R_thresh = cfg_set["R_threshold"]
        
        ## In case threshold is not exceeded, lower R_threshold by 20%
        while (dBR==dBRmin).all():
            if cfg_set["verbose"]: print("   *** Warning: Threshold not exceeded, "+
                                         "lower R_threshold by 20% to "+str(R_thresh*0.8)+" ***")
            R_thresh = R_thresh*0.8
            dBR, dBRmin = st.utils.mmhr2dBR(oflow_source_data, R_thresh)
            dBR[~np.isfinite(dBR)] = dBRmin
        
        ## For correction of residuals original mm/h values are used:
        if resid:
            oflow_source_data_min = oflow_source_data
            oflow_source_data_min[oflow_source_data_min<=R_thresh] = R_thresh
            oflow_source_data_min[~np.isfinite(oflow_source_data_min)] = R_thresh
  
    ## Calculate UV field
    oflow_method = st.optflow.get_method(cfg_set["oflow_method_name"])
    if not resid:
        UV, UV_vec, UV_vec_sp = oflow_method(dBR,return_single_vec=True,return_declust_vec=True)
    else:
        UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,winsize_LK5=(120,20),quality_level_ST=0.05,
                                             max_speed=20,nr_IQR_outlier=5,k=30,
                                             decl_grid=cfg_set["decl_grid_resid"],function=cfg_set["inter_fun_resid"],
                                             epsilon=cfg_set["epsilon_resid"],#factor_median=.2,
                                             return_single_vec=True,return_declust_vec=True,
                                             zero_interpol=cfg_set["zero_interpol"])
        #UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,block_size_ST=15,winsize_LK5=(120,20),quality_level_ST=0.05,
        #                                                  max_speed=20,nr_IQR_outlier=5,decl_grid=20,function="inverse",k=20,factor_median=0.05,
        #                                                  return_single_vec=True,return_declust_vec=True,zero_interpol=True)
        #UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,block_size_ST=15,winsize_LK5=(120,20),quality_level_ST=0.05,
        #                                                  max_speed=20,nr_IQR_outlier=5,decl_grid=20,function="nearest",k=20,factor_median=.2,
        #                                                  return_single_vec=True,return_declust_vec=True)
       
    ## In case no motion vectors were detected, lower R_threshold by 20%
    if np.any(~np.isfinite(UV)):
        dBR_orig = dBR
    n_rep = 0
    while np.any(~np.isfinite(UV)):
        if cfg_set["verbose"]:
            print("   *** Warning: No motion vectors detected, lower R_threshold by 30% to "+str(R_thresh*0.7)+" ***")
        R_thresh = R_thresh*0.7
        dBR, dBRmin = st.utils.mmhr2dBR(oflow_source_data, R_thresh)
        dBR[~np.isfinite(dBR)] = dBRmin
        
        if resid:
            oflow_source_data_min = oflow_source_data
            oflow_source_data[oflow_source_data<=R_thresh] = R_thresh
            oflow_source_data[~np.isfinite(oflow_source_data)] = R_thresh
        if not resid:
            UV, UV_vec, UV_vec_sp = oflow_method(dBR,return_single_vec=True,return_declust_vec=True)
        else:
            UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,winsize_LK5=(120,20),quality_level_ST=0.05,
                                                 max_speed=20,nr_IQR_outlier=5,k=30,
                                                 decl_grid=cfg_set["decl_grid_resid"],function=cfg_set["inter_fun_resid"],
                                                 epsilon=cfg_set["epsilon_resid"],#factor_median=.2,
                                                 return_single_vec=True,return_declust_vec=True,
                                                 zero_interpol=cfg_set["zero_interpol"])
            #UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,block_size_ST=15,winsize_LK5=(120,20),quality_level_ST=0.05,
            #                                                  max_speed=20,nr_IQR_outlier=5,decl_grid=20,function="inverse",k=20,epsilon=10,#factor_median=0.05,
            #                                                  return_single_vec=True,return_declust_vec=True,zero_interpol=True)
            #UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,block_size_ST=15,winsize_LK5=(120,20),quality_level_ST=0.05,
            #                                                  max_speed=20,nr_IQR_outlier=5,decl_grid=20,function="nearest",k=20,factor_median=.2,
            #                                                  return_single_vec=True,return_declust_vec=True)
        n_rep += 1
        if n_rep > 2:
            UV = np.zeros((2,dBR.shape[1],dBR.shape[2]))
            if cfg_set["verbose"]: print("   *** Warning: Return zero UV-array! ***")
            break
    
    ## Invert direction of intermediate motion vectors
    #if cfg_set["UV_inter"]:
    #    UV_vec[2:3,:,:] = -UV_vec[2:3,:,:]
    #    UV_vec_sp[2:3,:,:] = -UV_vec_sp[2:3,:,:]
    
    """
    ## Advect disp_test to get the advected test_array and the displacement array
    adv_method = st.advection.get_method(cfg_set["adv_method"])
    dBR_adv, D = adv_method(dBR[-1,:,:], UV, 1, return_displacement=True) 
    
    ## convert the forecasted dBR to mmhr
    if cfg_set["oflow_source"]=="RZC":
        if cfg_set["R_thresh_meth"] == "fix":
            R_tresh = cfg_set["R_threshold"]
        elif cfg_set["R_thresh_meth"] == "perc":
            R_tresh = np.min([np.nanpercentile(oflow_source_data[0,:,:],cfg_set["R_threshold"]),
                              np.nanpercentile(oflow_source_data[1,:,:],cfg_set["R_threshold"])])
        else: raise NotImplementedError("R_thresh_meth must either be set to 'fix' or 'perc'")
        oflow_source_data_forecast = st.utils.dBR2mmhr(dBR_adv, R_tresh)
    
    ## Print results:
    if False:
        calc_disparr_ctrl_plot(D,timestamps,oflow_source_data,oflow_source_data_forecast,cfg_set)
    """
    #plt.imshow(D[0,:,:])
    #plt.show()
    #fig, axes = plt.subplots(nrows=1, ncols=2)
    #fig1=axes[0].imshow(UV[0,:,:])
    #fig.colorbar(fig1,ax=axes[0])#,orientation='horizontal')
    #fig2=axes[1].imshow(UV[1,:,:])
    #fig.colorbar(fig2,ax=axes[1])#,orientation='horizontal')
    #fig.tight_layout()
    #plt.show()
    #sys.exit()
    

    if np.all(UV==0): #np.any(~np.isfinite(UV)):
        if cfg_set["instant_resid_corr"] and not resid:
            print("   *** Warning: No residual movement correction performed ***")
        D = UV.copy()
    else:
        adv_method = st.advection.get_method(cfg_set["adv_method"])
        dBR_disp, D = adv_method(dBR[-2,:,:],UV,1,return_displacement=True,return_XYW=False)
        
        if cfg_set["instant_resid_corr"] and not resid:    
            if cfg_set["verbose"]:
                print("   Make instantaneous residual movement correction")
            ## Advect second last observation to t0:
            dBR_disp[~np.isfinite(dBR_disp)] = dBRmin
            
            ## Convert dBR values of t0 and second last time step to mm/h:
            RZC_resid_fields = np.stack([st.utils.dBR2mmhr(dBR_disp[0,:,:], R_thresh),
                                         st.utils.dBR2mmhr(dBR[-1,:,:], R_thresh)])
            #plt.imshow(RZC_resid_fields[0,:,:]); plt.title("RZC_resid_fields[0,:,:]"); plt.show()
            #plt.imshow(RZC_resid_fields[1,:,:]); plt.title("RZC_resid_fields[1,:,:]"); plt.show()

            ## Get residual displacement field
            UV_resid = oflow_method(RZC_resid_fields,min_distance_ST=2,
                                    winsize_LK5=(120,20),quality_level_ST=0.05,
                                    max_speed=20,nr_IQR_outlier=5,k=30,
                                    decl_grid=cfg_set["decl_grid_resid"],function=cfg_set["inter_fun_resid"],
                                    epsilon=cfg_set["epsilon_resid"],#factor_median=.2,
                                    zero_interpol=cfg_set["zero_interpol"])
            
            ## Add UV_resid to original UV array
            n_rep = 0
            while np.any(~np.isfinite(UV_resid)):
                print("       No UV_resid field found")
                R_thresh *= 0.7
                RZC_resid_fields = np.stack([st.utils.dBR2mmhr(dBR_disp[0,:,:], R_thresh),
                                             st.utils.dBR2mmhr(dBR[-1,:,:], R_thresh)])
                
                UV_resid = oflow_method(RZC_resid_fields,min_distance_ST=2,
                                        winsize_LK5=(120,20),quality_level_ST=0.05,
                                        max_speed=20,nr_IQR_outlier=5,k=30,
                                        decl_grid=cfg_set["decl_grid_resid"],function=cfg_set["inter_fun_resid"],
                                        epsilon=cfg_set["epsilon_resid"],#factor_median=.2,
                                        zero_interpol=cfg_set["zero_interpol"])
                n_rep += 1
                if n_rep > 2:
                    UV_resid = np.zeros((2,dBR.shape[1],dBR.shape[2]))
                    #if cfg_set["verbose"]: 
                    print("   *** Warning: Return zero UV_resid array! ***")
                    break
            UV += UV_resid
            
            ## Displace with UV_resid field to get D_resid and add to D array:
            dBR_disp_disp, D = adv_method(RZC_resid_fields[0,:,:],UV,1,
                                          return_displacement=True,return_XYW=False)  
            #D += D_resid
    
    if not cfg_set["UV_inter"]:
        return D, UV
    else: return D, UV, UV_vec, UV_vec_sp
   
## In case displacement array does not exist yet, it is created with this function:
def create_new_disparray(cfg_set,extra_verbose=False,resid=False,precalc=False):
    """Create Displacement array.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
        
    precalc : bool
        Create precalculated displacement array?
        Default: False.
    """
    
    ## Calculate n_integ time deltas:
    resid_print = " for residual movement correction" if resid else ""
    print("Create new displacement array%s..." % resid_print)
    Dx = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
    Dy = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
    Vx = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
    Vy = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
    if cfg_set["UV_inter"]: UV_vec = []; UV_vec_sp = []

    ## Calculate flow fields and displacement fields for each integration timestep n_integ:
    i = 0   
    t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=cfg_set["timestep"])
    for t_d in t_delta:
        t_current = cfg_set["t0"] - cfg_set["time_change_factor"]*t_d
        if len(t_delta) > 10 and t_current.minute == 0: print("  ... for %s" % t_current.strftime("%Y-%m-%d %H:%M"))
        if cfg_set["UV_inter"]:
            D, UV, UV_vec_temp, UV_vec_sp_temp = calc_disparr(t_current, cfg_set, resid)
            UV_vec.append(UV_vec_temp) #if UV_vec_temp is not None: 
            UV_vec_sp.append(UV_vec_sp_temp) #if UV_vec_sp_temp is not None: UV_vec_sp.append(UV_vec_sp_temp)
        else:
            D, UV = calc_disparr(t_current, cfg_set, resid)
            
        Vx[i,:,:] = UV[0,:,:]; Vy[i,:,:] = UV[1,:,:]
        Dx[i,:,:] =  D[0,:,:]; Dy[i,:,:] =  D[1,:,:]
        if extra_verbose: print("   Displacement calculated between "+str(t_current-datetime.timedelta(minutes=cfg_set["timestep"]))+
                          " and "+str(t_current))
        i += 1
        if False:
            figname = "%s%s_%s_disparr_UV.png" % (cfg_set["output_path"], t_current.strftime("%Y%m%d%H%M"),
                                                  cfg_set["oflow_source"])
            plt.subplot(1, 2, 1)
            st.plt.quiver(D,None,25)
            plt.title('Displacement field\n%s' % t_current.strftime("%Y-%m-%d %H:%M"))
            plt.subplot(1, 2, 2,)
            st.plt.quiver(UV,None,25)
            plt.title('UV field\n%s' % t_current.strftime("%Y-%m-%d %H:%M"))
            plt.savefig(figname)

    ## Give different filename if the motion field is precalculated for training dataset:
    type = "resid" if resid else "standard"
    if precalc:
        if not cfg_set["future_disp_reverse"]:
            t0_str = (cfg_set["t0"]-datetime.timedelta(days=1)).strftime("%Y%m%d")
        else: 
            t0_str = cfg_set["t0"].strftime("%Y%m%d")
        filename = path_creator_UV_disparr(type,cfg_set,
                                           path=cfg_set["UV_precalc_path"],
                                           t0=t0_str)
        #filename1 = "%s/%s_%s_disparr_UV%s%s.%s" % (cfg_set["UV_precalc_path"], (cfg_set["t0"]-datetime.timedelta(days=1)).strftime("%Y%m%d"),
        #                                           cfg_set["oflow_source"],append_str_resid,cfg_set["file_ext_verif"],
        #                                           cfg_set["save_type"])
    else:
        filename = path_creator_UV_disparr(type,cfg_set)
        #filename = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                              cfg_set["oflow_source"],append_str_resid,cfg_set["file_ext_verif"],
        #                                              cfg_set["save_type"])
                                                  
    #np.savez(filename, Dx=Dx, Dy=Dy, Vx=Vx, Vy=Vy)
    save_file(filename, data_arr=[Dx,Dy,Vx,Vy],var_name=["Dx","Dy","Vx","Vy"],
              cfg_set=cfg_set)
    print("  ... new displacement array saved in:\n       %s" % filename)
        
    ## Save combined displacement array (initial displacment + residual displacment):
    if cfg_set["resid_disp_onestep"] and resid:
        ## Load initial displacement field:
        filename_ini = path_creator_UV_disparr("standard",cfg_set)
        #filename_ini = "%stmp/%s_%s_disparr_UV%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                    cfg_set["oflow_source"],cfg_set["file_ext_verif"],
        #                                                    cfg_set["save_type"])
        UVdisparr_ini = load_file(filename_ini)       
        
        ## Save summation of initial and residual displacment field
        filename_combi = path_creator_UV_disparr("resid_combi",cfg_set)
        #filename_combi = "%stmp/%s_%s_disparr_UV_resid_combi%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                              cfg_set["oflow_source"],cfg_set["file_ext_verif"],
        #                                                              cfg_set["save_type"])
        #np.savez(filename_combi, Dx=Dx+UVdisparr_ini["Dx"], Dy=Dy+UVdisparr_ini["Dy"],
        #                         Vx=Vx+UVdisparr_ini["Vx"], Vy=Vy+UVdisparr_ini["Vy"])
        save_file(filename_combi, data_arr=[Dx+UVdisparr_ini["Dx"][:,:,:],Dy+UVdisparr_ini["Dy"][:,:,:],
                                            Vx+UVdisparr_ini["Vx"][:,:,:],Vy+UVdisparr_ini["Vy"][:,:,:]],
                  var_name=["Dx","Dy","Vx","Vy"],cfg_set=cfg_set)
        print("  ... combined displacement array saved in:\n       %s" % filename_combi)
        
    ## Also save intermediate UV motion vectors:
    if cfg_set["UV_inter"]:
        type_vec = "vec_resid" if resid else "vec"
        filename = path_creator_UV_disparr(type_vec,cfg_set,save_type="npz")
        #filename = "%stmp/%s_%s_disparr_UV_vec%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                   cfg_set["oflow_source"],append_str_resid,cfg_set["file_ext_verif"],
        #                                                   cfg_set["save_type"])
                                                           
        
        ## Give different filename if the motion field is precalculated for training dataset:
        #if cfg_set["precalc_UV_disparr"]:
        #    filename = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], (cfg_set["t0"]-datetime.timedelta(days=1)).strftime("%Y%m%d"),
        #                                                  cfg_set["oflow_source"],append_str_resid,cfg_set["file_ext_verif"],
        #                                                  cfg_set["save_type"])
        
        ## Set last value to np.inf to otbain right numpy array of numpy arrays:
        UV_vec[-1]=np.inf
        UV_vec_sp[-1]=np.inf
        
        ## Save intermediate UV motion vectors:
        #np.savez(filename, UV_vec=UV_vec, UV_vec_sp=UV_vec_sp)
        save_file(filename, data_arr=[UV_vec,UV_vec_sp],
                  var_name=["UV_vec","UV_vec_sp"],cfg_set=cfg_set,filetype="npy")
        print("  ... new UV vector lists saved in:\n      %s" % filename)

    
## Calculate optical flow displacement array for specific time-step
def calc_disparr_ctrl_plot(D,timestamps,oflow_source_data,oflow_source_data_forecast,cfg_set):
    """Plot and save two frames as input of LucasKanade and the advected field, as well as plot the displacement field.
   
    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
    
    timestamps : datetime object
        Timestamps of optical flow input fields.
    
    D : array-like
        Displacement field.
    
    oflow_source_data : array-like
        Optical flow input fields.
        
    oflow_source_data_forecast : array-like
        Advected optical flow input field.
    """
    
    plt.subplot(1, 2, 1)
    plt.imshow(D[0,:,:])
    plt.title('X component of displacement field')

    plt.subplot(1, 2, 2)
    plt.imshow(D[1,:,:])
    plt.title('Y component of displacement field')
    #plt.show()

    # Plot two frames for UV-derivation and advected frame
    for i in range(oflow_source_data.shape[0] + 1):
        plt.clf()
        if True:
            if i < oflow_source_data.shape[0]:
                # Plot last observed rainfields
                st.plt.plot_precip_field(oflow_source_data[i,:,:], None, units="mmhr",
                              colorscale=cfg_set["colorscale"], 
                              title=timestamps[i].strftime("%Y-%m-%d %H:%M"), 
                              colorbar=True)
                
                figname = "%s%s_%s_simple_advection_%02d_obs.png" % (cfg_set["output_path"], timestamps[i].strftime("%Y%m%d%H%M"),
                                                                      cfg_set["oflow_source"], i)
                plt.savefig(figname)
                print(figname, 'saved.')
            else:
                # Plot nowcast
                st.plt.plot_precip_field(oflow_source_data_forecast[i - oflow_source_data.shape[0],:,:], 
                              None, units="mmhr", 
                              title="%s +%02d min" % 
                              (timestamps[-1].strftime("%Y-%m-%d %H:%M"),
                              (1 + i - oflow_source_data.shape[0])*cfg_set["timestep"]),
                              colorscale=cfg_set["colorscale"], colorbar=True)
                figname = "%s%s_%s_simple_advection_%02d_nwc.png" % (cfg_set["output_path"], timestamps[-1].strftime("%Y%m%d%H%M"),
                                                                      cfg_set["oflow_source"], i)
                plt.savefig(figname)
                print(figname, "saved.")

## Read COSMO NetCDF wind file and crop to ccs4:
def read_wind_nc(filename):
    """Read COSMO NetCDF file and crop to ccs4.

    Parameters
    ----------
    
    filenames : str
        Filepath to file to be imported.
    """
    
    ncfile = Dataset(filename,'r')
    
    ## Read wind variables from COSMO
    U_cosmo = ncfile.variables["U"][0,:,:]
    V_cosmo = ncfile.variables["V"][0,:,:]
    UV_cosmo = np.array([U_cosmo,V_cosmo])
        
    ## Crop from swissXXL to ccs4 format (see Uli's mail):
    crop_x1 = 40; crop_y1 = 155
    UV_cosmo = UV_cosmo[:,:,crop_x1:crop_x1+640,crop_y1:crop_y1+710]
        
    UV_cosmo[UV_cosmo == -99999.0] = np.nan
    #UV_cosmo_m = np.ma.masked_array(UV_cosmo) # convert numpy array to masked array
    #UV_cosmo_m.mask = (UV_cosmo == -99999.0)
    UV_cosmo = UV_cosmo[:,:,::-1,:]
    
    return UV_cosmo
    
## Read COSMO NetCDF convection file:
def read_convection_nc(filename,var,cfg_set):
    """Read COSMO NetCDF convection file.

    Parameters
    ----------
    
    filenames : str
        Filepath to file to be imported.
        
    var : string
        Name of variable to be returned
    """
    
    ncfile = Dataset(filename,'r')
    #conv_vararr = np.zeros((1,)+cfg_set["xy_ext"])
    if var in ["lat_1","lon_1"]:
        conv_vararr = np.array(ncfile.variables[var][:,:])
        conv_vararr = conv_vararr[::-1,:]
        conv_vararr = np.moveaxis(np.atleast_3d(conv_vararr),2,0)
    elif "POT_VORTIC" in var or "THETAE" in var or "MCONV" in var or \
         "geopotential_height" in var or "RELHUM" in var:
        var_name = var[:-6]; pressure = var[-5:]
        if pressure == "30000" or pressure == "75000":
            pressure_ind = 0
        elif pressure == "50000" or pressure == "85000":
            pressure_ind = 1
        elif pressure == "70000":
            pressure_ind = 2
        conv_vararr = np.array(ncfile.variables[var_name][:,pressure_ind,:,:])
        conv_vararr = conv_vararr[:,::-1,:]
        if "POT_VORTIC" in var or "MCONV" in var: conv_vararr*=10e10
    else:
        conv_vararr  = np.array(ncfile.variables[var][:,:,:])
        conv_vararr  = conv_vararr[:,::-1,:]
    conv_vararr[conv_vararr == -99999.0] = np.nan    
    return conv_vararr


## Read COSMO NetCDF convection file:
def read_convection_nc_old(filename,cfg_set):
    """Read COSMO NetCDF convection file.

    Parameters
    ----------
    
    filenames : str
        Filepath to file to be imported.
    """
    
    ncfile = Dataset(filename,'r')
    
    ## Read convection variables from COSMO
    config = configparser.RawConfigParser()
    config.read("%s/%s" % (cfg_set["CONFIG_PATH"],cfg_set["CONFIG_FILE_set"]))
    config_ds = config["variables"]
    conv_vars = config_ds["conv_vars"].split(',')
    
    conv_vararr = np.zeros((len(conv_vars),)+cfg_set["xy_ext"])
    i = 0
    for conv_var in conv_vars:
        conv_vararr[i,:,:] = ncfile.variables[conv_var][0,:,:]
        i += 1

    conv_vararr[conv_vararr == -99999.0] = np.nan
    #conv_vararr = np.ma.masked_array(conv_vararr) # convert numpy array to masked array
    #conv_vararr.mask = (conv_vararr == -99999.0)
    
    return conv_vararr

## Read MSG3 satellite data file:
def read_sat_nc(filename,cfg_set,var):
    """Read COSMO NetCDF convection file.

    Parameters
    ----------
    
    filenames : str
        Filepath to file to be imported.
    """
    
    corrupt_files = ["/data/COALITION2/database/meteosat/ccs4_PLAX/2018/05/30/MSG3_ccs4_201805302050_rad_PLAX.nc",
                     "/data/COALITION2/database/meteosat/ccs4_PLAX/2018/05/30/MSG3_ccs4_201805301830_rad_PLAX.nc",
                     "/data/COALITION2/database/meteosat/ccs4_PLAX/2018/05/30/MSG3_ccs4_201805301545_rad_PLAX.nc"]
    
    if filename in corrupt_files:
        print("   *** Dataset corrupted: %s ***" % str(filename))
        vararr = np.zeros((cfg_set["xy_ext"][0],cfg_set["xy_ext"][1]))*np.nan
        return vararr
    else: ncfile = Dataset(filename,'r')
    
    ## Read satellite SEVIRI channel
    sat_vararr = ncfile.variables[var][:,:,0]
    sat_vararr[sat_vararr == 999.0] = np.nan
    
    ncfile.close()
    return sat_vararr

## Read lightning data file:    
def read_lightning_data(var,filename,cfg_set,t_current):
    """Read THX lightning data file (Ascii .prd or NetCDF file).

    Parameters
    ----------
    
    var : str
        THX variable to be read from file.
    
    filenames : str
        Filepath to file to be imported.
    """
    from netCDF4 import Dataset
    config = configparser.RawConfigParser()
    config.read("%s/%s" % (cfg_set["CONFIG_PATH"],cfg_set["CONFIG_FILE_set"]))
    config_ds   = config["light_read"]
    
    ## Make sure only one lightning dataset is provided:
    if len(filename)>1:
        print("*** Several filenames provided, can only work with one: ***")
        print(filename)
        print("*** Break process ***"); sys.exit()
    filename = filename[0]
    
    #print("      Read lightning data from .%s files" % config_ds["fn_ext"])
    
    ## Case if text files should be read:
    if config_ds["fn_ext"]=="prd":
        ## Assure that algorithm works also at 00:00UTC:
        if t_current.hour==0 and t_current.minute==0:
            t_current = t_current-datetime.timedelta(seconds=0.5)
        ## Read in the data:
        vararr = swisslightning_jmz.readLightning(filename,False,t_current,cfg_set["timestep"])
        ## Get correct return:
        if var[4:] == "dens": var_ind = 1
        if var[4:] == "densIC": var_ind = 2
        if var[4:] == "densCG": var_ind = 3
        if var[4:] == "curr_abs": var_ind = 4
        if var[4:] == "curr_neg": var_ind = 5
        if var[4:] == "curr_pos": var_ind = 6
        vararr = vararr[var_ind]
        
    ## Case if nc files should be read:
    elif config_ds["fn_ext"]=="nc":
        ncfile = Dataset(filename,'r')
        nc_time = ncfile.variables["time"]        
        if nc_time[0]!=cfg_set["timestep"]:
            print("*** Lightning data only available at %dmin time-steps ***" %
                  nc_time[0])
            sys.exit()
        if t_current.hour==0 and t_current.minute==0:
            vararr = ncfile.variables[var][-1,:,:]
        else:
            t_diff_min = int((t_current - t_current.replace(hour=00, minute=00)).seconds/60)
            t_ind = np.where(nc_time[:]==t_diff_min)[0]
            if len(t_ind)>1:
                print("ERROR: Unambigous time information for lightning data")
                sys.exit()
            elif len(t_ind)<1:
                print("ERROR: No matching lighting data found")
                sys.exit()
            else: t_ind = t_ind[0]
            vararr = ncfile.variables[var][t_ind,:,:]
        # Get it into the right shape:
        #vararr = np.moveaxis(vararr,1,0)
    
    ## Unfold lightning data if necessary:
    if config_ds["unfold"]=="True":
        unfold_form = config_ds["unfold_form"]
        dx_unfold   = int(config_ds["dx_unfold"])
        vararr      = swisslightning_jmz.unfold_lightning(vararr,dx_unfold,unfold_form)
    return vararr
                
## Get array of a certain variable in a certain time step:
def get_vararr_t(t_current, var, cfg_set):
    """Displace fields with current displacement arrays.

    Parameters
    ----------
    
    t_current : datetime object
        Current time for which to calculate displacement array.
    
    var : string
        Name of variable to be returned
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
    """ 
    
    source = cfg_set["source_dict"][var]
    ## Implement different reader for different variable:
    if source == "RADAR":
        filenames, timestamps = path_creator(t_current, var, source, cfg_set)
        index_timestep = np.where([timestamp==t_current for timestamp in timestamps])[0][0]
        vararr = metranet.read_file(filenames[index_timestep], physic_value=True)
        #print(t_current,np.nanmax(vararr.data))
        vararr = np.moveaxis(np.atleast_3d(vararr.data),2,0)
        return vararr
    elif source == "THX":
        filenames, timestamps = path_creator(t_current, var, source, cfg_set)
        vararr = read_lightning_data(var, filenames, cfg_set, t_current)
        vararr = np.moveaxis(np.atleast_3d(vararr),2,0)#np.moveaxis(,2,1)
        return vararr
    elif source == "COSMO_WIND":
        filename, timestamps = path_creator(t_current, var, source, cfg_set)
        vararr = read_wind_nc(filename)
        plt.imshow(vararr[0,:,:,:])
        plt.show()
        sys.exit()
        
        return vararr
    elif source == "COSMO_CONV":
        if t_current.minute==0:
            filename, timestamps = path_creator(t_current, var, source, cfg_set)
            vararr = read_convection_nc(filename,var,cfg_set)    
        else:
            filename_h_old, timestamp_h_old = path_creator(t_current, var, source, cfg_set)
            vararr_old = read_convection_nc(filename_h_old,var,cfg_set)
            weight_old = 1-t_current.minute/60.
            
            t_current_plus1h = t_current + datetime.timedelta(hours=1)
            filename_h_new, timestamp_h_new = path_creator(t_current_plus1h, var, source, cfg_set)
            vararr_new = read_convection_nc(filename_h_new,var,cfg_set)
            weight_new = 1-weight_old
            
            vararr = weight_old*vararr_old+weight_new*vararr_new    

        ## Smooth fields if requested:            
        #if cfg_set["smooth_conv"]:
        #    vararr[0,:,:] = ndimage.gaussian_filter(vararr[0,:,:],cfg_set["smooth_sig"])
        #if cfg_set["smooth_conv"]: vararr = smooth_conv_vararr(vararr,cfg_set["smooth_sig"])
        return vararr
    #elif var=="Conv":
    #    filename, timestamps = path_creator(t_current, var, cfg_set)
    #    vararr = read_convection_nc(filename,cfg_set)
    #    vararr = vararr[:,::-1,:]
    #    return vararr
    elif source == "SEVIRI":
        filenames, timestamps = path_creator(t_current, var, source, cfg_set)
        if all(filename is None for filename in filenames):
            vararr = np.zeros((1,cfg_set["xy_ext"][0],cfg_set["xy_ext"][1]))*np.nan
        else:
            vararr = read_sat_nc(filenames[0],cfg_set,var)
            vararr = np.moveaxis(np.atleast_3d(vararr),2,0)#np.moveaxis(,2,1)
        return vararr
    else:
        raise NotImplementedError("So far path_creator implemented for radar products RZC, BZC... and THX")

## Perform smoothing of convective variables: 
def smooth_conv_vararr(vararr,sigma):
    """Perform smoothing of convective variables.

    Parameters
    ----------
    
    vararr : numpy array
        2D numpy array with convective data which should be smoothed.
        
    sigma : float
        Kernel width for smoothing.
    """
    vararr[0,:,:] = ndimage.gaussian_filter(vararr[0,:,:],sigma)
    return(vararr)
    
## Create empty dataframe with specific columns and datatypes:
def df_empty(columns, dtypes, index=None):
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df
    
## Produce array with TRT cell centre pixels 
def get_vararr_TRT_t0(t0, cfg_set):
    """Provide vararr style array filled with centre locations of TRT cells"""
    
    ## Read filepath of respective TRT file:
    filepaths, timestamps = path_creator(t0, "TRT", "TRT", cfg_set)
    cell_info_df = df_empty(cfg_set["TRT_cols"],cfg_set["TRT_dtype"])   
    filename = "%stmp/%s%s" % (cfg_set["root_path"],
                               cfg_set["t0"].strftime("%Y%m%d%H%M"),
                               "_TRT_df.pkl")
    
    ## Exception if no TRT-file is available:
    if filepaths[0] is None:
        print("   *** Warning: No TRT file found for %s ***" % t0)
        cell_info_df.to_pickle(filename)
        vararr = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
        return vararr
        
    ## Read in TRT files, get location (CHi,CHj) and TRT variables:
    traj_IDs, TRTcells, cell_mask = swisstrt.readRdt(filepaths[0])
    vararr = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"],dtype=np.int32)
    traj_ID_list = []
    for traj_ID in traj_IDs:
        dict_cellinfo = {key:value for key, value in TRTcells[traj_ID].__dict__.items() if not key.startswith('__') and not callable(key)}
        cell_info_df=cell_info_df.append(pd.DataFrame.from_records([dict_cellinfo],
                                         index=[9]), ignore_index=True, sort=True)
        vararr[:,int(TRTcells[traj_ID].iCH),
                 int(TRTcells[traj_ID].jCH)] = np.int32(traj_ID[8:])
        traj_ID_list.append(traj_ID)
    
    ## Change index to TRT_ID, set dtype for columns and save to disk:
    cell_info_df.index = traj_ID_list
    cell_info_df=cell_info_df.astype(cfg_set["type_dict_TRT"],errors='raise')
    cell_info_df.to_pickle(filename)
    return vararr

## Get indices based on i,j coordinate:
def get_indices_of_domain(i,j,cfg_set,X=None,Y=None):
    if cfg_set["stat_sel_form"] == "square":
        dx = int(cfg_set["stat_sel_form_width"]/2)
        bool_arr = np.full(cfg_set["xy_ext"], False, dtype=bool)
        bool_arr[:,:] = False
        bool_arr[i-dx:i+dx,j-dx:j+dx] = True
        indices = np.where(bool_arr.flat)
    elif cfg_set["stat_sel_form"] == "circle":
        interior = ((X-j)**2 + (Y-i)**2) < (cfg_set["stat_sel_form_width"]/2.)**2
        indices = np.where(interior.flat)
    return indices
    
## Function which reads the indices corresponding to the domain of interest around the displaced TRT cell centres:
def read_TRT_area_indices(cfg_set_input,reverse):
    """Function which reads the indices corresponding to the domain
    of interest around the displaced TRT cell centres"""

    ## Change settings related calculating statistics from future or current observations:
    cfg_set = cfg_set_input.copy()
    cfg_set["future_disp_reverse"] = True if reverse else False
    cfg_set["time_change_factor"]  = -1   if reverse else 1
    string_time = "future" if cfg_set["future_disp_reverse"] else "past"
    print("Read %s indices of the domains of interest..." % string_time)
    
    ## Read TRT info dataframe:
    filename = "%stmp/%s%s" % (cfg_set["root_path"],
                               cfg_set["t0"].strftime("%Y%m%d%H%M"),
                               "_TRT_df.pkl")
    cell_info_df = pd.read_pickle(filename)
    cell_info_df = cell_info_df.loc[cell_info_df["RANKr"] >= cfg_set["min_TRT_rank"]]*10
    cell_info_df["Border_cell"] = False

    ## Read file with displaced TRT centres:
    orig_disp_TRT = "disp" if cfg_set["displ_TRT_cellcent"] else "orig"
    filename = path_creator_vararr(orig_disp_TRT,"TRT",cfg_set)
    TRTarr = load_file(filename,var_name="TRT")
    
    ## Create Meshgrid or boolean array
    if cfg_set["stat_sel_form"] == "square":
        X = None
        Y = None
    elif cfg_set["stat_sel_form"] == "circle":
        X, Y = np.meshgrid(np.arange(0,TRTarr.shape[2]),np.arange(0,TRTarr.shape[1]))
    else: raise ValueError("stat_sel_form can only be 'square' or 'circle'")
    
    ## Create array to save indices in:
    time_delta_coord = np.arange(cfg_set["n_integ"],dtype=np.int16) * cfg_set["timestep"]
    arr_index        = np.zeros((cell_info_df.shape[0],len(time_delta_coord),cfg_set["stat_sel_form_size"]),dtype=np.uint32)
    xarr_index_flat = xr.DataArray(arr_index,
                              coords=[np.array(cell_info_df.index.tolist(),dtype=np.object),
                                      time_delta_coord, #time_dir_coord,
                                      np.arange(cfg_set["stat_sel_form_size"],dtype=np.int32)],
                              dims=['TRT_ID', 'time_delta', 'pixel_indices'], #'time_dir', 
                              name="TRT_domain_indices")
    xarr_index_ij  = xr.DataArray(np.zeros((cell_info_df.shape[0],len(time_delta_coord),2),dtype=np.uint16),
                              coords=[np.array(cell_info_df.index.tolist(),dtype=np.object),
                                      time_delta_coord,
                                      np.array(['CHi','CHj'],dtype=np.object)],
                              dims=['TRT_ID', 'time_delta', 'CHi_CHj'], #'time_dir', 
                              name="TRT_cellcentre_indices")
                              
    ## Save nc file showing domains around TRT cells for control:
    if cfg_set["save_TRT_domain_map"]:
        TRTarr_plot = TRTarr.copy()
        TRTarr_plot[:,:,:] = int(-1)
    
    ## Loop over TRT-cells to read in the statistics:
    for cell in cell_info_df.index.tolist():
        ind_triple = zip(*np.where(TRTarr==int(cell[8:])))
        
        if len(ind_triple) != cfg_set["n_integ"]:
            if len(ind_triple) < cfg_set["n_integ"]: print("   *** Warning: TRT cell centre is moved out of domain ***")
            if len(ind_triple) > cfg_set["n_integ"]: raise ValueError("cell centres occurring more than once in the same time step")
        border_cell = False
        for cell_index in ind_triple:
            ## Append information on cell centres:
            xarr_index_ij.loc[cell,cell_index[0]*cfg_set["timestep"],:] = [cell_index[1],cell_index[2]]
            
            ## Get indices of domain of specific TRT cell:
            indices = get_indices_of_domain(cell_index[1],cell_index[2],cfg_set,X,Y)[0]
            
            ## Check whether all pixels are within ccs4 domain:
            if indices.shape == (xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],:].values).shape:
                xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],:] = indices
            else:
                if not border_cell:
                    print("   *** Warning: Domain around TRT cell %s crosses observational domain ***" % cell)
                    cell_info_df.loc[cell_info_df.index==cell,"Border_cell"] = True
                border_cell = True
                xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],range(indices.shape[0])] = indices
            if cfg_set["save_TRT_domain_map"]: 
                TRTarr_plot[cell_index[0],:,:].flat[xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],:].values] *= 0
                #TRTarr_plot[cell_index[0],:,:].flat[xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],:].values] += int(cell)
        
    ## Create Xarray file containing the TRT information and the domain-of-interest indices:
    xr_ind_ds = xr.Dataset.from_dataframe(cell_info_df)
    xr_ind_ds.rename({"index": "TRT_ID"},inplace=True)
    xr_ind_ds = xr.merge([xr_ind_ds,xarr_index_flat,xarr_index_ij])
    
    ## Rename ID_TRT to Date_TRT_ID, including date of reading:
    xr_ind_ds.rename({"TRT_ID": "DATE_TRT_ID"},inplace=True)
    xr_ind_ds["DATE_TRT_ID"] = np.array([cfg_set["t0"].strftime("%Y%m%d%H%M")+"_"+TRT_ID for TRT_ID in xr_ind_ds["DATE_TRT_ID"].values],
                                        dtype=np.object)
    
    ## Save xarray object to temporary location:
    disp_reverse_str = "" if not cfg_set["future_disp_reverse"] else "_rev"
    filename = "%stmp/%s%s%s%s" % (cfg_set["root_path"],
                                   cfg_set["t0"].strftime("%Y%m%d%H%M"),
                                   "_stat_pixcount",disp_reverse_str,".pkl")
    with open(filename, "wb") as output_file: pickle.dump(xr_ind_ds, output_file, protocol=-1)
    
    ## Save nc file with TRT domains to disk:
    if cfg_set["save_TRT_domain_map"]: 
        TRTarr_plot += 1
        TRTarr_plot = np.array(TRTarr_plot,dtype=np.float32)
        filename = "%stmp/%s%s%s%s" % (cfg_set["root_path"],
                                       cfg_set["t0"].strftime("%Y%m%d%H%M"),
                                       "_TRT_disp_domain",disp_reverse_str,".nc")
        save_nc(filename,TRTarr_plot,"TRT",np.float32,"-","Domain around TRT cells",cfg_set["t0"],"",dt=5)

## Get variable combination as vararr array:
def get_variable_combination(var,cfg_set,cfg_var,cfg_var_combi,file_type):
    """ Read ingredients of channel/variable combination and return combination
    as variable array (according to the respective csv config file)
    """
    
    ## Get channel combination and take simple difference:
    if cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"OPERATION"].values=="diff":
        var_1_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_1"].values[0]
        vararr_1   = load_file(path_creator_vararr(file_type,var_1_name,cfg_set),var_1_name)
        
        var_2_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_2"].values[0]
        vararr_2   = load_file(path_creator_vararr(file_type,var_2_name,cfg_set),var_2_name)
        
        vararr_return = vararr_1 - vararr_2
    ## Get channel combination and make mixed summing-difference operation:
    elif cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"OPERATION"].values=="sum_2diff":
        var_1_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_1"].values[0]
        vararr_1   = load_file(path_creator_vararr(file_type,var_1_name,cfg_set),var_1_name)
        
        var_2_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_2"].values[0]
        vararr_2   = load_file(path_creator_vararr(file_type,var_2_name,cfg_set),var_2_name)
        
        var_3_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_3"].values[0]
        vararr_3   = load_file(path_creator_vararr(file_type,var_3_name,cfg_set),var_3_name)
        
        vararr_return = vararr_1 + vararr_2 - 2*vararr_3
    elif cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"OPERATION"].values=="none":
        raise ValueError("Variable/Channel combination only implemented for operations which are not 'none'")
    return(vararr_return)
        
        
## Gather statistics and pixel counts and save into xarray structure:
def append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=False):
    """ Wrapper function to get:
        - Domain indices
        - Variables to read statistics and pixel counts from
        - Read the actual statistics and pixel counts (NaN and "minimum-value").
    """

    ## Change settings related calculating statistics from future or current observations:
    cfg_set = cfg_set_input.copy()
    print_reverse = "future" if reverse else "past"
    cfg_set["future_disp_reverse"] = True  if reverse else False
    cfg_set["time_change_factor"]  = -1    if reverse else 1
    print("Read statistics of %s observations related to time step: %s" %
          (print_reverse,cfg_set["t0"].strftime("%Y-%m-%d %H:%M")))
    cfg_set["verbose"] = False
    
    ## Read file with TRT domain indices:
    disp_reverse_str = "" if not cfg_set["future_disp_reverse"] else "_rev"
    filename = "%stmp/%s%s%s%s" % (cfg_set["root_path"],
                                   cfg_set["t0"].strftime("%Y%m%d%H%M"),
                                   "_stat_pixcount",disp_reverse_str,".pkl")
    with open(filename, "rb") as output_file: xr_stat_pixcount = pickle.load(output_file)

    ## Check list of statistics (hard-coded):
    if any(cfg_set["stat_list"] != ["SUM","MEAN","STDDEV","MIN","PERC01","PERC05","PERC25",
                                    "PERC50","PERC75","PERC95","PERC99","MAX"]):
        raise NotImplementedError("Different of statistics than implemented.")
    xr_stat_pixcount["statistic"]   = cfg_set["stat_list"]
    xr_stat_pixcount["pixel_count"] = cfg_set["pixcount_list"][:2]
    
    ## Define file type to be read (depending on residual field correction method):
    if cfg_set["displ_TRT_cellcent"]:
        file_type = "orig"
    else:
        if cfg_set["instant_resid_corr"] or cfg_set["resid_method"]=="None":
            file_type = "disp"
        elif cfg_set["resid_method"]=="Twostep":
            file_type = "disp_resid"
        elif cfg_set["resid_method"]=="Onestep":
            file_type = "disp_resid_combi"
        
    ## Read in the number of "minimum-value" and NaN pixels and calculate the statistics:
    dtype_pixc, fill_value_pc = fill_value_pixelcount(cfg_set)
    for var in cfg_set["var_list"]+cfg_set["var_combi_list"]:
        calculate_statistics_pixcount(var,cfg_set,cfg_var,cfg_var_combi,file_type,xr_stat_pixcount,dtype_pixc,fill_value_pc)
        if False:
            plt.clf()
            time_fac = -cfg_set["time_change_factor"]
            plt.plot(time_fac*xr_stat_pixcount["time_delta"],np.moveaxis(xr_stat_pixcount[var+"_stat"].values[:,:,1],0,1))
            #plt.plot(-xr_stat_pixcount["time_delta"],np.moveaxis(xr_stat_pixcount[var+"_stat"].values[:,:,-3],0,1))
            plt.title("Mean of variable "+var+" ("+print_reverse+")")
            #plt.title("95% percentile of variable: "+var)
            plt.pause(1)
    
    ## Add minus sign to time_delta coordinate for observations in the past:
    if not reverse: xr_stat_pixcount["time_delta"] *= -1
    
    ## Save the respective pickle file to the temporary location on the disk:
    with open(filename, "wb") as output_file: pickle.dump(xr_stat_pixcount, output_file, protocol=-1)
    
    ## Save control-image if necessary:
    if cfg_set["save_stat_ctrl_imag"]:
        filename_fig = "%stmp/%s%s%s%s" % (cfg_set["root_path"],cfg_set["t0"].strftime("%Y%m%d%H%M"),
                                           "_RZC_stat",disp_reverse_str,".pdf")
        plot_var = cfg_set["var_list"][0]+"_stat" if "RZC" not in cfg_set["var_list"] else "RZC_stat"
        #""" DELETE +"_nonmin" in row below """
        #xr_stat_pixcount[plot_var+"_nonmin"][:,:,1].plot.line(x="time_delta",add_legend=False)
        xr_stat_pixcount[plot_var][:,:,1].plot.line(x="time_delta",add_legend=False)
        plt.savefig(filename_fig,format="pdf")
        plt.close()
    
    ## Potential parallelised version (DEPRECATED)
    #num_cores = np.max([multiprocessing.cpu_count()-2,1])
    #print("  Parallelising displacement with %s cores" % num_cores)
    #Parallel(n_jobs=num_cores)(delayed(calculate_statistics_pixcount)(var,cfg_set,
    #         cfg_var,cfg_var_combi,file_type,xr_stat_pixcount,dtype_pixc,fill_value_pc) for var in cfg_set["var_list"]+cfg_set["var_combi_list"])
    
## Calculate statistics and pixel counts for specific variable array:
def calculate_statistics_pixcount(var,cfg_set,cfg_var,cfg_var_combi,file_type,xr_stat_pixcount,dtype_pixc,fill_value_pc):
    """ Function reading the actual statistics and pixel counts for variable 'var'.
    
    Parameters
    ----------
    
    file_type : string
        String specifying which type of non-displaced (orig) or displaced (disp & resid or resid_combi) to be read
        
    xr_stat_pixcount : xarray object
        Object where into the statistics and pixels counts are written (with the information on the TRT cells
        already written into it)
        
    dtype_pixc : numpy.dtype object
        Data type of pixel count (if domain is small enough, a lower precision uint dtype can be chosen)
        
    fill_value_pc : int
        Fill value in case no pixels can be counted (e.g. in case no NaN pixels are within domain)
    """
    
    if var in cfg_set["var_list"]:
        ## Change setting of file type for U_OFLOW and V_OFLOW variable:
        if var in ["U_OFLOW","V_OFLOW"]:
            var_name = "Vx" if var=="U_OFLOW" else "Vy"
            if file_type=="orig":
                file_type_UV = "standard"
            else: file_type_UV = "resid" if file_type=="disp_resid" else "resid_combi"
            vararr   = load_file(path_creator_UV_disparr(file_type_UV,cfg_set),var_name)
        else:
            vararr  = load_file(path_creator_vararr(file_type,var,cfg_set),var)
        min_val = cfg_set["minval_dict"][var]
    elif var in cfg_set["var_combi_list"]:
        ## Get variable combination:
        vararr = get_variable_combination(var,cfg_set,cfg_var,cfg_var_combi,file_type)  
        min_val = np.nan

    if cfg_set["verbose"]: print("  read statistics for "+var)
    
    ## Fill nan-values in COSMO_CONV fields:
    if np.any(np.isnan(vararr)) and \
       cfg_var.loc[cfg_var["VARIABLE"]==var,"SOURCE"].values=="COSMO_CONV":
        t1_inter = datetime.datetime.now()
        vararr = interpolate_COSMO_fields(vararr, method="KDTree")
        t2_inter = datetime.datetime.now()
        if var=="RELHUM_85000": print("   Elapsed time for interpolating the data in %s: %s" % (var,str(t2_inter-t1_inter)))
    
    ## Calculate local standard deviation of specific COSMO_CONV fields:
    if cfg_var.loc[cfg_var["VARIABLE"]==var,"VARIABILITY"].values:
        t1_std = datetime.datetime.now()
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                           [-10+0j, 0+ 0j, +10 +0j],
                           [ -3+3j, 0+10j,  +3 +3j]])
        #plt.imshow(vararr[2,:,:]); plt.show()
        for t in range(vararr.shape[0]):
            vararr[t,:,:] = np.absolute(signal.convolve2d(vararr[t,:,:], scharr,
                                        boundary='symm', mode='same'))
        #plt.imshow(vararr[2,:,:]); plt.show()
        t2_std = datetime.datetime.now()
        if var=="POT_VORTIC_70000": print("   Elapsed time for finding the local standard deviation in %s: %s" % (var,str(t2_std-t1_std)))
        
    ## Smooth (COSMO) fields:
    if cfg_var.loc[cfg_var["VARIABLE"]==var,"SMOOTH"].values:
        t1_smooth = datetime.datetime.now()
        #if var=="RELHUM_85000": plt.imshow(vararr[3,:,:]); plt.title(var); plt.pause(.5)
        for t in range(vararr.shape[0]): vararr[t,:,:] = ndimage.gaussian_filter(vararr[t,:,:],cfg_set["smooth_sig"])
        #if var=="RELHUM_85000": plt.imshow(vararr[3,:,:]); plt.title(var+" smooth"); plt.show() #pause(.5)
        t2_smooth = datetime.datetime.now()
        if var=="RELHUM_85000": print("   Elapsed time for smoothing the fields of %s: %s" % (var,str(t2_smooth-t1_smooth)))
    
    ## Read in statistics and pixel counts / read in category counts:
    t1_stat = datetime.datetime.now()
    if var not in ["CMA","CT"]:
        ## Read in values at indices:
        vararr_sel = vararr.flat[xr_stat_pixcount["TRT_domain_indices"].values].astype(np.float32)       
        if np.any(xr_stat_pixcount["TRT_domain_indices"].values==0):
            vararr_sel[xr_stat_pixcount["TRT_domain_indices"].values==0] = np.nan
        
        """
        ## Get count of nans and minimum values:
        array_pixc = np.stack([np.sum(np.isnan(vararr_sel),axis=2),
                               np.sum(vararr_sel<=min_val,axis=2)],axis=2)
        xr_stat_pixcount[var+"_pixc"] = (('DATE_TRT_ID', 'time_delta', 'pixel_count'), array_pixc)
        """
        ## Calculate the actual statistics:
        perc_values = [0,1,5,25,50,75,95,99,100]
        """
        array_stat = np.array([np.sum(vararr_sel,axis=2),  #nansum
                               np.mean(vararr_sel,axis=2), #nanmean
                               np.std(vararr_sel,axis=2)]) #nanstd
        array_stat = np.moveaxis(np.concatenate([array_stat,np.percentile(vararr_sel,perc_values,axis=2)]),0,2) #nanpercentile
        xr_stat_pixcount[var+"_stat"] = (('DATE_TRT_ID', 'time_delta', 'statistic'), array_stat)
        """
        
        ## Add specific statistics for Radar variables, only analysing values above minimum value:
        if cfg_set["source_dict"][var]=="RADAR":
            vararr_sel[vararr_sel<=min_val] = np.nan
            array_stat_nonmin = np.array([np.nansum(vararr_sel,axis=2),
                                          np.nanmean(vararr_sel,axis=2),
                                          np.nanstd(vararr_sel,axis=2)])
            array_stat_nonmin = np.moveaxis(np.concatenate([array_stat_nonmin,np.nanpercentile(vararr_sel,perc_values,axis=2)]),0,2)
            xr_stat_pixcount[var+"_stat_nonmin"] = (('DATE_TRT_ID', 'time_delta', 'statistic'), array_stat_nonmin)
        
        if False: #var=="lat_1" or var=="lon_1": 
            print(xr_stat_pixcount[var+"_stat"].values[21,:,1])
            print(xr_stat_pixcount[var+"_stat"].values[21,:,-3])
            print(xr_stat_pixcount[var+"_pixc"].values[21,:,0])
            plt.plot(-xr_stat_pixcount["time_delta"],np.moveaxis(xr_stat_pixcount[var+"_stat"].values[:,:,1],0,1))
            #plt.plot(-xr_stat_pixcount["time_delta"],np.moveaxis(xr_stat_pixcount[var+"_stat"].values[:,:,-3],0,1))
            plt.title("Mean of variable: "+var)
            #plt.title("95% percentile of variable: "+var)
            plt.show() #pause(2)
        
    else:
        ## Read in values at indices:
        vararr_sel = vararr.flat[xr_stat_pixcount["TRT_domain_indices"].values]
        
        ## Get count different categories:
        raise ImplementationError("Categorical counting not yet implemented")
    t2_stat = datetime.datetime.now()
    if var=="RELHUM_85000": print("   Elapsed time for calculating the statistics of %s: %s" % (var,str(t2_stat-t1_stat)))
    
    ## Read number of pixels with max-echo value higher than 57dBZ
    if var=="CZC":
        xr_stat_pixcount[var+"_lt57dBZ"] = (('DATE_TRT_ID', 'time_delta'), np.sum(vararr_sel>57.,axis=2))
        #print("   Max CZC value: %s" % np.nanmax(vararr_sel))
        #print("   Number of CZC pixels > 57dBZ: %s" % np.sum(vararr_sel>57.,axis=2))

        
## Get the fill value for pixel count variable in xarray:
def fill_value_pixelcount(cfg_set):
    if (cfg_set["stat_sel_form"]=="circle" and cfg_set["stat_sel_form_width"]>18) or \
       (cfg_set["stat_sel_form"]=="square" and cfg_set["stat_sel_form_width"]>15):
        dtype_pixc = np.uint16
        fill_value = 2**16-1
    else:
        dtype_pixc = np.uint8
        fill_value = 2**8-1
    return dtype_pixc, fill_value

## Interpolating COSMO fields:
def interpolate_COSMO_fields(vararr, method="KDTree"):
    """ Interpolating COSMO fields fills NAN holes (due to topography or singularities)
    
    Parameters
    ----------
    
    vararr : numpy array
        Array with time_delta as the first dimension of COSMO variables.
        
    method : string
        Either 'KDTree' (default) or 'interpol_griddata'
    
    """
    
    ## Interpolating in 3D -> Less efficient
    """
    plt.imshow(vararr[3,:,:]); plt.title(var+" nan"); plt.pause(1)
    x = np.arange(0, vararr.shape[1])
    y = np.arange(0, vararr.shape[0])
    z = np.arange(0, vararr.shape[2])
    xx, yy, zz = np.meshgrid(x, y, z)
    bool_nanarr = np.isnan(vararr)
    x_nan    = xx[bool_nanarr]
    y_nan    = yy[bool_nanarr]
    z_nan    = zz[bool_nanarr]
    x_nonnan = xx[~bool_nanarr]
    y_nonnan = yy[~bool_nanarr]
    z_nonnan = zz[~bool_nanarr]
    vararr_nonnan = vararr[~bool_nanarr]
    vararr[bool_nanarr] = interpolate.griddata((x_nonnan, y_nonnan, z_nonnan), vararr_nonnan.ravel(),
                                               (x_nan, y_nan, z_nan), method='nearest')
    t2 = datetime.datetime.now()
    print("   Elapsed time using 3D method: %s" % (str(t2-t1)))
    plt.imshow(vararr[3,:,:]); plt.title(var+" interpol"); plt.pause(1)
    """
    
    ## Define grid on which to interpolate
    x = np.arange(0, vararr[0,:,:].shape[1])
    y = np.arange(0, vararr[0,:,:].shape[0])
    xx, yy = np.meshgrid(x, y)
    vararr2 = vararr.copy()
    
    if method=="interpol_griddata":
        for t in range(vararr.shape[0]):        
            ## Extract array and mask nan-values
            bool_nanarr = np.isnan(vararr[t,:,:])
            
            ## Get coordinates of nan and non-nan values:
            x_nan    = xx[bool_nanarr]
            y_nan    = yy[bool_nanarr]
            x_nonnan = xx[~bool_nanarr]
            y_nonnan = yy[~bool_nanarr]
            vararr_nonnan = vararr[t,~bool_nanarr]
            
            ## Interpolate at points with missing values
            vararr[t,bool_nanarr] = interpolate.griddata((x_nonnan, y_nonnan), vararr_nonnan.ravel(),
                                                         (x_nan, y_nan), method='nearest')
    
    elif method=="KDTree":
        ## Extract array and mask nan-values
        bool_nanarr_2d = np.any(np.isnan(vararr),axis=0)
        x_nan    = xx[bool_nanarr_2d]
        y_nan    = yy[bool_nanarr_2d]
        x_nonnan = xx[~bool_nanarr_2d]
        y_nonnan = yy[~bool_nanarr_2d]
        tree = spatial.cKDTree(zip(x_nonnan.ravel(), y_nonnan.ravel()), leafsize=2)
        _, inds  = tree.query(zip(x_nan.ravel(),y_nan.ravel()), k=1)
        
        ## Assign nearest values to nan-pixels:
        for t in range(vararr.shape[0]):
            vararr[t,bool_nanarr_2d] = vararr[t,~bool_nanarr_2d].flatten()[inds]                           
        
    else: raise ImplementationError("No other interpolation method implemented")
    return vararr
        

def solartime(time, lon_loc_rad, sun=ephem.Sun()):
    """Return sine and cosine value of solar hour angle depending on longitude and time of TRT cell at t0"""
    obs = ephem.Observer()
    obs.date = time; obs.lon = lon_loc_rad
    sun.compute(obs)
    ## sidereal time == ra (right ascension) is the highest point (noon)
    angle_rad  = ephem.hours(obs.sidereal_time() - sun.ra + ephem.hours('12:00')).norm
    return np.sin(angle_rad), np.cos(angle_rad)

def add_aux_static_variables(ds, cfg_set):
    """This function adds static auxilary variables like solar time (sin/cos), topography information and
    the quality information based on the frequency map"""
    
    ## Get paths to the auxiliary datasets:
    config = configparser.RawConfigParser()
    config.read(u"%s/%s" % (cfg_set["CONFIG_PATH"],cfg_set["CONFIG_FILE_set"]))
    config_aux = config["aux_data_read"]

    ## Add statistics on altitude, slope, and the alignment of the aspect vector with the flow vector:
    ## Define percentiles:
    if any(cfg_set["stat_list"] != ["SUM","MEAN","STDDEV","MIN","PERC01","PERC05","PERC25",
                                    "PERC50","PERC75","PERC95","PERC99","MAX"]):
        raise NotImplementedError("Different of statistics than implemented.")
    perc_values = [0,1,5,25,50,75,95,99,100]    
    
    ## Check that 'TRT_domain_indices' are integer, otherwise convert to uint:
    if not np.issubdtype(ds["TRT_domain_indices"].dtype,np.integer):
        if np.max(ds.TRT_domain_indices.values) < 65535.:
            ds["TRT_domain_indices"] = ds["TRT_domain_indices"].astype(np.uint16,copy=False)
        else: ds["TRT_domain_indices"] = ds["TRT_domain_indices"].astype(np.uint32,copy=False)
        
    ## Check whether topography information should be added:
    alt_var_ls = ["Aspect"] #"Altitude","Slope","Aspect"]
    if set(alt_var_ls).issubset(cfg_set["var_list"]):
        ## Add topography information:
        ds_alt = xr.open_dataset(config_aux["path_altitude_map"])

        for alt_var in list(set(alt_var_ls).intersection(cfg_set["var_list"])):
            print("  Get statistics of topography variable '%s'" % alt_var)
            DEM_vals = ds_alt[alt_var].values.flat[ds.TRT_domain_indices.values]
            if alt_var == "Aspect":
                # Get x- and y-component of 2d direction of the aspect-vector:
                x_asp   = np.cos(DEM_vals)
                y_asp   = np.sin(DEM_vals)
                del(DEM_vals)

                ## Get u- and v-component of optical flow and extent to the same extent as x_asp/y_asp (to the number of
                ## pixels in domain of interest = DEM_vals.shape[2]):
                #u_oflow = np.repeat(ds.U_OFLOW_stat.sel(statistic="PERC50").values[:,:,np.newaxis],x_asp.shape[2],axis=2)
                u_oflow = ds.U_OFLOW_stat.sel(statistic="PERC50")
                #v_oflow = np.repeat(ds.U_OFLOW_stat.sel(statistic="PERC50").values[:,:,np.newaxis],x_asp.shape[2],axis=2)
                v_oflow = ds.V_OFLOW_stat.sel(statistic="PERC50")
                denominator_2 = np.sqrt(u_oflow**2+v_oflow**2)
                
                ## Calculate aspect-flow-alignment factor:
                DEM_vals      = np.zeros(DEM_vals.shape)
                print("   Looping through %s pixel of TRT domain:" % DEM_vals.shape[2])
                for pix_ind in np.arange(DEM_vals.shape[2]):
                    if pix_ind%50==0: print("\r     Working on pixel index %s" % pix_ind)
                    numerator     = u_oflow*x_asp[:,:,pix_ind] + v_oflow*y_asp[:,:,pix_ind];   #print("     Calculated the numerator")
                    denominator_1 = np.sqrt(x_asp[:,:,pix_ind]**2+y_asp[:,:,pix_ind]**2)
                    #del(x_asp); del(y_asp)
                    #denominator_2 = np.sqrt(u_oflow**2+v_oflow**2)
                    #del(u_oflow); del(v_oflow)
                    denominator   = denominator_1*denominator_2;     #print("     Calculated the denominator")
                    #del(denominator_1)#; del(denominator_2)
                    DEM_vals[:,:,pix_ind]      = -numerator/denominator;          #print("     Calculated the Alignment")
                    #del(numerator); del(denominator)
                del(denominator, numerator, denominator_1, denominator_2, u_oflow, v_oflow, x_asp, y_asp)
            
            ## Calcualte the statistics:
            array_stat = np.array([np.sum(DEM_vals,axis=2),  #nansum
                                   np.mean(DEM_vals,axis=2), #nanmean
                                   np.std(DEM_vals,axis=2)]) #nanstd
            print("   Calculated sum / mean / standard deviation")
            array_stat = np.moveaxis(np.concatenate([array_stat,np.percentile(DEM_vals,perc_values,axis=2)]),0,2)
            print("   Calculated quantiles")

            ## Add variable to dataset:
            ds[alt_var+"_stat"] = (('DATE_TRT_ID', 'time_delta', 'statistic'), array_stat)
         
    ## Check whether topography information should be added:
    if "Radar_Freq_Qual" in cfg_set["var_list"]:
        print("  Get radar frequency qualitiy information")
        
        ## Import radar frequency map:
        from PIL import Image
        frequency_data = swissradar.convertToValue(Image.open(config_aux["path_frequency_image"]),
                                                   config_aux["path_frequency_scale"])
        frequency_data[np.logical_or((frequency_data >= 9999.0),(frequency_data <= 0.5))] = np.nan
        
        ## Get values in TRT domains:
        qual_vals = frequency_data.flat[ds.TRT_domain_indices.values]
        
        ## Calcualte the statistics:
        array_stat = np.array([np.sum(qual_vals,axis=2),  #nansum
                               np.mean(qual_vals,axis=2), #nanmean
                               np.std(qual_vals,axis=2)]) #nanstd
        print("   Calculated sum / mean / standard deviation")
        array_stat = np.moveaxis(np.concatenate([array_stat,np.percentile(qual_vals,perc_values,axis=2)]),0,2)
        print("   Calculated quantiles")

        ## Add variable to dataset:
        ds["Radar_Freq_Qual_stat"] = (('DATE_TRT_ID', 'time_delta', 'statistic'), array_stat)
    
    ## Check whether solar time information should be added:
    solar_time_ls = ["Solar_Time_sin","Solar_Time_cos"]
    if set(solar_time_ls).issubset(cfg_set["var_list"]):
        print("  Get local solar time (sind & cos component)")
        
        ## Sin and Cos element of local solar time:
        time_points = [datetime.datetime.strptime(DATE_TRT_ID_date, "%Y%m%d%H%M") for DATE_TRT_ID_date in ds.date.values]
        lon_loc_rad = np.deg2rad(ds.lon_1_stat.sel(statistic="MEAN",time_delta=0).values)
        solar_time_sincos = [solartime(time_points_i,lon_loc_rad_i) for time_points_i,lon_loc_rad_i in zip(time_points,lon_loc_rad)]
        ds["Solar_Time_sin"] = (('DATE_TRT_ID'), np.array(solar_time_sincos)[:,0])
        ds["Solar_Time_cos"] = (('DATE_TRT_ID'), np.array(solar_time_sincos)[:,1])
      
    return ds
    
    """
    ## Remove 'time_delta' coordinate in TRT variables (which are only available for t0):
    ds_keys  = np.array(ds.keys())
    keys_TRT = ds_keys[np.where(["stat" not in key_ele and "pixc" not in key_ele for key_ele in ds_keys])[0]]
    keys_TRT_timedelta = ["TRT_domain_indices","TRT_cellcentre_indices","CZC_lt57dBZ","pixel_indices","CHi_CHj","pixel_count"]
    for key_TRT in keys_TRT:
        if key_TRT in keys_TRT_timedelta: continue
        print(key_TRT)
        ds[key_TRT] = ds[key_TRT].sel(time_delta=0).drop("time_delta")
    """

## Read xarray files from disk, depending on file ending (as .pkl or .nc file):
def xarray_file_loader(path_str):
    import psutil
    if path_str[-3:]==".nc":
        expected_memory_need = float(os.path.getsize(path_str))/psutil.virtual_memory().available*100
        if expected_memory_need > 35:
            print("  *** Warning: File %i is opened as dask dataset (expected memory use: %02d%%) ***" %\
                  (path_number, expected_memory_need))
            xr_n = xr.open_mfdataset(path_str,chunks={"DATE_TRT_ID":1000})
        else: xr_n = xr.open_dataset(path_str)
    elif path_str[-4:]==".pkl":
        with open(path_str, "rb") as path: xr_n = pickle.load(path)
    return xr_n
    
## Calculate TRT Rank:
def calc_TRT_Rank(xr_stat,ET_option="cond_median"):
    """Calculate TRT Rank for square/circle with CCS4 radar data.
    The option "ET_option" states whether from the EchoTop 45dBZ values
    the conditional median (of all pixels where ET45 is non-zero),
    the median over all pixels, or the max of all pixels should be used."""
    
    if ET_option not in ["cond_median","all_median","all_max"]:
        raise ValueError("variable 'ET_option' has to be either 'cond_median', 'all_median', or 'all_max'")

    ## Read the variables:
    VIL_scal  = xr_stat.LZC_stat.sel(statistic="MAX")              ## Vertical integrated liquid (MAX)
    ME_scal   = xr_stat.CZC_stat.sel(statistic="MAX")              ## MaxEcho (Max)
    A55_scal  = xr_stat.CZC_lt57dBZ                                ## N pixels >57dBZ (#)
    
    if ET_option == "cond_median":
        ET45_scal = xr_stat.EZC45_stat_nonmin.sel(statistic="PERC50")  ## EchoTop 45dBZ (cond. Median)
    elif ET_option == "all_median":
        ET45_scal = xr_stat.EZC45_stat.sel(statistic="PERC50")         ## EchoTop 45dBZ (Median)
    elif ET_option == "all_max":
        ET45_scal = xr_stat.EZC45_stat.sel(statistic="MAX")            ## EchoTop 45dBZ (MAX)
    
    ## Scale variables to values between min and max according to Powerpoint Slide
    ## M:\lom-prod\mdr-prod\oper\adula\Innovation\6224_COALITION2\06-Presentations\2018-10-17_TRT_Workshop-DACH_MWO_hea.pptx:
    VIL_scal.values[VIL_scal.values>65.]   = 65. ## Max  VIL:       56 kg m-2
    ME_scal.values[ME_scal.values<45.]     = 45. ## Min MaxEcho:    45 dBZ
    ME_scal.values[ME_scal.values>57.]     = 57. ## Max MaxEcho:    57 dBZ
    ET45_scal.values[ET45_scal.values>10.] = 10. ## Max EchoTop:    10 km
    A55_scal.values[A55_scal.values>40.]   = 40. ## Max pix >57dBZ: 40
    
    ## Scale variables to values between 0 and 4:    
    VIL_scal  = VIL_scal/65.*4
    ET45_scal = ET45_scal/10.*4
    ME_scal   = (ME_scal-45.)/12.*4
    A55_scal  = A55_scal/40.*4
    
    ## Calculate TRT rank:
    TRT_Rank = (2.*VIL_scal+2*ET45_scal+ME_scal+2.*A55_scal)/7.
    TRT_Rank = TRT_Rank.drop("statistic")
    xr_stat["TRT_Rank"] = (('DATE_TRT_ID', 'time_delta'), TRT_Rank)
    
    ## Calculate TRT rank difference to t0:
    TRT_Rank_diff = TRT_Rank - TRT_Rank.sel(time_delta=0)
    xr_stat["TRT_Rank_diff"] = (('DATE_TRT_ID', 'time_delta'), TRT_Rank_diff)
    
    return(xr_stat)
    
## Add derived information (e.g. TRT-Rank):
def add_derived_variables(stat_path):
    print("Adding derived information:")
    file_path = os.path.join(stat_path,"Combined_stat_pixcount.pkl")
    xr_stat = xarray_file_loader(file_path)
    
    ## Add TRT-Rank
    print("  Adding TRT Rank:")
    xr_stat = calc_TRT_Rank(xr_stat,ET_option="cond_median")
    
    ## Save Pickle:
    file_new = os.path.join(stat_path,"Combined_stat_pixcount.pkl")
    with open(file_new, "wb") as output_file: pickle.dump(xr_stat, output_file, protocol=-1)
    print("  Saved to pickle file.")
    
    ## Save NetCDF:
    file_new = os.path.join(stat_path,"nc/Combined_stat_pixcount.nc")
    xr_stat.to_netcdf(file_new)
    print("  Saved to NetCDF file.")
    
## Create numpy arrays of variables for quick access:
def create_new_vararray(cfg_set,cfg_var):
    """Create numpy arrays of variables for quick access.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
    """
    
    print("Create new arrays of variables to be displaced...")
    cfg_var_pred = cfg_var.loc[cfg_var["PREDICTOR"]]
    
    ## Differentiate between data sources:
    """"
    var_parallel = cfg_var_pred["VARIABLE"].loc[cfg_var_pred["SOURCE"]=="RADAR"].tolist()
    if cfg_set["displ_TRT_cellcent"]: var_parallel = var_parallel.append("TRT")
    num_cores = np.max([multiprocessing.cpu_count()-2,1])
    print("  Parallelising reading of RADAR/TRT variable with %s cores" % num_cores)
    Parallel(n_jobs=num_cores)(delayed(create_new_vararray_core)(cfg_set,var) for var in var_parallel)
    """
    
    ## Loop over variables for displacement:
    create_new_vararray_core(cfg_set,"TRT")
    source = None
    
    for var in cfg_set["var_list"]:
        source_new = cfg_var["SOURCE"].loc[cfg_var["VARIABLE"]==var].values[0]
        if source_new!=source or var==cfg_set["var_list"][-1]:
            t2 = datetime.datetime.now()
            if source is not None: # and cfg_set["verbose"]:
                print("   Elapsed time for the reading %s variables: %s " % (source,t2-t1))
            t1 = datetime.datetime.now(); source = source_new
        create_new_vararray_core(cfg_set,var)
    
    ## In case the verification should be performed, initialise array with additional information:
    if cfg_set["verify_disp"]:
        filename_verif_stat = "%stmp/%s_%s_stat_verif.npy" % (cfg_set["root_path"],
                                cfg_set["verif_param"],str(cfg_set[cfg_set["verif_param"]]))
        stat_array = np.zeros((1,len(cfg_set["var_list"]),int(cfg_set["n_stat"])),
                               dtype=np.float16)-9999.
        np.save(filename_verif_stat, stat_array)


## Create variable array of specific variable:
def create_new_vararray_core(cfg_set,var):
    """Create variable array of specific variable."""
    if var in ["U_OFLOW","V_OFLOW"]:
        return
    
    t1 = datetime.datetime.now()
    if cfg_set["verbose"]: print("  ... new "+var+" array created in:")
    filename = path_creator_vararr("orig",var,cfg_set)
    vararr = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
    
    ## Get field of every time step (if map-function cannot be applied)
    i = 0
    t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=cfg_set["timestep"])
    if var == "TRT":
        vararr = get_vararr_TRT_t0(cfg_set["t0"], cfg_set)
    else:
        for t_d in t_delta:
            t_current = cfg_set["t0"] - cfg_set["time_change_factor"]*t_d
            vararr_t = get_vararr_t(t_current, var, cfg_set)
            vararr[i,:,:] = vararr_t[0,:,:]
            i += 1
    save_file(filename, data_arr=vararr,var_name=var,cfg_set=cfg_set)
    if cfg_set["verbose"]: print("      "+filename)
        
    ## In case verification of displacements should be performed, also initialise skill-score array:
    if cfg_set["verify_disp"]:
        filename_verif = "%stmp/%s_%s_%s_verif.npy" % (cfg_set["root_path"],
                         cfg_set["verif_param"],str(cfg_set[cfg_set["verif_param"]]), var)
        verif_array = np.zeros((1,len(cfg_set["scores_list"]),cfg_set["n_integ"]-1))-9999.
        np.save(filename_verif, verif_array)
    
    t2 = datetime.datetime.now()
    if False: print("    Elapsed time for creation of variable %s: %s" % (var,str(t2-t1)))
    
    
## Displace fields with current displacement arrays:
def displace_fields(cfg_set, resid=False):
    """Displace fields with current displacement arrays.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
    """
    
    ## Loop over variables for displacement:
    #print("Displace variables to time "+cfg_set["t0"].strftime("%d.%m.%Y %H:%M")+"...")
    
    ## Change suffixes of files to read and write in case residual movements are corrected:
    if not resid:
        input_suffix  = "_orig"
        output_suffix = "_disp"
        input_UV_suffix = "standard"
        append_str = ""
    else:
        input_suffix  = "_disp" if not cfg_set["resid_disp_onestep"] else "_orig"
        output_suffix = "_disp_resid" if not cfg_set["resid_disp_onestep"] else "_disp_resid_combi"
        input_UV_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        append_str    = " for residual movement" if not cfg_set["resid_disp_onestep"] else " for residual movement (combi)"
    print_options = [input_suffix,output_suffix,input_UV_suffix,append_str]
    
    ## Get current UV-field
    filename = path_creator_UV_disparr(input_UV_suffix,cfg_set)
    #filename = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
    #                                              cfg_set["oflow_source"], input_UV_suffix, cfg_set["file_ext_verif"],
    #                                              cfg_set["save_type"])
    #UVdisparr = np.load(filename)
    UVdisparr = load_file(filename)
    Vx = UVdisparr["Vx"][:,:,:]; Vy = UVdisparr["Vy"][:,:,:]
    #if var=="TRT":
    #    UV_t0 = np.moveaxis(np.dstack((-Vx[-1,:,:],-Vy[-1,:,:])),2,0)
    #else:
    UV_t0 = np.moveaxis(np.dstack((Vx[0,:,:],Vy[0,:,:])),2,0)
    
    ## Load displacement array and add up (don't include most recent displacement array,
    ## as the last displacement to t0 is done by the UV array loaded above
    #if var=="TRT":
    #    Dx = (UVdisparr["Dx"][:-1,:,:])[::-1,:,:]; Dy = (UVdisparr["Dy"][:-1,:,:])[::-1,:,:]
    #else:
    Dx = (UVdisparr["Dx"][:,:,:])[1:,:,:]; Dy = (UVdisparr["Dy"][:,:,:])[1:,:,:]
    Dx_sum = np.cumsum(Dx,axis=0); Dy_sum = np.cumsum(Dy,axis=0)
    if cfg_set["displ_TRT_cellcent"]:
        Dx_sum_neg = np.cumsum(-Dx,axis=0)
        Dy_sum_neg = np.cumsum(-Dy,axis=0)
        
    ## Initiate XYW array of semilagrangian function to avoid going through for-loop
    ## for all variables.
    #XYW_prev_0 = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])*np.nan
    #XYW_prev_1 = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])*np.nan
    #precalc_XYW = False
    
    ## Calculate displacement per variable:
    ## Loop over variables for displacement:
    if len(cfg_set["var_list"])>0 and not cfg_set["displ_TRT_cellcent"]:
        if cfg_set["use_precalc_XYW"]:
            XYW_prev_0, XYW_prev_1 = displace_specific_variable(cfg_set["var_list"][0],cfg_set,print_options,
                                                                UV_t0,Dx_sum,Dy_sum)
            
            num_cores = np.max([multiprocessing.cpu_count()-2,1])
            print("  Parallelising displacement with %s cores" % num_cores)
            Parallel(n_jobs=num_cores)(delayed(displace_specific_variable)(var,cfg_set,print_options,UV_t0,Dx_sum,Dy_sum,
                                               XYW_prev_0=XYW_prev_0,XYW_prev_1=XYW_prev_1) for var in cfg_set["var_list"][1:])

        else:
            for var in cfg_set["var_list"]: displace_specific_variable(var,cfg_set,print_options,
                                                                       UV_t0,Dx_sum,Dy_sum)
            #if cfg_set["displ_TRT_cellcent"]:
            #    raise NotImplementedError("So far backdisplacement of TRT cells only implemented if usage ")
    if cfg_set["displ_TRT_cellcent"]: displace_specific_variable("TRT",cfg_set,print_options,
                                                                 UV_t0=None,Dx_sum=Dx_sum_neg,Dy_sum=Dy_sum_neg,
                                                                 Vx=Vx,Vy=Vy)

     
    # for var in cfg_set["var_list"]:
        # if cfg_set["use_precalc_XYW"]:
            # XYW_prev_0, XYW_prev_1 = displace_specific_variable(var,cfg_set,print_options,
                                                                # UV_t0,Dx_sum,Dy_sum)
            # displace_specific_variable(var,cfg_set,print_options,UV_t0,Dx_sum,Dy_sum,
                                       # XYW_prev_0=XYW_prev_0,XYW_prev_1=XYW_prev_1)
        # else:
            # displace_specific_variable(var,cfg_set,print_options,UV_t0,Dx_sum,Dy_sum,
                                       # XYW_prev_0=XYW_prev_0,XYW_prev_1=XYW_prev_1)
    
    
def displace_specific_variable(var,cfg_set,print_options,UV_t0,Dx_sum,Dy_sum,
                               XYW_prev_0=None,XYW_prev_1=None,Vx=None,Vy=None):
    """Core function of displace_fields (variable specific displacement)"""
    ## Check that Vx and Vy are available if UV_t0 is None:
    if UV_t0 is None and (Vx is None or Vy is None):
        raise ValueError("If UV_t0 is None, the complete Vx and Vy arrays have to be provided.")
    
    ## Set config file entry for cfg_set["use_precalc_XYW"] to False if variable is TRT.
    if var=="TRT": cfg_set["use_precalc_XYW"] = False
    
    ## Unpack printing and naming variables:
    input_suffix    = print_options[0]
    output_suffix   = print_options[1]
    input_UV_suffix = print_options[2]
    append_str      = print_options[3]
    
    ## See whether precalculated XYW arrays are available:
    precalc_XYW = False if (XYW_prev_0 is None or XYW_prev_1 is None) else True
    
    if cfg_set["verbose"]: print("  ... "+var+" is displaced%s" % append_str)
    filename = path_creator_vararr(input_suffix,var,cfg_set)
    #filename = "%stmp/%s_%s%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
    #                                   input_suffix, cfg_set["file_ext_verif"], cfg_set["save_type"])
    vararr = load_file(filename,var_name=var)
        
    ## Loop over time steps to create displaced array:
    vararr_disp = np.copy(vararr)
        
    ## Displace every time-step of the variable
    ## Current time step: No displacement
    ## Last time step: Only displacement with current UV-field UV_t0
    adv_method = st.advection.get_method(cfg_set["adv_method"])
    if not precalc_XYW:
        #if var=="TRT": UV_t0 = np.moveaxis(np.dstack((-Vx[0,:,:],-Vy[0,:,:])),2,0)
        #vararr_disp[1,:,:], XYW_prev_ls = adv_method(vararr[1,:,:],UV_t0,1,
        #                                             return_XYW=True)
        if var=="TRT":
            UV_t0 = np.moveaxis(np.dstack((Vx[0,:,:],Vy[0,:,:])),2,0)
            vararr_disp[1,:,:] = adv_method(vararr[1,:,:],UV_t0,1,return_XYW=False,foward_mapping=True)
        else:
            vararr_disp[1,:,:], XYW_prev_ls = adv_method(vararr[1,:,:],UV_t0,1,
                                                         return_XYW=True)
        
        if cfg_set["use_precalc_XYW"]:
            XYW_prev_0 = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])*np.nan
            XYW_prev_1 = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])*np.nan
            XYW_prev_0[0,:,:] = XYW_prev_ls[0]
            XYW_prev_1[0,:,:] = XYW_prev_ls[1]
          
    else: vararr_disp[1,:,:], = adv_method(vararr[1,:,:],UV_t0,1,
                                           XYW_prev=[XYW_prev_0[0,:,:],XYW_prev_1[0,:,:]])
        
    ## For preceding time steps, also provide cumulatively summed displacement field
    for i in range(2,cfg_set["n_integ"]):
        if var=="TRT": UV_t0 = np.moveaxis(np.dstack((Vx[i-1,:,:],Vy[i-1,:,:])),2,0)
        D_prev_arr = np.moveaxis(np.dstack((Dx_sum[i-2,:,:],Dy_sum[i-2,:,:])),2,0)
        if not precalc_XYW:
            if var=="TRT":
                vararr_disp[i,:,:] = adv_method(vararr_disp[i-1,:,:],UV_t0,1,foward_mapping=True)
            else:
                vararr_disp[i,:,:], XYW_prev_ls = adv_method(vararr[i,:,:],UV_t0,1,D_prev=D_prev_arr,
                                                             return_XYW=True)
            if cfg_set["use_precalc_XYW"]:
                XYW_prev_0[i,:,:] = XYW_prev_ls[0]
                XYW_prev_1[i,:,:] = XYW_prev_ls[1]
        else: vararr_disp[i,:,:] = adv_method(vararr[i,:,:],UV_t0,1,D_prev=D_prev_arr,
                                              XYW_prev=[XYW_prev_0[i,:,:],XYW_prev_1[i,:,:]])
                
    ## If first var has been processed, set precalc_XYW from False to True
    #if cfg_set["use_precalc_XYW"]: precalc_XYW = True
                
    ## Save displaced variable:
    filename = path_creator_vararr(output_suffix,var,cfg_set)
    #filename = "%stmp/%s_%s%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
    #                                    output_suffix, cfg_set["file_ext_verif"], cfg_set["save_type"])
    #np.save(filename, vararr_disp)
    save_file(filename, data_arr=vararr_disp,var_name=var,cfg_set=cfg_set)
    if cfg_set["verbose"]: print("      "+var+" displaced array is saved")
            
    ## In case verification of displacements should be performed:
    if cfg_set["verify_disp"] and (cfg_set["resid_disp"]==resid):
        ## Do not calculate skill scores if residual movement is corrected
        ## but currently, the initial displacement was just calculated.
        #if not (cfg_set["resid_disp"] and resid): break
        calc_skill_scores(cfg_set,var,vararr_disp)
        calc_statistics(cfg_set,var,vararr_disp)
    
    ## Retrun precalculated arrays:
    if not precalc_XYW: return XYW_prev_0, XYW_prev_1
            
## Do further displacement to correct for residual movement:
def residual_disp(cfg_set):
    """Do further displacement to correct for residual movement.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
    """
    
    ## Create new displacement array based on discplaced RZC values:
    create_new_disparray(cfg_set,extra_verbose=False,resid=True)
    displace_fields(cfg_set, resid=True)     
    

    
## Displace fields with current displacement arrays:
def plot_displaced_fields(var,cfg_set,future=False,animation=False,TRT_form=False):
    """Plot displaced fields next to original ones.

    Parameters
    ----------
    
    var : str
        Variable which should be plotted
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
    """
    
    """
    if not resid:
        resid_str  = ""
        resid_suffix = "standard"
        resid_suffix_plot = ""
        resid_suffix_vec = "vec"
        resid_dir = ""
    else:
        resid_str  = " residual" if not cfg_set["resid_disp_onestep"] else " residual (combi)"
        resid_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        resid_suffix_plot = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        resid_suffix_vec = "vec_resid" if not cfg_set["resid_disp_onestep"] else "vec"
        resid_dir = "/_resid" if not cfg_set["resid_disp_onestep"] else "/_resid_combi"    
    """
    #resid_str = "" if not resid else " residual"
    #resid_suffix = "" if not resid else "_resid"
    #resid_dir = "" if not resid else "/_resid"
    print("Plot animation of %s..." % (var))
    
    
    ## Load files
    filename_orig = path_creator_vararr("orig",var,cfg_set,disp_reverse="")
    print("   File of original variable:         %s" % filename_orig)
    vararr = load_file(filename_orig,var_name=var)
    if future:
        filename_orig_future = path_creator_vararr("orig",var,cfg_set,disp_reverse="_rev")
        print("   File of future original variable:  %s" % filename_orig_future)
        vararr_future = load_file(filename_orig_future,var_name=var)
        vararr = np.concatenate([vararr_future[1:,:,:][::-1,:,:],vararr],axis=0)
    vararr = vararr[::-1,:,:]
    
    filename_UV = path_creator_UV_disparr("standard",cfg_set,disp_reverse="")
    print("   File of UV field:                  %s" % filename_UV)
    UVdisparr = load_file(filename_UV)
    Vx = UVdisparr["Vx"][:,:,:]; Vy = UVdisparr["Vy"][:,:,:]
    if future:
        filename_UV_future = path_creator_UV_disparr("standard",cfg_set,disp_reverse="_rev")
        print("   File of future UV field:           %s" % filename_UV_future)
        UVdisparr_future = load_file(filename_UV)
        Vx_future = UVdisparr_future["Vx"][1:,:,:][::-1,:,:]
        Vy_future = UVdisparr_future["Vy"][1:,:,:][::-1,:,:]
        Vx = np.concatenate([Vx_future,Vx],axis=0)
        Vy = np.concatenate([Vy_future,Vy],axis=0)    
    Vx = Vx[::-1,:,:]; Vy = Vy[::-1,:,:]
    
    cmap, norm, clevs, clevsStr = st.plt.get_colormap("mm/h", "MeteoSwiss")

    ## Set values to nan
    #plt.hist(vararr[~np.isnan(vararr)].flatten()); plt.show(); return
    if var=="RZC":
        vararr[vararr < 0.1] = np.nan
    elif var=="BZC":
        vararr[vararr <= 0] = np.nan
    elif var=="LZC":
        vararr[vararr < 5] = np.nan
    elif var=="MZC":
        vararr[vararr <= 2] = np.nan
    elif var in ["EZC","EZC45","EZC15"]:
        vararr[vararr < 1] = np.nan
    #elif "THX" in var:
    #    vararr[vararr < 0.001] = np.nan
    
    
    ## Get forms of TRT cells:
    if TRT_form:
        filename_TRT_disp = path_creator_vararr("disp","TRT",cfg_set,disp_reverse="")
        filename_TRT_disp = filename_TRT_disp[:-3]+"_domain.nc"
        print("   File of TRT domain:                %s" % filename_TRT_disp)
        vararr_TRT = load_file(filename_TRT_disp,var_name="TRT")
    
        if future:
            filename_TRT_disp_future = filename_TRT_disp[:-3]+"_rev.nc"
            vararr_TRT_future = load_file(filename_TRT_disp_future,var_name="TRT")
            vararr_TRT = np.concatenate([vararr_TRT_future[1:,:,:][::-1,:,:],vararr_TRT],axis=0)
        vararr_TRT = vararr_TRT[::-1,:,:]
 
    t_delta = np.array(range(cfg_set["n_integ"]))*-cfg_set["timestep"]
    if future:
        t_delta = np.concatenate([np.arange(1,cfg_set["n_integ"])[::-1] * \
                                  cfg_set["timestep"],t_delta])
    t_delta = t_delta[::-1]
    #t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=-cfg_set["timestep"])
    #if future:
    #    t_delta = np.concatenate([np.arange(1,cfg_set["n_integ"])[::-1] * \
    #                              datetime.timedelta(minutes=cfg_set["timestep"]),t_delta])

    ## Setup the plot:
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(10,10)
    plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
    plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
    
    ## Make plots
    for i in range(vararr.shape[0]):
        #if animation: plt.cla()
        
        ## Prepare UV field for quiver plot
        UV_field = np.moveaxis(np.dstack((Vx[i,:,:],Vy[i,:,:])),2,0)
        step = 40; X,Y = np.meshgrid(np.arange(UV_field.shape[2]),np.arange(UV_field.shape[1]))
        UV_ = UV_field[:, 0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]
        X_ = X[0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]; Y_ = Y[0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]
        
        
        ## Make plot:
        if not animation:
            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.set_size_inches(10,10)
            plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
            plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        axes.grid(which='major', color='orange', linestyle='-', linewidth=0.5)
        
        ## Generate title:
        t_current = cfg_set["t0"] + datetime.timedelta(minutes=t_delta[i])
        title_str_basic = "%s fields at %s" % (var, t_current.strftime("%d.%m.%Y - %H:%M"))
        if t_current > cfg_set["t0"]: title_str_addon = "(Future t0 + %02dmin)" % t_delta[i]
        elif t_current < cfg_set["t0"]: title_str_addon = "(Past t0 - %02dmin)" % -t_delta[i]
        elif t_current == cfg_set["t0"]: title_str_addon = "(Present t0 + 00min)"
        title_str = "%s\n%s" % (title_str_basic,title_str_addon)

        #axes.cla()
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        axes.grid(which='major', color='orange', linestyle='-', linewidth=0.5)
        axes.set_title(title_str)
        axes.imshow(vararr[i,:,:], aspect='equal', cmap=cmap)
        axes.quiver(X_, Y_, UV_[0,:,:], -UV_[1,:,:], pivot='tip', color='grey')
        #if cfg_set["UV_inter"] and type(UV_vec[i]) is not float and len(UV_vec[i].shape)>1:
        #    axes.quiver(UV_vec[i][1,:,0], UV_vec[i][0,:,0], UV_vec[i][2,:,0], -UV_vec[i][3,:,0], pivot='tip', color='red')
        #    axes.quiver(UV_vec_sp[i][1,:], UV_vec_sp[i][0,:], UV_vec_sp[i][2,:], -UV_vec_sp[i][3,:], pivot='tip', color='blue')
        #axes[1].set_title('Displaced')
        #axes[1].imshow(vararr_disp[i,:,:], aspect='equal', cmap=cmap)
        if TRT_form:
            #col_ls = ["r","b","g","m","k"]
            #for circ in range(len(iCH_ls)):
            #    col = col_ls[circ%len(col_ls)]
            #    axes[1].contour(circle_array[:,:,circ],linewidths=0.3,alpha=0.7) #,levels=diameters)
            axes.contour(vararr_TRT[i,:,:],linewidths=1,alpha=1,color="red") #,levels=diameters)
        #axes[1].grid(which='major', color='red', linestyle='-', linewidth=1)

        #fig.tight_layout()    
        
        if animation:
            #plt.show()
            plt.pause(1)
            axes.clear()
        else:
            path = "%sTRT_cell_disp/%s/%s/" % (cfg_set["output_path"],cfg_set["t0_str"],var)
            new_dir = ""
            if not os.path.exists(path):
                os.makedirs(path)
                new_dir = "(new) "
            figname = "%s%s_%s_TRT_domains.png" % (path, t_current.strftime("%Y%m%d%H%M"), var)
            fig.savefig(figname, dpi=100)
            if i == vararr.shape[0]-1:
                print("   Plots saved in %sdirectory: %s" % (new_dir, path))
    plt.close()  

    
    
## Displace fields with current displacement arrays:
def plot_displaced_fields_old(var,cfg_set,resid=False,animation=False,TRT_form=False):
    """Plot displaced fields next to original ones.

    Parameters
    ----------
    
    var : str
        Variable which should be plotted
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
    """
    if not resid:
        resid_str  = ""
        resid_suffix = "standard"
        resid_suffix_plot = ""
        resid_suffix_vec = "vec"
        resid_dir = ""
    else:
        resid_str  = " residual" if not cfg_set["resid_disp_onestep"] else " residual (combi)"
        resid_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        resid_suffix_plot = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        resid_suffix_vec = "vec_resid" if not cfg_set["resid_disp_onestep"] else "vec"
        resid_dir = "/_resid" if not cfg_set["resid_disp_onestep"] else "/_resid_combi"    
    
    #resid_str = "" if not resid else " residual"
    #resid_suffix = "" if not resid else "_resid"
    #resid_dir = "" if not resid else "/_resid"
    print("Plot comparison of moving and displaced%s %s..." % (resid_str,var))
    
    
    ## Load files
    if not resid or cfg_set["resid_disp_onestep"]:
        filename_orig = path_creator_vararr("orig",var,cfg_set)
        #filename_orig = "%stmp/%s_%s_orig%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
        #                                           cfg_set["file_ext_verif"], cfg_set["save_type"])
    else:
        filename_orig = path_creator_vararr("disp",var,cfg_set)
        #filename_orig = "%stmp/%s_%s_disp%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
        #                                           cfg_set["file_ext_verif"], cfg_set["save_type"])
    print("   File of original variable:  %s" % filename_orig)
    vararr = load_file(filename_orig,var_name=var)
    resid_suffix_temp = "" if resid_suffix=="standard" else resid_suffix
    filename_disp = path_creator_vararr("disp"+resid_suffix_temp,var,cfg_set)
    #filename_disp = "%stmp/%s_%s_disp%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
    #                                             resid_suffix, cfg_set["file_ext_verif"], cfg_set["save_type"])
    print("   File of displaced variable: %s" % filename_disp)
    vararr_disp = load_file(filename_disp,var_name=var)
    filename_UV = path_creator_UV_disparr(resid_suffix,cfg_set)
    #filename_UV = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
    #                                                  cfg_set["oflow_source"], resid_suffix, cfg_set["file_ext_verif"],
    #                                                  cfg_set["save_type"])
    print("   File of UV field:           %s" % filename_UV)
    if cfg_set["UV_inter"]:
        filename_UV_vec = path_creator_UV_disparr(resid_suffix_vec,cfg_set,save_type="npz")
        #filename_UV_vec = "%stmp/%s_%s_disparr_UV_%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                      cfg_set["oflow_source"], resid_suffix_vec, cfg_set["file_ext_verif"], 
        #                                                      cfg_set["save_type"])
        print("   File of UV vectors: %s" % filename_UV_vec)
    cmap, norm, clevs, clevsStr = st.plt.get_colormap("mm/h", "MeteoSwiss")

    ## Set values to nan
    #plt.hist(vararr[~np.isnan(vararr)].flatten()); plt.show(); return
    if var=="RZC":
        vararr[vararr < 0.1] = np.nan
        vararr_disp[vararr_disp < 0.1] = np.nan
    elif var=="BZC":
        vararr[vararr <= 0] = np.nan
        vararr_disp[vararr_disp <= 0] = np.nan
    elif var=="LZC":
        vararr[vararr < 5] = np.nan
        vararr_disp[vararr_disp < 5] = np.nan
    elif var=="MZC":
        vararr[vararr <= 2] = np.nan
        vararr_disp[vararr_disp <= 2] = np.nan
    elif var in ["EZC","EZC45","EZC15"]:
        vararr[vararr < 1] = np.nan
        vararr_disp[vararr_disp < 1] = np.nan
    elif "THX" in var:
        vararr[vararr < 0.001] = np.nan
        vararr_disp[vararr_disp < 0.001] = np.nan
    
    ## Prepare UV field for quiver plot
    #UVdisparr = np.load(filename_UV)
    UVdisparr = load_file(filename_UV)
    Vx = UVdisparr["Vx"][:,:,:]; Vy = UVdisparr["Vy"][:,:,:]
    UV_t0 = np.moveaxis(np.dstack((Vx[0,:,:],Vy[0,:,:])),2,0)
    
    ## Get forms of TRT cells:
    if TRT_form:
        print("   *** Warning: This part of the code (plotting the circles) is completely hard-coded! ***")
        TRT_info_file = pd.read_pickle("/opt/users/jmz/0_training_NOSTRADAMUS_ANN/TRT_sampling_df_testset_enhanced.pkl")
        jCH_ls = TRT_info_file["jCH"].loc[(TRT_info_file["date"]==cfg_set["t0"]) & (TRT_info_file["RANKr"]>=10)].tolist()
        iCH_ls = TRT_info_file["iCH"].loc[(TRT_info_file["date"]==cfg_set["t0"]) & (TRT_info_file["RANKr"]>=10)].tolist()
                
        diameters = [16,24,32]
        circle_array = np.zeros((640,710,len(iCH_ls)))
        for ind in range(len(jCH_ls)):
            X, Y = np.meshgrid(np.arange(0, 710), np.arange(0, 640))
            for diameter in diameters:
                interior = ((X-jCH_ls[ind])**2 + (Y-iCH_ls[ind])**2) < (diameter/2)**2
                circle_array[interior,ind] += 1
    
    ## Get UV vectors if these were created:
    if cfg_set["UV_inter"]:
        #UV_vec_arr = np.load(filename_UV_vec)
        UV_vec_arr = load_file(filename_UV_vec)
        UV_vec = UV_vec_arr["UV_vec"]; UV_vec_sp = UV_vec_arr["UV_vec_sp"]
        #print(UV_vec)
        #print(len(UV_vec))
        #print(UV_vec_sp)
        #print(len(UV_vec_sp))
    #pdb.set_trace()
    ## Get time array
    t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=cfg_set["timestep"])
    
    ## Setup the plot:
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12.5, 6.5)
    plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
    plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
    
    ## Make plots
    #if animation: plt.clf()
    for i in range(vararr.shape[0]):
    
        ## Prepare UV field for quiver plot
        UV_field = np.moveaxis(np.dstack((Vx[i,:,:],Vy[i,:,:])),2,0)
        step = 40; X,Y = np.meshgrid(np.arange(UV_field.shape[2]),np.arange(UV_field.shape[1]))
        UV_ = UV_field[:, 0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]
        X_ = X[0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]; Y_ = Y[0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]
        
        ## Generate title:
        t_current = cfg_set["t0"] - t_delta[i]
        title_str = "%s fields at %s" % (var, t_current.strftime("%d.%m.%Y - %H:%M"))
        plt.suptitle(title_str); 
        
        """## Make plot:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12.5, 6.5)
        plt.suptitle(title_str); 
        plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
        plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
        for ax in axes:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(which='major', color='orange', linestyle='-', linewidth=0.5)
        """

        for ax in axes:
            ax.cla()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(which='major', color='orange', linestyle='-', linewidth=0.5)
        axes[0].set_title('Original')
        axes[0].imshow(vararr[i,:,:], aspect='equal', cmap=cmap)
        axes[0].quiver(X_, Y_, UV_[0,:,:], -UV_[1,:,:], pivot='tip', color='grey')
        if cfg_set["UV_inter"] and type(UV_vec[i]) is not float and len(UV_vec[i].shape)>1:
            axes[0].quiver(UV_vec[i][1,:,0], UV_vec[i][0,:,0], UV_vec[i][2,:,0], -UV_vec[i][3,:,0], pivot='tip', color='red')
            axes[0].quiver(UV_vec_sp[i][1,:], UV_vec_sp[i][0,:], UV_vec_sp[i][2,:], -UV_vec_sp[i][3,:], pivot='tip', color='blue')
        axes[1].set_title('Displaced')
        axes[1].imshow(vararr_disp[i,:,:], aspect='equal', cmap=cmap)
        if TRT_form:
            col_ls = ["r","b","g","m","k"]
            for circ in range(len(iCH_ls)):
                col = col_ls[circ%len(col_ls)]
                axes[1].contour(circle_array[:,:,circ],linewidths=0.3,alpha=0.7) #,levels=diameters)
        #axes[1].grid(which='major', color='red', linestyle='-', linewidth=1)

        fig.tight_layout()    
        
        if animation:
            #plt.show()
            plt.pause(.2)
            #plt.clf()
        else:        
            figname = "%scomparison_move_disp/%s%s/%s_%s%s_displacement.png" % (cfg_set["output_path"], var, resid_dir,
                                                                                t_current.strftime("%Y%m%d%H%M"), var, resid_suffix_plot)
            fig.savefig(figname, dpi=100)
    plt.close()
    
## Update fields for next time step:
def update_fields(cfg_set,verbose_time=False):
    """Update fields for next time step."""
    
    print("\nUpdate fields for time %s..." % cfg_set["t0"].strftime("%d.%m.%Y %H:%M"))
    t0_old = cfg_set["t0"] - cfg_set["time_change_factor"] * datetime.timedelta(minutes=cfg_set["timestep"])
    
    if verbose_time: t1 = datetime.datetime.now()
    
    ## Load files of different variables
    for var in cfg_set["var_list"]:
        filename_new = path_creator_vararr("orig",var,cfg_set)
        #filename_new = "%stmp/%s_%s_orig%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
        #                                          cfg_set["file_ext_verif"], cfg_set["save_type"])
        filename_old = path_creator_vararr("orig",var,cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
        #filename_old = "%stmp/%s_%s_orig%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"), var,
        #                                          cfg_set["file_ext_verif"], cfg_set["save_type"])
        
        bool_new_hour = cfg_set["t0"].hour==t0_old.hour
        #if var in ["Wind","Conv"] and bool_new_hour:
        #    vararr = get_vararr_t(cfg_set["t0"], var, cfg_set)
        #    #np.save(filename_new, vararr)
        #    save_file(filename_new, data_arr=vararr,
        #              var_name=var,cfg_set=cfg_set)
        #else:
        ## Load old array, move fields back in time and drop oldest field
        vararr = load_file(filename_old,var_name=var)
        vararr[1:,:,:] = np.copy(vararr)[:-1,:,:]
        
        ## Get field for new time step and assign to newest position
        vararr_t = get_vararr_t(cfg_set["t0"], var, cfg_set)
        vararr[0,:,:] = vararr_t[0,:,:]
        #np.save(filename_new, vararr)
        save_file(filename_new, data_arr=vararr,
                  var_name=var,cfg_set=cfg_set)
        if cfg_set["verbose"]: print("  ... "+var+" is updated")
        if cfg_set["delete_prec"]:
            os.remove(filename_old)
            if cfg_set["verbose"]: print("      and old _orig file removed")
    if verbose_time:  print("     Update _orig files: "+str(datetime.datetime.now()-t1)+"\n")

    
    ## Update disparr fields and displace variables with initial flow field:
    if verbose_time: t1 = datetime.datetime.now()
    update_disparr_fields(cfg_set, t0_old)
    if verbose_time:  print("    Update _disparr file: "+str(datetime.datetime.now()-t1)+"\n")

    if verbose_time: t1 = datetime.datetime.now()
    print("  Displace fields to new time step %s..." % cfg_set["t0"].strftime("%d.%m.%Y %H:%M"))
    displace_fields(cfg_set)
    if verbose_time:  print("    Displace fields: "+str(datetime.datetime.now()-t1)+"\n")
    
    ## Update disparr fields for residual movement and displace variables with
    ## residual movement flow field:
    if cfg_set["resid_disp"]:
        if verbose_time: t1 = datetime.datetime.now()
        update_disparr_fields(cfg_set, t0_old, resid=True)
        if verbose_time:  print("    Update _disparr file (resid): "+str(datetime.datetime.now()-t1)+"\n")

        if verbose_time: t1 = datetime.datetime.now()
        print("  Displace fields (resid) to new time step %s..." % cfg_set["t0"].strftime("%d.%m.%Y %H:%M"))
        displace_fields(cfg_set, resid=True)
        if verbose_time:  print("    Displace fields (resid): "+str(datetime.datetime.now()-t1)+"\n")
    
    ## Delete files associated with preceding time step:
    if verbose_time: t1 = datetime.datetime.now()
    if cfg_set["delete_prec"] and cfg_set["t0"]!=cfg_set["t0_orig"]:
        for var in cfg_set["var_list"]:
            print_extension = ""
            filename_old = path_creator_vararr("disp",var,cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
            #filename_old = "%stmp/%s_%s_disp%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"),
            #                                           var, cfg_set["file_ext_verif"], cfg_set["save_type"])
            os.remove(filename_old)
            if cfg_set["resid_disp"]:
                fileext_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
                filename_old_resid = path_creator_vararr("disp"+fileext_suffix,var,cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
                #filename_old_resid = "%stmp/%s_%s_disp%s%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"),
                #                                                   var, fileext_suffix, cfg_set["file_ext_verif"],
                #                                                   cfg_set["save_type"])
                os.remove(filename_old_resid)
                print_extension = " (including %s)" % fileext_suffix
        print("     and old _disp file removed"+print_extension)
    if verbose_time:  print("    Delete old files: "+str(datetime.datetime.now()-t1)+"\n")        
    
## Helper function of update_fields(cfg_set) updating motion fields:
def update_disparr_fields(cfg_set, t0_old, resid=False):
    """Helper function of update_fields(cfg_set) updating motion fields.
    
    Parameters
    ----------
    
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
    """

    ## Change suffixes of files to read and write in case residual movements are corrected:   
    if not resid:
        #UV_suffix = ""
        append_str = ""
    else:
        #UV_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        append_str    = " for residual movement" if not cfg_set["resid_disp_onestep"] else " for residual movement (combi)"
    
    print("  Calculate new displacement field%s (%s)..." % (append_str,cfg_set["t0"].strftime("%d.%m.%Y %H:%M")))
    resid_suffix = "resid" if resid else "standard"
    filename_old = path_creator_UV_disparr(resid_suffix,cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
    #filename_old = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"),
    #                                                  cfg_set["oflow_source"], resid_suffix, cfg_set["file_ext_verif"],
    #                                                  cfg_set["save_type"])
    
    ## Load old array, move fields back in time and drop oldest field
    #UVdisparr = np.load(filename_old)
    UVdisparr = load_file(filename_old)
    Vx = UVdisparr["Vx"][:,:,:]; Vy = UVdisparr["Vy"][:,:,:]
    Dx = UVdisparr["Dx"][:,:,:]; Dy = UVdisparr["Dy"][:,:,:]
    Vx[1:,:,:] = np.copy(Vx)[:-1,:,:]; Vy[1:,:,:] = np.copy(Vy)[:-1,:,:]
    Dx[1:,:,:] = np.copy(Dx)[:-1,:,:]; Dy[1:,:,:] = np.copy(Dy)[:-1,:,:]

    ## Get flow field for new time step and assign to newest position
    if cfg_set["UV_inter"]:
        D_new, UV_new, UV_vec_temp, UV_vec_sp_temp = calc_disparr(cfg_set["t0"], cfg_set, resid)
    else:
        D_new, UV_new = calc_disparr(cfg_set["t0"], cfg_set, resid)
    Vx[0,:,:] = UV_new[0,:,:]; Vy[0,:,:] = UV_new[1,:,:]
    Dx[0,:,:] =  D_new[0,:,:]; Dy[0,:,:] =  D_new[1,:,:]
    
    ## Save displacement field file
    filename_new = path_creator_UV_disparr(resid_suffix,cfg_set)
    #filename_new = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
    #                                                  cfg_set["oflow_source"], resid_suffix, cfg_set["file_ext_verif"],
    #                                                  cfg_set["save_type"])
    
    #np.savez(filename_new, Dx=Dx, Dy=Dy, Vx=Vx, Vy=Vy)
    save_file(filename_new, data_arr=[Dx,Dy,Vx,Vy],var_name=["Dx","Dy","Vx","Vy"],
              cfg_set=cfg_set)
    print("  ...UV and displacement arrays are updated"+append_str)
    
    ## Save combined displacement array (initial displacment + residual displacment):
    if cfg_set["resid_disp_onestep"] and resid:
        ## Load initial displacement field:
        filename_ini = path_creator_UV_disparr("standard",cfg_set)
        #filename_ini = "%stmp/%s_%s_disparr_UV%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                cfg_set["oflow_source"],cfg_set["file_ext_verif"], cfg_set["save_type"])
        UVdisparr_ini = load_file(filename_ini)
        
        ## Save summation of initial and residual displacment field
        filename_combi = path_creator_UV_disparr("resid_combi",cfg_set)
        #filename_combi = "%stmp/%s_%s_disparr_UV_resid_combi%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                              cfg_set["oflow_source"],cfg_set["file_ext_verif"], cfg_set["save_type"])
        #np.savez(filename_combi, Dx=Dx+UVdisparr_ini["Dx"], Dy=Dy+UVdisparr_ini["Dy"],
        #                         Vx=Vx+UVdisparr_ini["Vx"], Vy=Vy+UVdisparr_ini["Vy"])
        save_file(filename_combi, data_arr=[Dx+UVdisparr_ini["Dx"][:,:,:],Dy+UVdisparr_ini["Dy"][:,:,:],
                                            Vx+UVdisparr_ini["Vx"][:,:,:],Vy+UVdisparr_ini["Vy"][:,:,:]],
                  var_name=["Dx","Dy","Vx","Vy"],cfg_set=cfg_set)
        print("      & combined UV and displacement array is updated")
        
        ## Remove old disparr_UV_resid_combi file:
        filename_combi_old = path_creator_UV_disparr("resid_combi",cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
        #filename_combi_old = "%stmp/%s_%s_disparr_UV_resid_combi%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"),
        #                                                                  cfg_set["oflow_source"],cfg_set["file_ext_verif"], cfg_set["save_type"])
        
        if cfg_set["delete_prec"]:
            #if ("disparr" in filename_combi_old or "UV_vec" in filename_combi_old) and filename_combi_old[-4:]==".npy":
            #    filename_combi_old = filename_combi_old[:-4]+".npz"
            os.remove(filename_combi_old)
            print("     and old disparr_UV file"+append_str+" removed")    
    
    if cfg_set["delete_prec"]:
        #if ("disparr" in filename_old or "UV_vec" in filename_old) and filename_old[-4:]==".npy":
        #    filename_old = filename_old[:-4]+".npz"
        os.remove(filename_old)
        print("     and old disparr_UV file"+append_str+" removed")
    
        
# Calculate statistics for current variable:
def calc_statistics(cfg_set,var,vararr_disp):
    """Calculate skill scores for current displaced variable var.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable for which skill score is calculated
        
    vararr : numpy.array
        Array with the current observation
    """
    
    ## Import numpy array with preceding statistics results:
    filename_verif_stat = "%stmp/%s_%s_stat_verif.npy" % (cfg_set["root_path"],
                           cfg_set["verif_param"],
                           str(cfg_set[cfg_set["verif_param"]]))
    stat_array = np.load(filename_verif_stat)
    
    ## Create temporary statistics array:
    stat_array_temp = np.zeros((1,len(cfg_set["var_list"]),int(cfg_set["n_stat"])))*np.nan
    
    ## Get threshold for pixels with values
    value_threshold = 0. #cfg_set["R_threshold"] if var=="RZC" else 0.0
    
    ## Assign statistics to respective column:
    var_list = np.array(cfg_set["var_list"])
    ind_var = np.where(var_list==var)[0][0]
    
    ## Find pixels which are not NaN or zero
    bool_nonnan_nonzero = np.logical_and(np.isfinite(vararr_disp[0,:,:].flatten()),
                                      vararr_disp[0,:,:].flatten()>value_threshold)
    
    stat_array_temp[0,ind_var,0] = np.nanmin(vararr_disp[0,:,:])
    stat_array_temp[0,ind_var,1] = np.nanmax(vararr_disp[0,:,:])
    stat_array_temp[0,ind_var,2] = np.nanmean(vararr_disp[0,:,:].flatten()[np.where(bool_nonnan_nonzero)])
    stat_array_temp[0,ind_var,3] = np.nanpercentile(vararr_disp[0,:,:].flatten()[np.where(bool_nonnan_nonzero)],90)
    stat_array_temp[0,ind_var,4] = np.sum(bool_nonnan_nonzero)
          
    ## Save stats results again:
    if ind_var!=0:
        stat_array[-1,ind_var,:] = stat_array_temp[0,ind_var,:]
        print("      "+var+" statistics are saved")
    else:
        if stat_array.shape[0]==1 and np.all(stat_array < -9998): #np.all(np.isnan(stat_array)):
            stat_array = stat_array_temp
            print("      New stats array is created (with %s stats in first column)" % var)
        else:
            stat_array = np.concatenate((stat_array,stat_array_temp), axis=0)
            #if np.all(np.isnan(stat_array)): print("Still all nan"); print(stat_array)

    np.save(filename_verif_stat,stat_array)
    
    
# Calculate skill scores for current variable:
def calc_skill_scores(cfg_set,var,vararr_disp):
    """Calculate skill scores for current displaced variable var.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable for which skill score is calculated
        
    vararr_disp : numpy.array
        Array where the current observation is at the level 0 of the first dimension
    """
    
    ## Import numpy array with preceding verification results:
    filename_verif = "%stmp/%s_%s_%s_verif.npy" % (cfg_set["root_path"],
                      cfg_set["verif_param"], str(cfg_set[cfg_set["verif_param"]]),var)
    verif_array = np.load(filename_verif)
    ## Copy new layer (if its not the first verification, then just overwrite the NaNs)
    #if (np.isnan(verif_array)).all:
    #    verif_array_temp = verif_array
    #else:
    verif_array_temp = np.zeros((1,len(cfg_set["scores_list"]),
                                cfg_set["n_integ"]-1))*np.nan
    
    ## Define threshold for categorical forecast:
    if cfg_set["R_thresh_meth"] == "fix":
        R_tresh = cfg_set["R_threshold"]
    elif cfg_set["R_thresh_meth"] == "perc":
        R_tresh = 0.1
    #    R_tresh = np.min([np.nanpercentile(oflow_source_data[0,:,:],cfg_set["R_threshold"]),
    #                      np.nanpercentile(oflow_source_data[1,:,:],cfg_set["R_threshold"])])
    threshold = R_tresh if var=="RZC" else 0.0

    ## Loop through integration steps:
    #t1 = datetime.datetime.now()
    for t_step in range(1,cfg_set["n_integ"]):
        verif_array_temp[0,:,t_step-1] = st.vf.scores_det_cat_fcst(vararr_disp[t_step,:,:],
                                                                   vararr_disp[0,:,:],
                                                                   threshold,
                                                                   cfg_set["scores_list"])
                                                                 
    #t2 = datetime.datetime.now()
    #test_arr = np.apply_over_axes(st.vf.scores_det_cat_fcst,[1,2],vararr_disp[1:,:,:],
    #                               obs=vararr_disp[0,:,:],thr=threshold,scores=cfg_set["scores_list"])
    #print("         Elapsed time for for-loop calculating scores: "+str(t2-t1))
    
    #if np.array_equal(test_arr[1:,:,:],verif_array_temp[:,:,:]):
    #    print("np.array_equal successfull")
    #else: print("np.array_equal NOT successfull"); print(test_arr); print(verif_array_temp[-1,:,:])
    #print(verif_array_temp)
    
    ## Save verification results again:
    if verif_array.shape[0]==1 and np.all(verif_array < -9998): #np.all(np.isnan(verif_array)):
        verif_array = verif_array_temp
        print("      New "+var+" skill score array is created")
    else:
        verif_array = np.concatenate((verif_array,verif_array_temp), axis=0)
    #if np.all(np.isnan(verif_array)): print("Still all nan"); print(verif_array)
    
    np.save(filename_verif,verif_array)
    print("      "+var+" skill score array is saved")
    
    # Do procedure only in case some interesting variable is actually detected? But then, what else could be skipped?
    # Have array where time information is collected, when displacement was done, and skill scores were calculated.

## Analyse skill scores:
def analyse_skillscores(cfg_set,var):
    """Analyse skill scores.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable whose skill scores should be analysed
    """
    
    ## Read in skill scores of the respective variable:
    filename_verif = "%stmp/%s_%s_%s_verif.npy" % (cfg_set["root_path"],
                      cfg_set["verif_param"], str(cfg_set[cfg_set["verif_param"]]),var)
    verif_array = np.load(filename_verif)
    
    ## Read in statistics:
    filename_verif_stat = "%stmp/%s_%s_stat_verif.npy" % (cfg_set["root_path"],
                           cfg_set["verif_param"],
                           str(cfg_set[cfg_set["verif_param"]]))
    stat_array = np.load(filename_verif_stat)
    
    ## verif_array & stat_array:
    ## Dim 1 -> Zeit
    
    ## verif_array:
    ## Dim 2 -> Skill score (csi,hk,sedi)
    ## Dim 3 -> Lead time (5 - 45min)
    
    ## stat_array:
    ## Dim 2 -> Variable (RZC,BZC,LZC,MZC,EZC,THX)
    ## Dim 3 -> Statistic:
    ## - Min
    ## - Max
    ## - Mean
    ## - 90% quantile
    ## - Number of Pixels
    
    ## Make plot of decrease in skill with increasing lead-time (HK score):
    plt.clf()
    fig, ax = plt.subplots()
    plt.grid(True)
    bp = plt.boxplot(verif_array[:,1,:],notch=True,patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgrey')
   
    plt.title("HK score as function of Lead time\nVariable: %s" % var)
    plt.xlabel("Lead time [min]")
    plt.ylabel("Hanssen-Kuipers Discriminant")
    ax.set_xticklabels(np.arange(5,45,5))
    #plt.show()
    #plt.figure(figsize=(2,2))
    filename = "%splot_verif/HKvsLeadT_%s_%s_%s.pdf" % (cfg_set["output_path"],cfg_set["verif_param"],
                                                          str(cfg_set[cfg_set["verif_param"]]),var)
    plt.savefig(filename)
    
    ## Make plot of skill at lead time = 5min as function of 
    ## average rain-rate (used for optical flow):
    plt.clf()
    fig, ax = plt.subplots()
    plt.plot(stat_array[:,0,2], verif_array[:,1,0], 'bo')
    plt.title("HK score (lead time = 5min) as function of mean rain rate\nVariable: %s" % var)
    plt.xlabel("Rain rate [mm/h]")
    plt.ylabel("Hanssen-Kuipers Discriminant")
    plt.grid(True)
    #plt.show()
    #plt.figure(figsize=(2,2))
    filename = "%splot_verif/HKvsRR_%s_%s_%s.pdf" % (cfg_set["output_path"],cfg_set["verif_param"],
                                                          str(cfg_set[cfg_set["verif_param"]]),var)
    plt.savefig(filename)
    
## Compare skill scores:
def compare_skillscores_help(cfg_set,var,verif_param_ls):
    """Analyse skill scores.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable whose skill scores should be analysed
        
    verif_param_ls : list
        List with additional verification parameter.
    """
    
    ## Read in statistics:
    filename_verif_stat = "%stmp/%s_%s_stat_verif.npy" % (cfg_set["root_path"],
                           cfg_set["verif_param"],
                           str(cfg_set[cfg_set["verif_param"]]))
    stat_array = np.load(filename_verif_stat)
    
    ## Read in skill scores of the respective variable:
    verif_array_ls = []
    time_dim_ls = []
    for verif_param in verif_param_ls:
        #filename = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/threshold_verif/180527_12_20/%s_%s_%s_verif.npy" % (cfg_set["verif_param"],verif_param, var)
        filename = "%stmp/%s_%s_%s_verif.npy" % (cfg_set["root_path"],cfg_set["verif_param"],verif_param,var)
        verif_array_ls.append(np.load(filename))
        time_dim_ls.append(verif_array_ls[-1].shape[0])
        
    ## Check for time dimension (if not equal, set to minimum length):
    min_time_dim = np.nanmin(time_dim_ls)
    if any(x.shape[0]!=min_time_dim for x in verif_array_ls):
        print("   *** Warning: Verification arrays are not of the same length! ***")
        for i in range(len(verif_array_ls)):
            verif_array_ls[i] = verif_array_ls[i][:min_time_dim,:,:]
    
    ## Concatenate to one big file (along new, first dimension):
    verif_array_con = np.stack(verif_array_ls)
    
    """
    filename_verif_01 = "%stmp/%s_0.1_%s_verif.npy" % (cfg_set["root_path"],
                         cfg_set["verif_param"], var)
    filename_verif_05 = "%stmp/%s_0.5_%s_verif.npy" % (cfg_set["root_path"],
                         cfg_set["verif_param"], var)
    filename_verif_10 = "%stmp/%s_1.0_%s_verif.npy" % (cfg_set["root_path"],
                         cfg_set["verif_param"], var)
    verif_array_01 = np.load(filename_verif_01)
    verif_array_05 = np.load(filename_verif_05)
    verif_array_10 = np.load(filename_verif_10)
    
    ## Concatenate to minimum time covered by all verification datasets:
    min_time_dim = np.nanmin([verif_array_01.shape[0],verif_array_05.shape[0],verif_array_10.shape[0]])
    verif_array_con = np.stack([verif_array_01[:min_time_dim,:,:],
                                verif_array_05[:min_time_dim,:,:],
                                verif_array_10[:min_time_dim,:,:]])
    """
    
    ## Plot skill score comparison:
    compare_skillscores(cfg_set,var,verif_array_con,stat_array,verif_param_ls)
    
## Compare skill scores:
def compare_skillscores(cfg_set,var,verif_array_con,stat_array,verif_param_ls): #,verif_param_ls):
    """Analyse skill scores.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable whose skill scores should be analysed
        
    verif_param_ls : list
        List with additional verification parameter.
    """

    from cycler import cycler
    reduced_ss_set = False #True
    if reduced_ss_set:
        verif_array_con = verif_array_con[:,:,[0,2],:]
        list_ss_names = ["Critical Success\nIndex CSI","Symmetric Extremal\nDependence Index SEDI"]
    
    legend_title = "<Insert title here>"
    if cfg_set["verif_param"]=="R_threshold":
        legend_unit = "mm/h" if verif_param_ls[i_verif_samp] < 50. else "%"
        legend_title = "%s Threshold =" % cfg_set["abbrev_dict"][cfg_set["oflow_source"]] #"RZC Threshold ="
    elif cfg_set["verif_param"]=="resid_method":
        legend_unit = ""
        legend_title = "Method to reduce\nresidual movement:"
    else: 
        legend_unit = "<Insert Unit>"
        legend_title = "<Insert title>"
    
    ## Define some basic variables:
    n_verif_samp  = verif_array_con.shape[0]
    n_skillscores = verif_array_con.shape[2]
    n_leadtimes   = verif_array_con.shape[3]
    
    x_val = np.arange(cfg_set["timestep"],
                      cfg_set["timestep"]*(verif_array_con.shape[3]+1),
                      cfg_set["timestep"])
        
    ## Initialise plot:
    plt.clf()
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'm', 'k'])))
    fig, axs = plt.subplots(n_skillscores, sharex=True, figsize=(8,7))
    #axs[-1].set_xticklabels(np.arange(5,45,5))
    
    plot_array = np.zeros([verif_array_con.shape[0],verif_array_con.shape[0]])
       
    for i_skillscore in np.arange(n_skillscores):
        ## Initialise array for median and IQR (3rd dim -> Median and IQR)
        plot_array = np.zeros([n_verif_samp,n_leadtimes,3])*np.nan
        
        ## Calculate median and IQR:
        for i_verif_samp in np.arange(n_verif_samp):
            for i_leadtimes in np.arange(n_leadtimes):
                plot_array[i_verif_samp,i_leadtimes,0] = np.nanmedian(verif_array_con[i_verif_samp,
                                                                                      :,
                                                                                      i_skillscore,
                                                                                      i_leadtimes])
                                                                                   
                plot_array[i_verif_samp,i_leadtimes,1] = np.nanpercentile(verif_array_con[i_verif_samp,
                                                                                          :,
                                                                                          i_skillscore,
                                                                                          i_leadtimes], 
                                                                          25)
        
                plot_array[i_verif_samp,i_leadtimes,2] = np.nanpercentile(verif_array_con[i_verif_samp,
                                                                                          :,
                                                                                          i_skillscore,
                                                                                          i_leadtimes], 
                                                                          75)
                
        col_list = ["#17becf","#1f77b4","#bcbd22","#ff7f0e","#d62728"]
        
        ## Plot IQR:
        for i_verif_samp in np.arange(n_verif_samp):
            axs[i_skillscore].fill_between(x_val,
                                           plot_array[i_verif_samp,:,1],
                                           plot_array[i_verif_samp,:,2],
                                           alpha=0.1, facecolor=col_list[i_verif_samp]) # facecolor='green',
        
        ## Plot Median:
        ylab = cfg_set["abbrev_dict"][cfg_set["scores_list"][i_skillscore]] if not reduced_ss_set else list_ss_names[i_skillscore]
        
        for i_verif_samp in np.arange(n_verif_samp):
            legend_entry = "%s %s" % (verif_param_ls[i_verif_samp], legend_unit)
            axs[i_skillscore].plot(x_val, plot_array[i_verif_samp,:,0],marker='o',linestyle='solid', label=legend_entry, color=col_list[i_verif_samp]) # facecolor='green',
            axs[i_skillscore].set(ylabel=ylab)
            axs[i_skillscore].grid(True)
            axs[i_skillscore].set_xlim((x_val[0]-1,x_val[-1]+1))
            axs[i_skillscore].set_ylim((0,1))
            
        ## Insert title and legend:
        if i_skillscore==0:
            axs[i_skillscore].set(title="Skill scores for variable %s" % cfg_set["abbrev_dict"][var]) #var)            
            axs[i_skillscore].legend(loc='upper right',title=legend_title,fontsize="small",
                                     ncol=3) #loc='lower center', bbox_to_anchor=(0, -0.5))
        
        ## Insert y-label:
        if i_skillscore==np.arange(n_skillscores)[-1]:
            axs[i_skillscore].set(xlabel="Lead time [min]")
            
    plt.tight_layout()
    
    ## Save file:
    filename = "%splot_verif/SkillScores_%s_%s.pdf" % (cfg_set["output_path"],cfg_set["verif_param"],var)
    plt.savefig(filename)
    #fig.delaxes(axs[1])
    #plt.draw()
    #plt.show()
    
    #plt.clf()
    #fig2, axs_2 = plt.subplots(2, sharex=True, figsize=(7,8))
    #axs_2[0] = axs[0]
    #axs_2[1] = axs[2]
    #fig2.show()
        
    
## Save variable array (vararr) or displacement array (disparr) as Numpy or NetCDF file:
def save_file(output_file_path_str, data_arr,
              var_name=None, t0_datetime=None, filetype=None,
              cfg_set=None, var_type=None, var_unit=None, longname=None, dt=None):
    """Save variable and displacement arrays into files.

    Parameters
    ----------
        
    output_file_path_str : str
        File path defining location of .nc file to be written.
        
    data_arr : (list of) numpy arrays
        (List of) numpy arrays containing the data to be saved.
        
    var_name : (list of) strings
        (List of) strings stating the names of the datasets.
        
    var_type : (list of) numpy datatypes (e.g. <type 'numpy.int8'>)
        (List of) numpy datatypes stating the datatype of the datasets.
        If only one type is provided, it is assumed to apply for all arrays.
        
    var_unit : (list of) units related to the respective array data.
        If only one type is provided, it is assumed to apply for all arrays.
        
    longname : (list of) describing longnames related to the respective array data.
        If only one type is provided, it is assumed to apply for all arrays.
        
    t0_datetime : (list of) datetime objects
        (List of) datetime objects stating the time steps of the data
        (along the first dimension, starting with the most current observation).
        
    dt : int
        Number of minutes between different layers in first dimension of vararr
    """
    
    if filetype!="npy":
        ## Check arguments needed when potentially creating a NetCDF file.
        if (t0_datetime is None or filetype is None or
            cfg_set is None or var_type is None) and cfg_set is None:
            print("either the non-compulsory arguments are provided or "+
                   "a cfg_set dictionary has to be provided")
        ## Set filetype to the one stated in cfg_set (if not provided)
        if filetype is None: filetype = cfg_set["save_type"]

    ## Save numpy file (npy/npz):
    if filetype == "npy":
        if "disparr" not in output_file_path_str and type(data_arr) is not list:
            np.save(output_file_path_str, data_arr)
        elif "disparr" in output_file_path_str and len(data_arr)==4:
            output_file_path_str = output_file_path_str[:-1]+"z"
            if var_name!=["Dx","Dy","Vx","Vy"]: raise ValueError('Ordering must be "Dx","Dy","Vx","Vy"')
            np.savez(output_file_path_str, Dx=data_arr[0], Dy=data_arr[1],
                                           Vx=data_arr[2], Vy=data_arr[3])
        elif "disparr" in output_file_path_str and len(data_arr)==2:
            output_file_path_str = output_file_path_str[:-1]+"z"
            if var_name!=["UV_vec","UV_vec_sp"]: raise ValueError('Ordering must be "UV_vec","UV_vec_sp"')
            np.savez(output_file_path_str, UV_vec=data_arr[0], UV_vec_sp=data_arr[1])
        else: raise ValueError("saving procedure for list of arrays into npz file not yet implemented")
    
    ## Save NetCDF file (nc)
    elif filetype == "nc":
    
        if t0_datetime is None: t0_datetime = cfg_set["t0"]
        if dt is None:
            dt = cfg_set["time_change_factor"]*cfg_set["timestep"]
        
        ## Read auxilary data from cfg_set file:
        if var_name==["Dx","Dy","Vx","Vy"] or var_name==["UV_vec","UV_vec_sp"]:
            var_unit = ["Pixel "+str(dt)+"min-1","Pixel "+str(dt)+"min-1",\
                        "km "+str(dt)+"min-1","km "+str(dt)+"min-1"]
            var_type = np.float32
            longname = ["Displacement eastward","Displacement northward",\
                        "Optical flow eastward","Optical flow northward"]
        else:
            if var_type is None and type(var_name) is not list:
                var_type = cfg_set["type_dict"][var_name]
            if var_unit is None and type(var_name) is not list:
                var_unit = cfg_set["unit_dict"][var_name]
            if longname is None and type(var_name) is not list:
                longname = cfg_set["abbrev_dict"][var_name]
            
        ## Further checks whether all the necessary data is provided:
        if var_type is None and cfg_set is None:
            raise ValueError("either a variable type (var_type) or "+
                             "a cfg_set dictionary has to be provided")
        if var_type is None and var_name not in cfg_set["var_list"]:
            raise ValueError("variable name (var_name) not found in cfg_set dictionary")
        if var_unit is None and cfg_set is None:
            raise ValueError("either a variable unit (var_unit) or "+
                             "a cfg_set dictionary has to be provided")
        if var_unit is None and var_name not in cfg_set["var_list"]:
            raise ValueError("variable name (var_name) not found in cfg_set dictionary")
        
        ## Make description for different datasets:
        var_descib = var_name
        if "_orig" in output_file_path_str:
            description_nc = "Original observation of "
        elif "_disp_resid_combi" in output_file_path_str:
            description_nc = "Displaced observation (with residual movement correction with one displacement) of "
        elif "_disp_resid" in output_file_path_str:
            description_nc = "Displaced observation (with residual movement correction with one displacement) of "
        elif "_disparr_UV_resid_combi" in output_file_path_str:
            description_nc = "Displacement field (with residual movement)"
            var_descib = ""
        elif "_disparr_UV_resid" in output_file_path_str:
            description_nc = "Residual displacement field"
            var_descib = ""
        elif "_disparr_UV" in output_file_path_str:
            description_nc = "Displacement field"
            var_descib = ""
        elif "_disp" in output_file_path_str:
            description_nc = "Displaced observation of "
        else:
            print("  *** Warning: No description added to NetCDF file ***")
        
        description = description_nc+var_descib
        
        ## Save as NetCDF file:
        save_nc(output_file_path_str,data_arr,var_name,var_type,var_unit,longname,
                t0_datetime,description,dt=dt)
    else: raise ValueError("filetype must either be npy or nc.")    
  
 
## Read variable array (vararr) from Numpy or NetCDF file:
def load_file(input_file_path_str, var_name=None):
    """Read variable and displacement arrays into files.

    Parameters
    ----------
        
    input_file_path_str : str
        File path defining location of .nc file to be read.
        
    var_name : (list of) strings
        (List of) strings stating the names of the datasets.
        
    """
    
    ## Adjust for disparr files (which should be understood as .npz files):
    if ("disparr" in input_file_path_str or "UV_vec" in input_file_path_str) and input_file_path_str[-4:]==".npy":
        input_file_path_str = input_file_path_str[:-4]+".npz"
    
    ## Analyse file ending:
    if input_file_path_str[-3:]==".nc":
        #if var_name==["Dx","Dy","Vx","Vy"] or var_name==["UV_vec","UV_vec_sp"]:
        #    data_arr = read_nc(input_file_path_str,var_name)
        #    return data_arr
        #if var_name is None:
        #    raise ValueError("variable name necessary to read NetCDF file.")
        if var_name is None:
            print("   *** Warning: Returning opened NetCDF file without closing ***")
            nc_file = read_nc(input_file_path_str,var_name)
            return nc_file
        elif type(var_name) is not list:
            data_arr = read_nc(input_file_path_str,var_name)
            return data_arr
        else:
            data_arr_ls = []
            for var in var_name:
                data_arr_ls.append(read_nc(input_file_path_str,var))
            return data_arr_ls
    elif input_file_path_str[-4:]==".npy":
        if type(var_name) is not list:
            data_arr = np.load(input_file_path_str)
            return(data_arr)
        else:
            raise ValueError("only one variable saved in .npy file.")
    elif input_file_path_str[-4:]==".npz":
        if var_name is None:
            #raise ValueError("several variable names needed to extract arrays from .npz file.")
            data_arr = np.load(input_file_path_str)
            return data_arr
        elif type(var_name) is list:
            data_arr = np.load(input_file_path_str)
            data_arr_ls = []
            for var in var_name:
                data_arr_ls.append(data_arr[var])
            return data_arr_ls
        else:
            data_arr = np.load(input_file_path_str)[var_name]
            return data_arr

## Save variable array (data_arr) as NetCDF file:
def save_nc(output_file_path_str,data_arr,var_name,var_type,var_unit,longname,
            datetime_object,description,dt=None,verbose=False):
    """Save variable array (data_arr) as NetCDF file.

    Parameters
    ----------
        
    output_file_path_str : str
        File path defining location of .nc file to be written.
        
    data_arr : (list of) numpy arrays
        (List of) numpy arrays containing the data to be saved.
        
    var_name : (list of) strings
        (List of) strings stating the names of the datasets.
        
    var_type : (list of) numpy datatypes (e.g. <type 'numpy.int8'>)
        (List of) numpy datatypes stating the datatype of the datasets.
        If only one type is provided, it is assumed to apply for all arrays.
        
    var_unit : (list of) units related to the respective array data.
        If only one type is provided, it is assumed to apply for all arrays.
        
    longname : (list of) describing longnames related to the respective array data.
        If only one type is provided, it is assumed to apply for all arrays.
        
    datetime_object : (list of) datetime objects
        (List of) datetime objects stating the time steps of the data
        (along the first dimension, starting with the most current observation).
        
    dt : int
        Number of minutes between different layers in first dimension of vararr
    
    """
    
    ## Put input data into lists (if not provided as list):
    if type(data_arr)        is not list: data_arr = [data_arr]
    if type(var_name)        is not list: var_name = [var_name]
    if type(var_type)        is not list: var_type = [var_type]
    if type(var_unit)        is not list: var_unit = [var_unit]
    if type(longname)        is not list: longname = [longname]
    if type(datetime_object) is not list: datetime_object = [datetime_object]
    
    ## Check length of input data:
    if len(set(map(np.shape, data_arr))) > 1:
        raise ValueError('variable arrays are not of the same shape (%s)' %
                          map(np.shape, data_arr))
    if len(datetime_object)!=1 and len(datetime_object)!=data_arr[0].shape[0]:
        raise ValueError('length of datetime object (%s) is unequal '+
                          'to one or the first dimension of the variable array (%s)' %
                          (len(datetime_object),data_arr[0].shape[0]))
    elif len(var_name)!=len(data_arr):
        raise ValueError('length of var_name list (%s) is unequal '+
                          'to the number of variable arrays (%s)' %
                          (len(var_name),len(data_arr)))
    #elif len(var_name)!=1 and len(var_name)!=len(data_arr):
    #    raise ValueError('length of var_name list (%s) is unequal '+
    #                      'to one or the first dimension of the variable array (%s)' %
    #                      (len(var_name),len(data_arr)))
    elif len(var_type)!=1 and len(var_type)!=len(data_arr):
        raise ValueError('length of var_type list (%s) is unequal '+
                          'to one or the first dimension of the variable array (%s)' %
                          (len(var_name),len(data_arr)))
    elif len(var_unit)!=1 and len(var_unit)!=len(data_arr):
        raise ValueError('length of var_unit list (%s) is unequal '+
                          'to one or the first dimension of the variable array (%s)' %
                          (len(var_unit),len(data_arr)))
    elif len(longname)!=1 and len(longname)!=len(data_arr):
        raise ValueError('length of longname list (%s) is unequal '+
                          'to one or the first dimension of the variable array (%s)' %
                          (len(longname),len(data_arr)))
    if len(datetime_object)==1 and len(datetime_object)!=len(data_arr) and dt is None:
        raise ValueError('length of time step has to be provided')
    
    
    ## Create NetCDF file:
    from netCDF4 import Dataset, num2date, date2num
    dataset = Dataset(output_file_path_str,
                      'w', format='NETCDF4_CLASSIC')
    dataset.history = 'Created ' + datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    dataset.description = description
                      
    ## Dimension creation:
    x    = dataset.createDimension('x', data_arr[0].shape[2])
    y    = dataset.createDimension('y', data_arr[0].shape[1])
    time = dataset.createDimension('time', None) # data_arr.shape[0])
    
    ## Auxilary variable creation:
    x_axis = dataset.createVariable('x', np.float32, ('x',))
    y_axis = dataset.createVariable('y', np.float32, ('y',))
    times  = dataset.createVariable('time', np.int16, ('time',)) # u8 or i8 does not work...
    times.calendar = 'standard'
    times.units = 'minutes since %s' % datetime_object[0].strftime("%Y-%m-%d %H:%M:%S")
    #times.units = 'seconds since 1970-01-01 00:00:00.0'
    
    ## Create time stamps variable:
    if len(datetime_object)==1:
        datetime_list = datetime_object - np.arange(data_arr[0].shape[0])*datetime.timedelta(minutes=dt)
    else: datetime_list = datetime_object
    times[:] = date2num(datetime_list,units=times.units)
    
    ## Create spatial coordinate variable:
    y_axis.units = 'Swiss northing CH1903 [km]'
    x_axis.units = 'Swiss easting CH1903 [km]'
    x_axis[:]    = np.arange(255,965)+0.5
    y_axis[::-1] = np.arange(-160,480)+0.5
    
    ## Data variable creation:
    var_name_list = var_name #if len(var_name)==1 else var_name*len(data_arr)
    var_type_list = var_type if len(var_name)==1 else var_type*len(data_arr)

    ## Write data into variables:
    id_var_list=[]
    for i in range(len(data_arr)):
        #if "int" in var_type_list[i]:
        #    id_var_list.append(dataset.createVariable(var_name_list[i],var_type_list[i],
        #                                              ('time','y','x'),zlib=True))
        #else:
        id_var_list.append(dataset.createVariable(var_name_list[i],var_type_list[i],
                                                 ('time','y','x'),zlib=True,
                                                 least_significant_digit=3)) #,least_significant_digit=2
        id_var_list[i].setncatts({'long_name': longname[i],'units': var_unit[i]})
        id_var_list[i][:,:,:] = data_arr[i]

    ## Close file:
    dataset.close()
    if verbose: print("   Written NetCDF file for: %s (%s)" %
                      (description,datetime_object[0].strftime("%d.%m.%y %H:%M")))
    
    
## Read variable array (data_arr) as NetCDF file:
def read_nc(input_file_path_str,var_name=None):
    """Read variable array (data_arr) as NetCDF file.

    Parameters
    ----------
        
    input_file_path_str : str
        File path defining location of .nc file to be read.
        
    var_name : (list of) strings
        (List of) strings stating the names of the datasets.
    
    Returns
    -------
    
    varrarr : (list of) numpy arrays
        (List of) numpy arrays containing the data to be read.

    """
    if var_name is None:
        nc_file = Dataset(input_file_path_str,'r')
        return nc_file
    else: 
        nc_file = Dataset(input_file_path_str,'r')
        file = nc_file.variables[var_name][:,:,:]
        nc_file.close()
        return(file)
    
    
    
    
    
    
    
