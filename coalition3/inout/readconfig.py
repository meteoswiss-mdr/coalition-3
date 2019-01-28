""" [COALITION3] Reading config information"""

from __future__ import division
from __future__ import print_function

import os
import configparser
import datetime as dt
import numpy as np
import pandas as pd

## =============================================================================
## FUNCTIONS:

## Get settings from config file and some additional static settings:
def get_config_info_tds(coalition3_path=None, CONFIG_PATH=None):
    #(CONFIG_PATH_set_train,CONFIG_FILE_set_train):
    """Get information from configuration file containing config
    information for the creation of the training dataset and
    return in a dictionary "cfg_set_tds".


    Parameters
    ----------

    coalition3_path : str
        Root path to the project (default: git repo)
        
    CONFIG_PATH : str
        Path to config file.
    
    ## (DEPRECATED! Names are now hard-coded)
    #CONFIG_PATH : str
    #    Path to config file.
    #    
    #CONFIG_FILE_set : str
    #    Name of settings config file.
        
    Output
    ------

    cfg_set_tds : dict
        Dictionary with the basic variables used throughout the code
     
    """
    
    ## ===== Get path to config files: =========================
    if coalition3_path is None:
        coalition3_path = os.path.abspath(os.path.join(
                                os.path.dirname(__file__),'../../'))
    if CONFIG_PATH is None:
        CONFIG_PATH = os.path.join(coalition3_path,u'config/')
    cfg_set_tds = {"CONFIG_PATH": CONFIG_PATH}
                                
    ## ===== Read the data source/output configuration file: ============
    config = configparser.RawConfigParser()
    config.read(os.path.join(CONFIG_PATH,u"training_dataset.cfg"))

    ## Add source paths:
    config_ds = config["datasource"]
    if config_ds["root_path"]=="":
        root_path_tds = coalition3_path
    else:
        root_path = config_ds["root_path"]
    
    cfg_set_tds.update({
        "root_path_tds":      root_path_tds,
        "PATH_stat_output":   config_ds["PATH_stat_output"],
        "PATH_stdout_output": config_ds["PATH_stdout_output"]
    })

    ## Read further config information on the training dataset
    config_bs = config["basicsetting"]
    cfg_set_tds.update({
        "tds_period_start":     datetime.datetime.strptime(config_bs["tds_period_start"], "%Y%m%d").date(),
        "tds_period_end":       datetime.datetime.strptime(config_bs["tds_period_end"], "%Y%m%d").date(),
        "dt_samples":           int(config_bs["dt_samples"]),
        "dt_daily_shift":       int(config_bs["dt_daily_shift"]),
        "tds_period_start_doy": tds_period_start.timetuple().tm_yday,
        "tds_period_end_doy":   tds_period_end.timetuple().tm_yday,
    })
    
    return(cfg_set_tds)  
    
## Print information before running script:
def print_config_info_tds(cfg_set_tds,CONFIG_FILE_set):
    """Print information before running script"""
    print("\n-------------------------------------------------------------------------------------------------------\n")
    print_str = '    Configuration of COALITION3 training dataset preparation procedure:'+ \
    '\n      Date range:       '+cfg_set_tds["tds_period_start"].strftime("%Y-%m-%d")+ \
        ' to '+cfg_set_tds["tds_period_end"].strftime("%Y-%m-%d")+ \
    '\n      dt of samples:    '+str(cfg_set_tds["dt_samples"])+"min"+ \
    '\n      dt change:        '+str(cfg_set_tds["dt_daily_shift"])+"min per day \n"
    #'\n      Config files:     '+CONFIG_FILE_set+' (Settings)'+ \
    print(print_str)
    print("-------------------------------------------------------------------------------------------------------\n")
    
    
## Get settings from config file and some additional static settings:
def get_config_info_op(coalition3_path=None, CONFIG_PATH=None): #**kwargs):
    """Get information from configuration file and return in a dictionary
    "cfg_set". Furthermore, get data from variable configuration file
    (cfg_var.csv) and return in a pandas.dataframe "cfg_var"

    Parameters
    ----------

    coalition3_path : str
        Root path to the project (default: git repo)
        
    CONFIG_PATH : str
        Path to config file.

    ## (DEPRECATED! Names are now hard-coded)
    # CONFIG_FILE_set : str
    #     Name of settings config file.

    # CONFIG_FILE_var : str
    #     Name of variable config file (.csv).

    Output
    ------

    cfg_set : dict
        Dictionary with the basic settings used throughout the code.

    cfg_var : dict
        Dictionary with information on the variables to be read.

    cfg_var_combi : dict
        Dictionary with information on the combined variables (SEVIRI channels)
        to be read.

    Dictionary elements
    -------------------

    See Manual.docx
    """

    ## ===== Get path to config files: =========================
    if coalition3_path is None:
        coalition3_path = os.path.abspath(os.path.join(
                                os.path.dirname(__file__),'../../'))

    #CONFIG_PATH,CONFIG_FILE_set,CONFIG_PATH_set, \
    #                        CONFIG_FILE_var,CONFIG_FILE_var_combi
    #if "CONFIG_PATH" in kwargs.keys():
    #    CONFIG_PATH = kwargs["CONFIG_PATH"]
    #else:
    if CONFIG_PATH is None:
        CONFIG_PATH = os.path.join(coalition3_path,u'config/')
    cfg_set = {"CONFIG_PATH": CONFIG_PATH}

    ## ===== Import configuration information on variables: =============
    dtype_cfg_var = {'N_CATEGORIES': 'uint8', 'MIN_VALUE': 'float32'}
    dtype_cfg_var.update(dict.fromkeys(['PREDICTOR', 'LAGRANGIAN', 'PAST_OBS', \
                                        'SMOOTH', 'SUM', 'MEAN', 'STDDEV','IQR', \
                                        'PERC01', 'PERC05', 'PERC30','PERC50', 'PERC80', \
                                        'PERC95','PERC99', 'MIN', 'MAX', \
                                        'PC_NONNAN', 'PC_NONZERO', 'PC_LT55','PC_CAT', \
                                        'READ','READ_IF','VARIABILITY'], 'bool'))
    dtype_cfg_var_combi = {'dt': 'bool','PREDICTOR': 'bool'}
    cfg_var       = pd.read_csv(os.path.join(CONFIG_PATH,'cfg_var.csv'),
                                delimiter=";",dtype=dtype_cfg_var)
    cfg_var_combi = pd.read_csv(os.path.join(CONFIG_PATH,'cfg_var_combi.csv'),
                                delimiter=";",dtype=dtype_cfg_var_combi)
    
    ## Read further config information, which variables should be displaced
    #var_list        = config_vr["var_list"].split(',') ## DEPRECATED!
    var_list       = cfg_var["VARIABLE"][cfg_var["READ_IF"]].tolist()
    var_combi_list = cfg_var_combi["VARIABLE"][cfg_var_combi["PREDICTOR"]].tolist()
    cfg_set.update({
        "var_list":       var_list,
        "var_combi_list": var_combi_list
    })

    ## ===== Make static settings: =========================
    cfg_set.update({
        "adv_method":          "semilagrangian",
        "square_domain":       False,
        "colorscale":          "MeteoSwiss",  # MeteoSwiss or STEPS-BE
        "motion_plot":         "quiver",      # streamplot or quiver
    })
    
    ## ===== Read the general settings: ============
    config = configparser.RawConfigParser()
    config.read(os.path.join(CONFIG_PATH,u"general_settings.cfg"))

    ## Add source paths:
    config_sp = config["source_paths"]
    if config_sp["root_path"]=="":
        root_path = coalition3_path
    else:
        root_path = config_sp["root_path"]
    if config_sp["fig_output_path"]=="":
        fig_output_path = os.path.join(coalition3_path,u"figures/")
    else:
        fig_output_path = config_sp["fig_output_path"]
    if config_sp["tmp_output_path"]=="":
        tmp_output_path = os.path.join(coalition3_path,u"tmp/")
    else:
        tmp_output_path = config_sp["tmp_output_path"]
    UV_precalc_path = config_sp["UV_precalc_path"]

    cfg_set.update({
        "root_path":           root_path,
        "output_path":         fig_output_path,
        "tmp_output_path":     tmp_output_path,
        "fig_output_path":     fig_output_path
    })
    
    ## Add tmp/ and figure/ paths if not yet existant (not in git repo)
    check_create_tmpdir(cfg_set)

    ## Add additional general settings:
    cfg_set.update({
        "save_type":           config["file_handling"]["save_type"],
        "delete_prec":         config["file_handling"]["delete_prec"]=="True",
        "xy_ext":              (int(config["domain_extent"]["xy_ext_x"]),
                                int(config["domain_extent"]["xy_ext_y"]))
    })
    config_td = config["training_dataset"]
    time_change_factor = -1 if config_td["future_disp_reverse"]=="True" else 1
    cfg_set.update({
        "generating_train_ds": config_td["generating_train_ds"]=="True",
        "future_disp_reverse": config_td["future_disp_reverse"]=="True",
        "time_change_factor":  time_change_factor,
        "precalc_UV_disparr":  config_td["precalc_UV_disparr"]=="True"
    })

    ## ===== Read settings on lagrangian_displacement: =========================
    config.read(os.path.join(CONFIG_PATH,"lagrangian_displacement.cfg"))

    ## Check that optical flow source and method comply with only
    ## those implemented:
    config_ld = config["lagrangian_displacement"]
    oflow_source      = config_ld["oflow_source"]
    oflow_method_name = config_ld["oflow_method_name"]
    if oflow_source != "RZC":
        raise NotImplementedError("So far UV displacement retrieval \
                                   only implemented for RZC")
    else:
        config_input_data = configparser.RawConfigParser()
        config_input_data.read(os.path.join(CONFIG_PATH,"input_data.cfg"))
        oflow_source_path = config_input_data["radar_read"]["RZC_path"]
    
    if oflow_method_name != "lucaskanade":
        raise NotImplementedError("So far UV displacement retrieval \
                                   only implemented for lucaskanade method")

    ## N.B. Add smoothing sigma (for the smoothing of COSMO conv variables):
    smooth_sig = float(config_input_data["conv_read"]["smooth_sig"])
    
    ## Check how residual movement should be corrected:
    instant_resid_corr  = config_ld["instant_resid_corr"] == 'True'
    resid_method        = config_ld["resid_method"] \
                            if not instant_resid_corr else "None"
    resid_disp          = False if resid_method=="None" else True
    resid_disp_onestep  = True if resid_method=="Onestep" and resid_disp else \
                            False
                            
    ## Fill up cfg_set dictionary:
    cfg_set.update({
        ## General settings for the lagrangian settings:
        "oflow_method_name":   oflow_method_name,
        "oflow_source":        oflow_source,
        "oflow_source_path":   oflow_source_path,
        "verbose":             config_ld["verbose"]=="True",
        "timestep":            int(config_ld["timestep"]),
        "n_integ":             int(config_ld["n_integ"]),
        "n_past_frames":       int(config_ld["n_past_frames"]),
        
        ## Set threshold to restrict optical flow input data:
        "R_threshold":         float(config_ld["R_threshold"]),
        "R_thresh_meth":       config_ld["R_thresh_meth"],
        
        ## Should precalculated XYW matrix be used:
        "use_precalc_XYW":     config_ld["use_precalc_XYW"]=="True",
        
        ## Settings regarding the correction of residual movement:
        "instant_resid_corr":  instant_resid_corr,
        "resid_method":        resid_method,
        "resid_disp":          resid_disp,
        "resid_disp_onestep":  resid_disp_onestep,
        "n_past_frames_resid": int(config_ld["n_past_frames_resid"]),
        "R_threshold_resid":   float(config_ld["R_threshold_resid"]),
        "R_thresh_meth_resid": config_ld["R_thresh_meth_resid"],
        "UV_inter":            config_ld["UV_inter"]=='True',
        "decl_grid_resid":     int(config_ld["decl_grid_resid"]),
        "inter_fun_resid":     config_ld["inter_fun_resid"],
        "epsilon_resid":       int(config_ld["epsilon_resid"]),
        "zero_interpol":       config_ld["zero_interpol"]=='True',

        ## Displace TRT cell centres against the flow
        ## (instead of the fields along the flow):
        "displ_TRT_cellcent":  config_ld["displ_TRT_cellcent"]=='True',
        
        ## Add sigma for smoothing of COSMO conv variables:
        "smooth_sig":          smooth_sig
    })
    ## Make sanity checks:
    if 60%cfg_set["timestep"]!=0:
        raise ValueError("60min modulo Timestep = %smin is not zero, choose a different one" % cfg_set["timestep"])

    
    ## Read further config information on verification and skill score calculation
    config_ve = config["verification"]
    cfg_set.update({
        "verify_disp": config_ve["verify_disp"]=='True',
        "verif_param": config_ve["verif_param"],
        "scores_list": config_ve["scores_list"].split(','),
        "n_stat":      int(config_ve["n_stat"])
    })
    ## Add extention to filenames if verification is performed:
    file_ext_verif  = "_"+str(cfg_set[cfg_set["verif_param"]]) if cfg_set["verify_disp"] else ""
    cfg_set["file_ext_verif"] = file_ext_verif

    ## ===== Read settings on the domain statistics to be calculated: ================
    config.read(os.path.join(CONFIG_PATH,"domain_statistics.cfg"))
    config_sc = config["statistics_calculation"]
    stat_list           = cfg_var.columns[np.where(cfg_var.columns=="SUM")[0][0] : \
                                          np.where(cfg_var.columns=="MAX")[0][0]+1]
    pixcount_list       = cfg_var.columns[np.where(cfg_var.columns=="PC_NONNAN")[0][0]:]
    stat_sel_form       = config_sc["stat_sel_form"]
    stat_sel_form_width = int(config_sc["stat_sel_form_width"])
    
    ## Get size of domain of interest:
    stat_sel_form_size = form_size(stat_sel_form_width,stat_sel_form)
    
    ## Fill up cfg_set dictionary:
    cfg_set.update({
        "stat_sel_form":       stat_sel_form,
        "stat_sel_form_width": stat_sel_form_width,
        "stat_sel_form_size":  stat_sel_form_size,
        "min_TRT_rank":        float(config_sc["min_TRT_rank"]),
        "save_TRT_domain_map": config_sc["save_TRT_domain_map"]=='True',
        "save_stat_ctrl_imag": config_sc["save_stat_ctrl_imag"]=='True',
        "stat_list":           stat_list,
        "pixcount_list":       pixcount_list
    })

    ## ===== Add abbreviations, units, source, and min. vals of ================
    ## ===== of different variables to the cfg_set: ============================
    ## Static abbreviations:
    abbrev_dict = {"csi": "CSI", "sedi": "SEDI", "hk": "HK"}

    ## Add type (np.datatype), unit (physical unit), source (Radar, SEVIRI..), 
    ## minimum value (minimum value which are recorded of specific variables):
    type_dict = {}; unit_dict = {}; source_dict = {}; minval_dict = {}
    
    ## Loop over variables in var_list:
    for i in range(len(cfg_set["var_list"])):
        abbrev_dict.update({var_list[i]: cfg_var["DESCRITPION"][cfg_var["READ_IF"]].tolist()[i]})
        type_dict.update({var_list[i]:   cfg_var["DATA_TYPE"][cfg_var["READ_IF"]].tolist()[i]})
        unit_dict.update({var_list[i]:   cfg_var["UNIT"][cfg_var["READ_IF"]].tolist()[i]})
        source_dict.update({var_list[i]: cfg_var["SOURCE"][cfg_var["READ_IF"]].tolist()[i]})
        minval_dict.update({var_list[i]: cfg_var["MIN_VALUE"][cfg_var["READ_IF"]].tolist()[i]})
    if cfg_set["displ_TRT_cellcent"]:
        abbrev_dict.update({"TRT": "TRT Cell Centre"})
        type_dict.update({"TRT": "int32"})
        unit_dict.update({"TRT": "Cell ID"})
        source_dict.update({"TRT": "TRT"})
        minval_dict.update({"TRT": None})
    cfg_set.update({
        "abbrev_dict":         abbrev_dict,
        "type_dict":           type_dict,
        "unit_dict":           unit_dict,
        "source_dict":         source_dict,
        "minval_dict":         minval_dict
    })
    
    ## Loop over combi variables (channel combinations) in var_list:
    type_dict_combi = {}; unit_dict_combi = {}; source_dict_combi = {}; abbrev_dict_combi = {}
    for i in range(len(var_combi_list)):
        abbrev_dict_combi.update({var_combi_list[i]: return_var_combi_information(var_combi_list[i],
                                                        cfg_var_combi,abbrev_dict,"abbrev")})
        type_dict_combi.update({var_combi_list[i]:   return_var_combi_information(var_combi_list[i],
                                                        cfg_var_combi,type_dict,"data_type")})
        unit_dict_combi.update({var_combi_list[i]:   return_var_combi_information(var_combi_list[i],
                                                        cfg_var_combi,unit_dict,"unit")})
        source_dict_combi.update({var_combi_list[i]: return_var_combi_information(var_combi_list[i],
                                                        cfg_var_combi,source_dict,"source")})
    cfg_set.update({
        "abbrev_dict_combi":   abbrev_dict_combi,
        "type_dict_combi":     type_dict_combi,
        "unit_dict_combi":     unit_dict_combi,
        "source_dict_combi":   source_dict_combi
    })
    
    ## (Very inelegant) Hard coded list of TRT columns which should be read:
    TRT_cols = ['date', 'lon', 'lat', 'iCH', 'jCH', 'ell_L', 'ell_S', 'area', 'vel_x', 'vel_y', 'det',
                'RANKr', 'CG', 'CG_plus', 'CG_minus', 'perc_CG_plus', 'ET45', 'ET45m', 'ET15', 'ET15m', 'VIL',
                'maxH', 'maxHm', 'POH', 'RANK', 'Dvel_x', 'Dvel_y', 'angle']
    TRT_dtype = [np.object, np.float32, np.float32, np.int16, np.int16, np.float32, np.float32,
                 np.int32, np.float32, np.float32, np.int16,
                 np.int16, np.int16, np.int16, np.int16, np.int16, np.float32, np.float32,
                 np.float32, np.float32, np.float32,
                 np.float32, np.float32, np.float32, np.int16, np.float32, np.float32, np.float32]
    type_dict_TRT = {}
    for i in range(len(TRT_cols)): type_dict_TRT.update({TRT_cols[i]: TRT_dtype[i]})
    cfg_set.update({
        "TRT_cols":            TRT_cols,
        "TRT_dtype":           TRT_dtype,
        "type_dict_TRT":       type_dict_TRT
    })

    return(cfg_set, cfg_var, cfg_var_combi)


## Append current time (t0) information to setting config (cfg_set) file:
def cfg_set_append_t0(cfg_set,t0_str):
    """Append current time (t0) information to setting config (cfg_set) file.

    Parameters
    ----------

    cfg_set : dict
        Dictionary with the basic variables used throughout the code:

    t0_str : str
        String of the time t0 (displacement target time)
    """

    ## Get information on date and time
    t0      = dt.datetime.strptime(t0_str, "%Y%m%d%H%M")
    t0_orig = dt.datetime.strptime(t0_str, "%Y%m%d%H%M")
    t0_doy  = t0.timetuple().tm_yday

    cfg_set.update({
        "t0":      t0,
        "t0_str":  t0_str,
        "t0_doy":  t0_doy,
        "t0_orig": t0_orig
    })
    return cfg_set

## Check whether temporary subdirectories are present, otherwise create those:
def check_create_tmpdir(cfg_set):
    """Check whether tmp and figure directory exist, if not, create those.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
    """
    if not os.path.exists(cfg_set["tmp_output_path"]):
        os.makedirs(cfg_set["tmp_output_path"])
        print("  Created tmp/ directory: %s" % cfg_set["tmp_output_path"])
    if not os.path.exists(cfg_set["fig_output_path"]):
        os.makedirs(cfg_set["fig_output_path"])
        print("  Created figures/ directory: %s" % cfg_set["fig_output_path"])

## Get size of the domain form chosen (from where the stats are read):
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
    
## Function that returns configuration information of the combi variables (SEVIRI channel combis):
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

## Print information before running script:
def print_config_info_op(cfg_set): #,CONFIG_FILE_set,CONFIG_FILE_var
    """Print information before running script"""
    print("\n-------------------------------------------------------------------------------------------------------\n")
    print_logo()
    if cfg_set["verif_param"] == "R_threshold":
        unit_verifparam = " mm/h" if cfg_set["R_thresh_meth"] == "fix" else "% quantile"
    else: unit_verifparam = ""
    print_str = '    Configuration of NOSTRADAMUS input date preparation procedure:'+ \
    '\n      Date:             '+cfg_set["t0"].strftime("%Y-%m-%d %H:%M")+ \
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
    #'\n      Config files:     '+CONFIG_FILE_set+' (Settings) & '+CONFIG_FILE_var+' (Variables)'+ \
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
    
## Print COALITION3 Logo:
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
    