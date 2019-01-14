""" [COALITION3] Reading config information"""

from __future__ import division
from __future__ import print_function

import configparser
import datetime as dt
import numpy as np

## =============================================================================
## FUNCTIONS:

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

    cfg_set = {"t0":       t0,
               "t0_str":   t0_str,
                "t0_doy":  t0_doy,
                "t0_orig": t0_orig
               }
    return cfg_set


## Get settings from config file and some additional static settings:
def get_config_info(CONFIG_PATH=None): #**kwargs):
    """Get information from configuration file and return in a dictionary
    "cfg_set". Furthermore, get data from variable configuration file
    (cfg_var.csv) and return in a pandas.dataframe "cfg_var"

    Parameters
    ----------

    CONFIG_PATH : str
        Path to config file.

    CONFIG_FILE_set : str
        Name of settings config file.

    CONFIG_FILE_var : str
        Name of variable config file (.csv).

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
    coalition3_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '../../'))
    #CONFIG_PATH,CONFIG_FILE_set,CONFIG_PATH_set, \
    #                        CONFIG_FILE_var,CONFIG_FILE_var_combi
    if "CONFIG_PATH" in kwargs.keys():
        CONFIG_PATH = kwargs["CONFIG_PATH"]
    else:
    if CONFIG_PATH is None:
        CONFIG_PATH = os.path.abspath(os.path.join(coalition3_path,'/config/'))

    ## ===== Make static settings: =========================
    cfg_set = {
        "adv_method":          "semilagrangian",
        "square_domain":       False,
        "colorscale":          "MeteoSwiss",  # MeteoSwiss or STEPS-BE
        "motion_plot":         "quiver",      # streamplot or quiver
    }

    ## ===== Read the general settings: ============
    config = configparser.RawConfigParser()
    config.read(os.path.join(CONFIG_PATH,"general_settings.cfg"))

    ## Add source paths:
    config_sp = config["source_paths"]
    if config_sp["root_path"]=="":
        root_path = coalition3_path
    else:
        root_path = config_sp["root_path"]
    if config_sp["fig_output_path"]=="":
        fig_output_path = os.path.join(coalition3_path,"")
    else:
        fig_output_path = config_sp["fig_output_path"]
    if config_sp["tmp_output_path"]=="":
        tmp_output_path = os.path.join(coalition3_path,"")
    else:
        tmp_output_path = config_sp["tmp_output_path"]
    UV_precalc_path = config_sp["UV_precalc_path"]

    cfg_set = {
        "root_path":           root_path,
        "output_path":         fig_output_path,
        "tmp_output_path":     tmp_output_path
    }

    ## Add additional general settings:
    cfg_set = {
        "save_type":           config["file_handling"]["save_type"],
        "delete_prec":         config["file_handling"]["delete_prec"]=="True",
        "xy_ext":              (int(config["domain_extent"]["xy_ext_x"]),
                                int(config["domain_extent"]["xy_ext_y"])),
        "delete_prec":         config["domain_extent"]["delete_prec"]=="True"
    }
    config_td = config["training_dataset"]
    time_change_factor = -1 if config_td["future_disp_reverse"]=="True" else 1
    cfg_set = {
        "generating_train_ds": config_td["generating_train_ds"]=="True",
        "future_disp_reverse": config_td["future_disp_reverse"]=="True",
        "time_change_factor":  time_change_factor,
        "time_change_factor":  config_td["time_change_factor"]=="True"
    }

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
    if oflow_method_name != "lucaskanade":
        raise NotImplementedError("So far UV displacement retrieval \
                                   only implemented for lucaskanade method")

    ## Check how residual movement should be corrected:
    instant_resid_corr  = config_ld["instant_resid_corr"] == 'True'
    resid_method        = config_ld["resid_method"] \
                            if not instant_resid_corr else "None"
    resid_disp          = False if resid_method=="None" else True
    resid_disp_onestep  = True if resid_method=="Onestep" and resid_disp else \
                            False



    cfg_set = {
              "oflow_method_name":   oflow_method_name,
              "oflow_source":        oflow_source,
              "verbose":             config_ld["verbose"]=="True",
              "timestep":            int(config_ld["timestep"]),
              "n_integ":             int(config_ld["n_integ"]),
              "n_past_frames":       int(config_ld["n_past_frames"]),

              "instant_resid_corr":  instant_resid_corr,
              "resid_method":        resid_method,
              "resid_disp":          resid_disp,
              "resid_disp_onestep":  resid_disp_onestep,



              "R_threshold":         float(config_ld["R_threshold"]),
              "R_thresh_meth":       config_ld["R_thresh_meth"],
              #"wind_path":           wind_path,
              #"conv_path":           conv_path,
              "oflow_source_path":   oflow_source_path,
              "n_past_frames_resid": int(config_ld["n_past_frames_resid"]),
              "use_precalc_XYW":     config_ld["use_precalc_XYW"]=="True",
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




    ## ===== Import configuration information on variables: =============
    dtype_cfg_var = {'N_CATEGORIES': 'uint8', 'MIN_VALUE': 'float32'}
    dtype_cfg_var.update(dict.fromkeys(['PREDICTOR', 'LAGRANGIAN', 'PAST_OBS', 'SMOOTH', 'SUM', 'MEAN', 'STDDEV','IQR', \
                                        'PERC01', 'PERC05', 'PERC30','PERC50', 'PERC80', 'PERC95','PERC99', 'MIN', 'MAX', \
                                        'PC_NONNAN', 'PC_NONZERO', 'PC_LT55','PC_CAT','READ','READ_IF','VARIABILITY'], 'bool'))
    dtype_cfg_var_combi = {'dt': 'bool','PREDICTOR': 'bool'}
    cfg_var       = pd.read_csv("%s/%s" % (CONFIG_PATH_set,CONFIG_FILE_var),delimiter=";",dtype=dtype_cfg_var)
    cfg_var_combi = pd.read_csv("%s/%s" % (CONFIG_PATH_set,CONFIG_FILE_var_combi),delimiter=";",dtype=dtype_cfg_var_combi)








    displ_TRT_cellcent  = config_ld["displ_TRT_cellcent"]=='True'

    #resid_disp_onestep  = config_bs["resid_disp_onestep"]=='True'
    R_threshold_resid   = float(config_ld["R_threshold_resid"])
    R_thresh_meth_resid = config_ld["R_thresh_meth_resid"]
    UV_inter            = config_ld["UV_inter"]=='True'
    decl_grid_resid     = int(config_ld["decl_grid_resid"])
    zero_interpol       = config_ld["zero_interpol"]=='True'
    inter_fun_resid     = config_ld["inter_fun_resid"]
    epsilon_resid       = int(config_ld["epsilon_resid"])

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



    oflow_source_path = RZC_path






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
