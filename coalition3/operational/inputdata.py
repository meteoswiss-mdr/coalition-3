""" [COALITION3] Reading the input data (from CCS4 domain (Radar, SEVIRI, COSMO, THX)
    and from TRT)."""

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np

from coalition3.inout.readccs4 import get_vararr_t
from coalition3.inout.readTRT import get_vararr_TRT_t0
from coalition3.inout.paths import path_creator_vararr
from coalition3.inout.iotmp import save_file

## =============================================================================
## FUNCTIONS:


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
        if cfg_set["source_dict"][var]=="METADATA":
            continue
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
    if cfg_set["source_dict"][var]=="METADATA":
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
    