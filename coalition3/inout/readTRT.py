""" [COALITION3] Reading functions for TRT data from text files"""

from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

from mpop.satin import swisstrt

import coalition3.inout.paths as pth

## =============================================================================
## FUNCTIONS:
    
## Produce array with TRT cell centre pixels 
def get_vararr_TRT_t0(t0, cfg_set):
    """Provide vararr style array filled with centre locations of TRT cells"""
    
    ## Read filepath of respective TRT file:
    filepaths, timestamps = pth.path_creator(t0, "TRT", "TRT", cfg_set)
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


## Create empty dataframe with specific columns and datatypes:
def df_empty(columns, dtypes, index=None):
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df