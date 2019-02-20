""" [COALITION3] Convert (to NetCDF) and concatenate single xarray pickle files
    containing the statistics and pixel counts into two big files, one containing
    all the past observations, one containing all the future observations (_rev.pkl).
    These are then concatenated into one big file (containing past and future obs),
    then the auxiliary variables are added. Finally, derived variables (TRT Rank
    and the respective difference to the Rank of the TRT files) are added."""

# ===============================================================================
# Import packages and functions
import sys
import pdb

import coalition3.training.concat as cnc
import coalition3.operational.statistics as stat
import coalition3.inout.paths as pth
 
## ===============================================================================
## Get path to datasets:
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
stat_path      = pth.get_stat_path(user_argv_path)

## Convert single '<Datetime>_stat_pixcount*.pkl' files to NetCDF:
#cnc.convert_stat_files(stat_path)

## Concatenate single event <Datetime>_stat_pixcount*.pkl' files to one big file (past and future seperate):
#ds_past   = cnc.concat_stat_files(stat_path,False)
#ds_future = cnc.concat_stat_files(stat_path,True)

## Concatenate past and future statistics into the final mega-super-duper stats file:
#ds = cnc.concat_future_past_concat_stat_files(stat_path, ds_past, ds_future)

## Add auxiliary static variables (solar time, topographical and qualitiy
## (freq. of radar returns) information) and TRT Rank:
#ds = cnc.wrapper_fun_add_aux_static_variables(stat_path, ds)   

## [DEPRECATED] Add derived information (e.g. TRT-Rank):
ds = cnc.wrapper_fun_add_derived_variables(stat_path, ds)

## Make various plots (nothing too serious, just for an overview):
#cnc.collection_of_plotting_functions(stat_path)

