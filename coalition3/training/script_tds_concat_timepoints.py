#!/opt/users/common/packages/anaconda2//bin

""" Concatenate single xarray pickle files containing the statistics and pixel counts"""

# ===============================================================================
# Import packages and functions
import sys
import pdb

import NOSTRADAMUS_0_training_ds_fun as Nds
import NOSTRADAMUS_1_input_prep_fun as Nip
   
## ===============================================================================
## Get path to datasets:
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
stat_path      = Nds.get_stat_path(user_argv_path)
"""
## Convert single '<Datetime>_stat_pixcount*.pkl' files to NetCDF:
Nds.convert_stat_files(stat_path)

## Concatenate single event <Datetime>_stat_pixcount*.pkl' files to one big file (past and future seperate):
Nds.concat_stat_files(stat_path,True)
Nds.concat_stat_files(stat_path,False)
"""
## Concatenate past and future statistics into the final mega-super-duper stats file:
Nds.concat_future_past_concat_stat_files(stat_path)
"""
## Add auxiliary static variables (solar time, topographical and qualitiy (freq. of radar returns) information):
CONFIG_PATH_set_tds   = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset_30min_16km_23km_20181128/"
CONFIG_FILE_set_tds   = "NOSTRADAMUS_0_training_ds_20181128.cfg"
Nds.wrapper_fun_add_aux_static_variables(CONFIG_PATH_set_tds,CONFIG_FILE_set_tds,stat_path)   

## Add derived information (e.g. TRT-Rank):
Nip.add_derived_variables(stat_path)
pdb.set_trace()
"""
"""
## Make various plots (nothing too serious, just for an overview):
Nds.collection_of_plotting_functions(stat_path)
"""