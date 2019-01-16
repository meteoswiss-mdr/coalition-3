from __future__ import division
from __future__ import print_function

import sys
import datetime
import pandas as pd
import xarray as xr
import numpy as np
import pdb

import NOSTRADAMUS_0_training_ds_fun as Nds
import NOSTRADAMUS_1_input_prep_fun as Nip

## ===============================================================================
## Make static setting

## Define path to config file
CONFIG_PATH_set_tds   = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/"
CONFIG_FILE_set_tds   = "NOSTRADAMUS_0_training_ds.cfg"

cfg_set_tds   = Nds.get_config_info(CONFIG_PATH_set_tds,CONFIG_FILE_set_tds)
timestep      = datetime.datetime(2018,07,04,17,10)
cfg_set_input, cfg_var = Nip.get_config_info(cfg_set_tds["CONFIG_PATH_set_input"],
                                             cfg_set_tds["CONFIG_FILE_set_input"],
                                             cfg_set_tds["CONFIG_FILE_var_input"],
                                             timestep.strftime("%Y%m%d%H%M"))

TRT_disp = xr.open_dataset('/opt/users/jmz/2_input_NOSTRADAMUS_ANN/tmp/201807041710_TRT_disp.nc')
print(np.unique(TRT_disp["TRT"][0,:,:][np.where(TRT_disp["TRT"][0,:,:]>0)]))
print(np.unique(TRT_disp["TRT"][np.where(TRT_disp["TRT"]>0)]))
print(np.sum(TRT_disp>0,axis=(1,2)))

TRT_disp = xr.open_dataset('/opt/users/jmz/2_input_NOSTRADAMUS_ANN/tmp/201807041710_TRT_disp_rev.nc')
print(np.unique(TRT_disp["TRT"][0,:,:][np.where(TRT_disp["TRT"][0,:,:]>0)]))
print(np.unique(TRT_disp["TRT"][np.where(TRT_disp["TRT"]>0)]))
print(np.sum(TRT_disp>0,axis=(1,2)))

pdb.set_trace()