""" [COALITION3] Script that looks the NetCDF files containing the displaced
    TRT cell centres, and checks that there is no increase or decrease in
    the number of TRT cells."""

from __future__ import division
from __future__ import print_function

import sys
import datetime
import pandas as pd
import xarray as xr
import numpy as np
import pdb

## ===============================================================================
## Make static setting

## Define path to config file
TRT_disp = xr.open_dataset('<path..%Y%m%d%H%M_TRT_disp.nc')
print(np.unique(TRT_disp["TRT"][0,:,:][np.where(TRT_disp["TRT"][0,:,:]>0)]))
print(np.unique(TRT_disp["TRT"][np.where(TRT_disp["TRT"]>0)]))
print(np.sum(TRT_disp>0,axis=(1,2)))

TRT_disp = xr.open_dataset('<path..%Y%m%d%H%M_TRT_disp_rev.nc')
print(np.unique(TRT_disp["TRT"][0,:,:][np.where(TRT_disp["TRT"][0,:,:]>0)]))
print(np.unique(TRT_disp["TRT"][np.where(TRT_disp["TRT"]>0)]))
print(np.sum(TRT_disp>0,axis=(1,2)))

pdb.set_trace()