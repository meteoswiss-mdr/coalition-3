""" [COALITION3] Reading functions for xarray datasets"""

from __future__ import division
from __future__ import print_function

import os
import psutil
import pickle
import xarray as xr

## =============================================================================
## FUNCTIONS:
    
## Read xarray datasets into RAM (either fully or lazy if too big): 
def xarray_file_loader(path_str):
    if path_str[-3:]==".nc":
        expected_memory_need = float(os.path.getsize(path_str))/psutil.virtual_memory().available*100
        if expected_memory_need > 50:
            print("  *** Warning: File %i is opened as dask dataset (expected memory use: %02d%%) ***" %\
                  (expected_memory_need))
            xr_n = xr.open_mfdataset(path_str,chunks={"DATE_TRT_ID":1000})
        else: xr_n = xr.open_dataset(path_str)
    elif path_str[-4:]==".pkl":
        with open(path_str, "rb") as path: xr_n = pickle.load(path)
    return xr_n
