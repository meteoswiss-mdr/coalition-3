# coding: utf-8
""" [COALITION3] Import xarray dataset containing statistics and
    pixel counts, and convert into 2d Pandas dataframe containing
    the predictive variables (statistics and TRT information)
    and the target variables (TRT Ranks) """
    
## Import packages and define functions:
from __future__ import print_function

import os
import sys

import coalition3.inout.paths as pth
import coalition3.inout.readxr as rxr
import coalition3.operational.convertds as cds

## ============================================================================
print("\n%s\n Converting xarray training dataset to 2D Pandas dataframe\n" % (80*'-'))

print("  Read path to xarray training dataset")
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_ds = pth.file_path_reader("xarray training dataset",user_argv_path)
path_to_df = "%s_df.h5" % (os.path.splitext(path_to_ds)[0])

## Load xarray dataset:
ds = rxr.xarray_file_loader(path_to_ds)

## Convert to pandas dataframe:
cds.convert_ds2df(ds, outpath=path_to_df, diff_option=None)