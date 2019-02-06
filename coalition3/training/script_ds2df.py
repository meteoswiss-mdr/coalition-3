# coding: utf-8
""" [COALITION3] Import xarray dataset containing statistics and
    pixel counts, and convert into 2d Pandas dataframe containing
    the predictive variables (statistics and TRT information)
    and the target variables (TRT Ranks) """
    
## Import packages and define functions:
from __future__ import print_function

import os
import sys
import dask
import pickle
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pylab as plt
#import dask.dataframe as dd
#from dask.distributed import Client
#from dask_ml.xgboost import XGBRegressor
from xgboost import plot_importance, XGBRegressor

import coalition3.inout.paths as pth
import coalition3.inout.readxr as rxr
import coalition3.inout.readconfig as cfg

sys.stdout.flush()

#@dask.delayed
def da2df(da,key):
    print("  Working on %s" % key, end='\r')
    return da.to_pandas().to_frame(filter_observations=False).T
    
## ============================================================================
print("\n%s\n Converting xarray training dataset to 2D Pandas dataframe\n" % (80*'-'))

## Open xarray dataset:
#ds = xr.open_mfdataset("output_data/Combined_stat_pixcount.nc",
#                       chunks={"DATE_TRT_ID": 10000})
#ds = xr.open_dataset("/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset_combined/diam_23km/Combined_stat_pixcount.nc")

print("  Read path to xarray training dataset")
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_ds = pth.file_path_reader("xarray training ds",user_argv_path)
ds = rxr.xarray_file_loader(path_to_ds)

## Check for new variables which cannot be converted yet (e.g. categorical variables):
unconvertable_vars = [var for var in ds.data_vars if "CMA" in var or "CT" in var]
if len(unconvertable_vars)>0:
    raise ImplementationError("Categorical counting not yet implemented")

## Extract future TRT Ranks (target variable) and calculate
## Rank difference to t0.
print("  Extract future TRT Ranks and pixel counts (treated seperately)")
ds_TRTrank_future = ds["TRT_Rank"].where(ds["time_delta"]>0, drop=True).rename("TRT_Rank")
ds_TRTrank_future = ds_TRTrank_future.sel(time_delta=slice(5,45)) - \
                    ds["TRT_Rank"].sel(time_delta=0)

## Extract pixel counts of Radar variables with "nonmin" statistics:
ds_pixc_radar = ds[[var[:-12]+u"_pixc" for var in ds.data_vars if "nonmin" in var]]
ds_pixc_radar = ds_pixc_radar.sel(pixel_count="PC_NONMIN").drop("pixel_count").where(ds_pixc_radar["time_delta"]<=0, drop=True).astype(np.int16)
                    
## Delete unwanted or already extracted (see above) variables
## (e.g. pixel counts, TRT_Ranks):
drop_list = [var for var in ds.data_vars if "_pixc" in var]
drop_list += [u"TRT_Rank",u"TRT_Rank_diff",u"TRT_domain_indices",
              u"pixel_count",u"TRT_cellcentre_indices","date"]
ds_drop = ds.drop(drop_list)

## Extract TRT variables (CG, Dvel_x, ..) and solar time:
print("  Extract 1D variables (TRT vars and solar time)")
ds_1d   = ds_drop[[var for var in ds_drop.data_vars if len(ds_drop[var].shape)<2]]

## Delete future values (time_delta > 0) and calculate absolute difference
## between statistics at t0 and time_delta < 0. Also, set NaN-values in 
## "_nonmin" statistics to min_value:
print("  Extract 2D variables (with 'time_delta' coordinate)")
ds_23d  = ds_drop[[var for var in ds_drop.data_vars if len(ds_drop[var].shape)>=2]]
del(ds_drop)

print("     Take difference to t0 value / set NaN to min_value in '_nonmin' statistics (TIME CONSUMING)")
ds_past = ds_23d.where(ds_23d["time_delta"]<=0, drop=True)
del(ds_23d)

cfg_set, cfg_var, cfg_var_combi = cfg.get_config_info_op()
for var in ds_past.data_vars:
    if len(ds_past[var].sel(time_delta=0).values.shape)==1:
        ## Special case for variable 'CZC_lt57dBZ'
        sub_val = ds_past[var].sel(time_delta=slice(-45,-5)).values-ds_past[var].sel(time_delta=0).values[:,np.newaxis]
        ds_past[var].values = np.concatenate([sub_val,ds_past[var].sel(time_delta=0).values[:,np.newaxis]],axis=1)
    else:
        sub_val = ds_past[var].sel(time_delta=slice(-45,-5)).values-ds_past[var].sel(time_delta=0).values[:,np.newaxis,:]
        ds_past[var].values = np.concatenate([sub_val,ds_past[var].sel(time_delta=0).values[:,np.newaxis,:]],axis=1)
        
    ## Set NaN-values in "_nonmin" statistics to min_value:
    if "_nonmin" in var:
         ds_past[var].values[np.isnan(ds_past[var].values)] = cfg_set["minval_dict"][var[:-12]]


## Convert 3d dataarrays (xarray) to 2d dataframes (pandas) - TIME CONSUMING!
print("  Converting 3D variables to dataframe (TIME CONSUMING)")
df_list_3d = [da2df(ds_past[da],da) for da in ds_past.data_vars if len(ds_past[da].shape)==3]
#df_list.compute()
df_3d = pd.concat(df_list_3d,axis=1,copy=False,
                  keys=[da for da in ds_past.data_vars if len(ds_past[da].shape)==3])
del(df_list_3d)

## Concatenate column names:
df_3d.columns.set_levels(df_3d.columns.levels[2].values.astype(np.unicode),
                          level=2,inplace=True)
df_3d.columns.rename("Variable", level=0, inplace=True)
df_3d.columns = df_3d.columns.map('{0[0]}|{0[1]}|{0[2]}'.format)
df_3d.index   = df_3d.index.astype(np.unicode)
#df_3d.to_hdf("df_23km_nd.h5",key="df_3d",mode="w",complevel=0)

## Convert 2d dataarrays (xarray) to 2d dataframes (pandas)
print("  Converting 2D variables to dataframe")
df_list_2d = [ds_past[u'CZC_lt57dBZ'].sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in np.arange(-45,5,5)]
df_list_colnames = [u'CZC_lt57dBZ|%i|SUM' % deltime for deltime in np.arange(-45,5,5)]
for var in ds_pixc_radar.data_vars:
    df_list_2d += [ds_pixc_radar[var].sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in np.arange(-45,5,5)]
    df_list_colnames += [u'%s_NONMIN|%i|SUM' % (var,deltime) for deltime in np.arange(-45,5,5)]
df_2d = pd.concat(df_list_2d,axis=1,copy=False)
df_2d.columns = df_list_colnames
df_2d = df_2d.astype(np.int16)
del(df_list_2d,df_list_colnames,ds_past,ds_pixc_radar)
#df_2d.to_hdf("df_23km_nd.h5",key="df_2d",mode="a",complevel=0)

df_list_TRT = [ds_TRTrank_future.sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in np.arange(5,50,5)]
df_TRT = pd.concat(df_list_TRT,axis=1,copy=False)
df_TRT.columns = [u'TRT_Rank|%i' % deltime for deltime in np.arange(5,50,5)]
del(df_list_TRT)
#df_TRT.to_hdf("df_23km_nd.h5",key="df_TRT",mode="a",complevel=0)

## Convert 1d dataarrays (xarray) to 2d dataframes (pandas)
print("  Converting 1D variables to dataframe")
df_1d = ds_1d.to_dataframe()
#df_1d.to_hdf("df_23km_nd.h5",key="df_1d",mode="a",complevel=0)

## Concatenate 3d/2d/1d dataframes and save to disk:
print("  Concatenate into one big dataframe and save to disk")
df = pd.concat([df_1d,df_2d,df_3d,df_TRT],axis=1,copy=False)
save_path = "%s_df.h5" % os.path.splitext(user_argv_path)[0]
df.to_hdf(save_path,key="df",mode="w",complevel=0)
print("    Saving successfull")




