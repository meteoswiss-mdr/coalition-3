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

import coalition3.inout.paths as pth
import coalition3.inout.readxr as rxr
import coalition3.inout.readconfig as cfg

#@dask.delayed
def da2df(da,data_vars=None):
    if data_vars is not None:
        perc_fin = np.float(list(data_vars).index(da.name)+1)/len(data_vars)*100
        print("  Working on %s (%3d%%)                     " % \
                (da.name,int(perc_fin)), end='\r')
    else:
        print("  Working on %s           " % da.name, end='\r')
    sys.stdout.flush()
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

## Get time delta values (also just pos/neg ones):
time_del = ds.time_delta.values
neg_del  = time_del[time_del<0]; neg0_del  = time_del[time_del<=0]
pos_del  = time_del[time_del>0]; pos0_del  = time_del[time_del>=0]

## Check for new variables which cannot be converted yet (e.g. categorical variables):
unconvertable_vars = [var for var in ds.data_vars if "CMA" in var or "CT" in var]
if len(unconvertable_vars)>1: # and unconvertable_vars[0]!='TOPO_ASPECT_stat':
    raise NotImplementedError("Categorical counting not yet implemented")

## Extract future TRT Ranks (target variable) and calculate
## Rank difference to t0.
print("  Extract future TRT Ranks and pixel counts (treated seperately)")
ds_TRTrank_val  = ds["TRT_Rank"] #.where(ds["time_delta"]>0, drop=True).rename("TRT_Rank")
ds_TRTrank_diff = ds_TRTrank_val - ds["TRT_Rank"].sel(time_delta=0) #ds_TRTrank_val.sel(time_delta=slice(5,45)) - \
ds_TRTrank_diff = ds_TRTrank_diff.rename("TRT_Rank_diff")

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

## Decide between deltas between time steps to delta to t0:
print_text = """
\nHow should variables be treated over time:
  Option 1 -> Keep the absolute values of the statistics [path addon 'nodiff']
              (e.g. MaxRZC(t0-45min), MaxRZC(t0-40min), .. , MaxRZC(t0))
  Option 2 -> Take the difference to the statistic at t0 and keep absolute value at t0 [path addon 't0diff']
              (e.g. MaxRZC(t0)-MaxRZC(t-45min), MaxRZC(t0)-MaxRZC(t-40min), .. , MaxRZC(t0))
  Option 3 -> Between each time step (and keep absolute value at t0) [path addon 'dtdiff']
              (e.g. MaxRZC(t0-40min)-MaxRZC(t0-45min), MaxRZC(t0-35min)-MaxRZC(t0-40min), .. , MaxRZC(t0))
"""
print(print_text)
diff_option = None
while (diff_option!="1" and diff_option != "2" and diff_option != "3"):
    diff_option = str(raw_input("Which option do you choose? [1/2/3] "))

## Delete "future" values:
print("     Take difference to t0 value / set NaN to min_value in '_nonmin' statistics (TIME CONSUMING)")
ds_past = ds_23d.where(ds_23d["time_delta"]<=0, drop=True)
del(ds_23d)

## Take the difference:
cfg_set, cfg_var, cfg_var_combi = cfg.get_config_info_op()
for var in ds_past.data_vars:
    if diff_option == "2":
        if len(ds_past[var].sel(time_delta=0).values.shape)==1:
            ## Special case for variable 'CZC_lt57dBZ'
            sub_val = ds_past[var].sel(time_delta=slice(neg_del[0],neg_del[-1])).values-ds_past[var].sel(time_delta=0).values[:,np.newaxis]
            ds_past[var].values = np.concatenate([sub_val,ds_past[var].sel(time_delta=0).values[:,np.newaxis]],axis=1)
        else:
            sub_val = ds_past[var].sel(time_delta=slice(neg_del[0],neg_del[-1])).values-ds_past[var].sel(time_delta=0).values[:,np.newaxis,:]
            ds_past[var].values = np.concatenate([sub_val,ds_past[var].sel(time_delta=0).values[:,np.newaxis,:]],axis=1)
        
    elif diff_option == "3":
        sub_val = ds_past[var].sel(time_delta=slice(neg_del[1],0)).values-ds_past[var].sel(time_delta=slice(neg_del[0],neg_del[-1])).values
        if len(ds_past[var].sel(time_delta=0).values.shape)==1:
            ## Special case for variable 'CZC_lt57dBZ'
            ds_past[var].values = np.concatenate([sub_val,ds_past[var].sel(time_delta=0).values[:,np.newaxis]],axis=1)
        else:
            ds_past[var].values = np.concatenate([sub_val,ds_past[var].sel(time_delta=0).values[:,np.newaxis,:]],axis=1)
            
    ## Set NaN-values in "_nonmin" statistics to min_value:
    if "_nonmin" in var:
         ds_past[var].values[np.isnan(ds_past[var].values)] = cfg_set["minval_dict"][var[:-12]]

## Convert 3d dataarrays (xarray) to 2d dataframes (pandas) - TIME CONSUMING!
print("  Converting 3D variables to dataframe (TIME CONSUMING)")
df_list_3d = [da2df(ds_past[da],ds_past.data_vars) for da in ds_past.data_vars if len(ds_past[da].shape)==3]
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
df_list_2d = [ds_past[u'CZC_lt57dBZ'].sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in neg0_del]
df_list_colnames = [u'CZC_lt57dBZ|%i|SUM' % deltime for deltime in neg0_del]
for var in ds_pixc_radar.data_vars:
    df_list_2d += [ds_pixc_radar[var].sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in neg0_del]
    df_list_colnames += [u'%s_NONMIN|%i|SUM' % (var,deltime) for deltime in neg0_del]
df_2d = pd.concat(df_list_2d,axis=1,copy=False)
df_2d.columns = df_list_colnames
df_2d = df_2d.astype(np.int16)
del(df_list_2d,df_list_colnames,ds_past,ds_pixc_radar)
#df_2d.to_hdf("df_23km_nd.h5",key="df_2d",mode="a",complevel=0)

df_list_TRT_val     = [ds_TRTrank_val.sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in time_del]
df_TRT_val          = pd.concat(df_list_TRT_val,axis=1,copy=False)
df_TRT_val.columns  = [u'TRT_Rank|%i' % deltime for deltime in time_del]
df_list_TRT_diff    = [ds_TRTrank_diff.sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in time_del]
df_TRT_diff         = pd.concat(df_list_TRT_diff,axis=1,copy=False)
df_TRT_diff.columns = [u'TRT_Rank_diff|%i' % deltime for deltime in time_del]
del(df_list_TRT_val,df_list_TRT_diff)
#df_TRT.to_hdf("df_23km_nd.h5",key="df_TRT",mode="a",complevel=0)

## Convert 1d dataarrays (xarray) to 2d dataframes (pandas)
print("  Converting 1D variables to dataframe")
df_1d = ds_1d.to_dataframe()
#df_1d.to_hdf("df_23km_nd.h5",key="df_1d",mode="a",complevel=0)

## Concatenate 3d/2d/1d dataframes and save to disk:
print("  Concatenate into one big dataframe and save to disk")
df = pd.concat([df_1d,df_2d,df_3d,df_TRT_val,df_TRT_diff],axis=1,copy=False)
if diff_option == "1":
    path_addon = "nodiff"
elif diff_option == "2":
    path_addon = "t0diff"
elif diff_option == "3":
    path_addon = "dtdiff"
save_path = "%s_df_%s.h5" % (os.path.splitext(path_to_ds)[0],path_addon)
df.to_hdf(save_path,key="df",mode="w",complevel=0)
print("    Saving successfull")




