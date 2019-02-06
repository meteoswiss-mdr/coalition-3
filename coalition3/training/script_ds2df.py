# coding: utf-8
""" [COALITION3] Import xarray dataset containing statistics and
    pixel counts, and convert into 2d Pandas dataframe containing
    the predictive variables (statistics and TRT information)
    and the target variables (TRT Ranks) """
    
## Import packages and define functions:
import os
import xarray as xr
import numpy as np
import pandas as pd
import pickle
import dask

import dask.dataframe as dd
from dask.distributed import Client
#from dask_ml.xgboost import XGBRegressor
from xgboost import plot_importance, XGBRegressor
import matplotlib.pylab as plt

import coalition3.inout.readconfig as cfg

#@dask.delayed
def da2df(da,key):
    print(key)
    return da.to_pandas().to_frame(filter_observations=False).T
    
## ============================================================================
## Open xarray dataset:
#ds = xr.open_mfdataset("output_data/Combined_stat_pixcount.nc",
#                       chunks={"DATE_TRT_ID": 10000})
ds = xr.open_dataset("/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset_combined/diam_23km/Combined_stat_pixcount.nc")

## Extract future TRT Ranks (target variable) and calculate
## Rank difference to t0.
ds_TRTrank_future = ds["TRT_Rank"].where(ds["time_delta"]>0, drop=True).rename("TRT_Rank")
ds_TRTrank_future = ds_TRTrank_future.sel(time_delta=slice(5,45)) - \
                    ds["TRT_Rank"].sel(time_delta=0)

## Extract pixel counts of Radar variables with "nonmin" statistics:
ds_pixc_radar = ds[[var[:-12]+u"_pixc" for var in ds.data_vars if "nonmin" in var]]
ds_pixc_radar = ds_pixc_radar.sel(pixel_count="PC_NONMIN").drop("pixel_count").where(ds_pixc_radar["time_delta"]<=0, drop=True).astype(np.int16)
                    
## Delete unwanted variables (e.g. pixel counts):
drop_list = [var for var in ds.data_vars if "_pixc" in var]
drop_list += [u"TRT_Rank",u"TRT_Rank_diff",u"TRT_domain_indices",
              u"pixel_count",u"TRT_cellcentre_indices","date"]
ds_drop = ds.drop(drop_list)

## Extract TRT variables and solar time variables:
ds_1d   = ds_drop[[var for var in ds_drop.data_vars if len(ds_drop[var].shape)<2]]

## Delete future values (time_delta > 0) and calculate absolute difference
## between statistics at t0 and time_delta < 0:
ds_23d  = ds_drop[[var for var in ds_drop.data_vars if len(ds_drop[var].shape)>=2]]
del(ds_drop)
ds_past = ds_23d.where(ds_23d["time_delta"]<=0, drop=True)
del(ds_23d)

cfg_set, cfg_var, cfg_var_combi = cfg.get_config_info_op()
for var in ds_past.data_vars:
    if len(ds_past[var].sel(time_delta=0).values.shape)==1:
        sub_val = ds_past[var].sel(time_delta=slice(-45,-5)).values-ds_past[var].sel(time_delta=0).values[:,np.newaxis]
        ds_past[var].values = np.concatenate([sub_val,ds_past[var].sel(time_delta=0).values[:,np.newaxis]],axis=1)
    else:
        sub_val = ds_past[var].sel(time_delta=slice(-45,-5)).values-ds_past[var].sel(time_delta=0).values[:,np.newaxis,:]
        ds_past[var].values = np.concatenate([sub_val,ds_past[var].sel(time_delta=0).values[:,np.newaxis,:]],axis=1)
        
    ## Set NaN-values in "_nonmin" statistics to min_value:
    if "_nonmin" in var:
         ds_past[var].values[np.isnan(ds_past[var].values)] = cfg_set["minval_dict"][var[:-12]]
    
    #ds_past[var].sel(time_delta=slice(-45,-5)).values = (ds_past[var].sel(time_delta=slice(-45,-5)) - ds_past[var].sel(time_delta=0)).values
#ds_past.sel(time_delta=slice(-45,-5)).values = \
#    (ds_past.sel(time_delta=slice(-45,-5)) - \
#     ds_past.sel(time_delta=0)).values

## Convert 3d dataarrays (xarray) to 2d dataframes (pandas)
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
df_3d.to_hdf("df_23km_nd.h5",key="df_3d",mode="w",complevel=0)

## Convert 2d dataarrays (xarray) to 2d dataframes (pandas)
df_list_2d = [ds_past[u'CZC_lt57dBZ'].sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in np.arange(-45,5,5)]
df_list_colnames = [u'CZC_lt57dBZ|%i|SUM' % deltime for deltime in np.arange(-45,5,5)]
for var in ds_pixc_radar.data_vars:
    df_list_2d += [ds_pixc_radar[var].sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in np.arange(-45,5,5)]
    df_list_colnames += [u'%s_NONMIN|%i|SUM' % (var,deltime) for deltime in np.arange(-45,5,5)]
df_2d = pd.concat(df_list_2d,axis=1,copy=False)
df_2d.columns = df_list_colnames
df_2d = df_2d.astype(np.int16)
df_2d.to_hdf("df_23km_nd.h5",key="df_2d",mode="a",complevel=0)

df_list_TRT = [ds_TRTrank_future.sel(time_delta=deltime).drop("time_delta").to_dataframe() for deltime in np.arange(5,50,5)]
df_TRT = pd.concat(df_list_TRT,axis=1,copy=False)
df_TRT.columns = [u'TRT_Rank|%i' % deltime for deltime in np.arange(5,50,5)]
df_TRT.to_hdf("df_23km_nd.h5",key="df_TRT",mode="a",complevel=0)

## Convert 1d dataarrays (xarray) to 2d dataframes (pandas)
df_1d = ds_1d.to_dataframe()
df_1d.to_hdf("df_23km_nd.h5",key="df_1d",mode="a",complevel=0)

## Concatenate 3d/2d/1d dataframes:
df = pd.concat([df_1d,df_2d,df_3d,df_TRT],axis=1,copy=False)
df.to_hdf("df_23km.h5",key="df",mode="w",complevel=0)





## Make analysis of how many values are missing per column:
df_nan_count_sort = df.isna().sum().sort_values(ascending=False)
#plt.step(np.arange(20)+1,df_nan_count_sort.values[:20])
df_nan_count_sort[:6].plot(drawstyle="steps", linewidth=2)
plt.grid(); plt.ylabel("NaN count"); plt.xlabel("Features (sorted)")
plt.title("Number of missing values per feature")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


## Delete rows with nan-entries:
df_nonnan = df.dropna(0,'any')
df.to_hdf("df_23km_nonnan.h5",key="df_nonnan",mode="w",complevel=0)




## Choose columsn not containing TRT
X = df_nonnan[[Xcol for Xcol in df_nonnan.columns if "TRT" not in Xcol]]
y_30 = df_nonnan[["TRT_Rank|30"]]




client = Client('scheduler-address:8786')



os.environ['KMP_DUPLICATE_LIB_OK']='True'
#model_30 = XGBRegressor(n_jobs=6,nthreads=6) #objective=reg:linear
model_30 = XGBRegressor(n_jobs=6,nthreads=6) #objective=reg:linear
model_30.fit(X, y_30, verbose=True)
with open("model_30.pkl","wb") as file: pickle.dump(model_30,file,protocol=-1)

def plot_feature_importance(model,delta_t):
    sort_ind = np.argsort(model.feature_importances_)[::-1]
    df_featimp = pd.DataFrame(np.array([X.columns[sort_ind[:500]],
                                        model.feature_importances_[sort_ind[:500]]]).T,
                              columns=["Feature","Importance"])
    df_featimp.plot(drawstyle="steps", linewidth=2)
    plt.grid(); plt.ylabel("Feature importance"); plt.xlabel("Features (sorted)")
    plt.title("Feature importance - TRT t+%imin" % delta_t)
    #plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("Feature_importance_%imin_long.pdf" % delta_t,orientation="portrait")

    fig = plt.figure(figsize = [10,10])
    ax1 = fig.add_subplot(2,1,1)
    plot_importance(model,ax1,max_num_features=20,importance_type="weight",
                    title="Feature importance (Weight) - TRT t+%imin" % delta_t)
    ax2 = fig.add_subplot(2,1,2)
    plot_importance(model,ax2,max_num_features=20,importance_type="gain",
                    title="Feature importance (Gain) - TRT t+%imin" % delta_t,show_values=False)
    #ax3 = fig.add_subplot(3,1,3)
    #plot_importance(model,ax3,max_num_features=20,importance_type="cover")
    #plt.subplots_adjust(left=0.3,right=0.95)
    plt.tight_layout()
    plt.savefig("Feature_importance_%imin.pdf" % delta_t,orientation="portrait")



"""
df = ds_past_stat[["RZC_stat","BZC_stat"]].to_dataframe()


df_past_stat = ds_past_stat["RZC_stat"].to_pandas().to_frame(filter_observations=False).T
for var in ds_past_stat:
    if "RZC" not in var: ds_past_stat = ds_past_stat.drop(var)

for
df_past_stat = ds_past_stat[var].to_pandas().to_frame(filter_observations=False).T
pd.concat([df_RZC,df_CD1],axis=1,keys=["RZC","CD1"])
"""
