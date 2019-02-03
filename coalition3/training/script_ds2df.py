# coding: utf-8
import os
import xarray as xr
import numpy as np
import pandas as pd
import pickle

from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pylab as plt

def da2df(da,key):
    print(key)
    for tdelta in np.arange(-45,0,5):
        da.sel(time_delta=tdelta).values = da.sel(time_delta=0).values - \
                                           da.sel(time_delta=tdelta).values
    return da.to_pandas().to_frame(filter_observations=False).T

ds = xr.open_mfdataset("output_data/Combined_stat_pixcount.nc",
                       chunks={"DATE_TRT_ID": 10000})
ds = xr.open_dataset("output_data/Combined_stat_pixcount.nc")
ds_TRTrank = ds["TRT_Rank"]
ds_TRTrank_10min = ds["TRT_Rank"].sel(time_delta=10)
ds_past = ds.where(ds["time_delta"]<=0, drop=True)
ds_past = ds_past.drop("TRT_Rank_diff")
for var in ds_past:
    if "_pixc" in var: ds_past = ds_past.drop(var)
ds_past = ds_past.drop("pixel_count")
ds_past = ds_past.drop("TRT_domain_indices")
ds_past = ds_past.drop("TRT_cellcentre_indices")

ds_past_stat = ds_past
for var in ds_past:
    if "_stat" not in var: ds_past_stat = ds_past_stat.drop(var)

df_list = [da2df(ds_past_stat[da],da) for da in ds_past_stat]
df_all = pd.concat(df_list,axis=1,
                   keys=[da for da in ds_past_stat],copy=False)
#df_all.columns = df_all.columns.map('.'.join)
df_all.columns.set_levels(df_all.columns.levels[2].values.astype(np.unicode),
                          level=2,inplace=True)
df_all.columns.rename("Variable", level=0, inplace=True)
df_all.columns = df_all.columns.map('{0[0]}|{0[1]}|{0[2]}'.format)
df_all.index = df_all.index.astype(np.unicode)

df_TRT10 = ds_TRTrank_10min.drop("time_delta").to_dataframe()
df_TRT10.index = df_TRT10.index.astype(np.unicode)

df = pd.concat([df_all, df_TRT10],axis=1,join="inner").dropna(0,'any')
df.to_pickle("output_data/Combined_stat_pixcount_df_diff.pkl")

X = df.loc[:, df.columns != "TRT_Rank"]
y = df["TRT_Rank"]


os.environ['KMP_DUPLICATE_LIB_OK']='True'
model = XGBRegressor(n_jobs=6) #objective=reg:linear
model.fit(X, y)

sort_ind = np.argsort(model.feature_importances_)
X.columns[sort_ind[-10:]]

plot_importance(model)
plt.show()



"""
df = ds_past_stat[["RZC_stat","BZC_stat"]].to_dataframe()


df_past_stat = ds_past_stat["RZC_stat"].to_pandas().to_frame(filter_observations=False).T
for var in ds_past_stat:
    if "RZC" not in var: ds_past_stat = ds_past_stat.drop(var)

for
df_past_stat = ds_past_stat[var].to_pandas().to_frame(filter_observations=False).T
pd.concat([df_RZC,df_CD1],axis=1,keys=["RZC","CD1"])
"""
