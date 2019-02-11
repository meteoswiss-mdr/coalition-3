# coding: utf-8
""" [COALITION3] This script contains code for some EDA of the training df (Pandas)"""
    
## Import packages and define functions:
import os
import sys

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime as dt
import dask.dataframe as dd
import matplotlib.pylab as plt

import coalition3.inout.paths as pth

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
    xgb.plot_importance(model,ax1,max_num_features=20,importance_type="weight",
                        title="Feature importance (Weight) - TRT t+%imin" % delta_t)
    ax2 = fig.add_subplot(2,1,2)
    xgb.plot_importance(model,ax2,max_num_features=20,importance_type="gain",
                        title="Feature importance (Gain) - TRT t+%imin" % delta_t,show_values=False)
    #ax3 = fig.add_subplot(3,1,3)
    #plot_importance(model,ax3,max_num_features=20,importance_type="cover")
    #plt.subplots_adjust(left=0.3,right=0.95)
    plt.tight_layout()
    plt.savefig("Feature_importance_%imin.pdf" % delta_t,orientation="portrait")


## ============================================================================
## Open pandas training dataframe:
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_df = pth.file_path_reader("pandas training dataframe",user_argv_path)
df = pd.read_hdf(path_to_df,key="df")

## Make analysis of how many values are missing per column:
"""
df_nan_count_sort = df.isna().sum().sort_values(ascending=False)
df_nan_count_sort[:6].plot(drawstyle="steps", linewidth=2)
plt.grid(); plt.ylabel("NaN count"); plt.xlabel("Features (sorted)")
plt.title("Number of missing values per feature")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
"""

## Analyse pearson correlation between features (VERY TIME CONSUMING!):
"""
print("Start calculating Pearson correlation")
d_start = dt.datetime.now()
df_pears_corr_feat = df.corr()
print("  Elapsed time for calculating Pearson correlation: %s" % (dt.datetime.now()-d_start))
save_path = "%s_pcorr.h5" % os.path.splitext(path_to_df)[0]
df_pears_corr_feat.to_hdf(save_path,key="pearson_corr",mode="w",complevel=0)
del(df_pears_corr_feat)

## [Never conducted spearmans rank correlation calculation]
print("Start calculating Spearmans rank correlation")
d_start = dt.datetime.now()
df_rank_corr_feat = df.corr(method="spearman")
print("  Elapsed time for calculating rank correlation: %s" % (dt.datetime.now()-d_start))
df_rank_corr_feat.to_hdf(save_path,key="rank_corr",mode="w",complevel=0)
del(df_rank_corr_feat)

df_pears_corr_feat = pd.read_hdf("%s_pcorr.h5" % os.path.splitext(path_to_df)[0],key="pearson_corr")
percentage_corr = [(df_pears_corr_feat>corr_thr).sum().sum()/((10099**2)/2.) for corr_thr in np.arange(1,0.1,-0.025)]
fig = plt.figure(figsize = [8,6])
ax1 = fig.add_subplot(1,1,1)
ax1.plot(np.arange(1,0.1,-0.025)[:35],percentage_corr[:35],"b-")
ax1.plot(np.arange(1,0.1,-0.025)[34:],percentage_corr[34:],"b--")
ax1.set_title("Feature correlation")
ax1.set_xlabel(r"Pearson correlation coefficient $\rho$")
ax1.set_ylabel(r"Fraction of feature pairs with correlation $\rho_{pair} \geq \rho$")
plt.gca().invert_xaxis()
vals = ax1.get_yticks()
ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.grid()
plt.show()
"""

## Delete rows with nan-entries:
df_nonnan = df.dropna(0,'any')
df_nonnan.to_hdf("df_23km_nonnan.h5",key="df_nonnan",mode="w",complevel=0)

## Choose columns not containing TRT
X = df_nonnan[[Xcol for Xcol in df_nonnan.columns if "TRT" not in Xcol]]
y_10 = df_nonnan[["TRT_Rank|10"]]
y_30 = df_nonnan[["TRT_Rank|30"]]

#os.environ['KMP_DUPLICATE_LIB_OK']='True'
model_10 = xgb.XGBRegressor(max_depth=10,silent=False,n_jobs=6,nthreads=6) #objective=reg:linear
d_start = dt.datetime.now()
model_10.fit(X, y_10, verbose=True)
print("  Elapsed time for XGBoost model fitting: %s" % (dt.datetime.now()-d_start))
with open("model_10_t0diff_maxdepth10.pkl","wb") as file: pickle.dump(model_10,file,protocol=-1)
plot_feature_importance(model_10,10)

model_30 = xgb.XGBRegressor(max_depth=10,silent=False,n_jobs=6,nthreads=6) #objective=reg:linear
d_start = dt.datetime.now()
model_30.fit(X, y_30)
print("  Elapsed time for XGBoost model fitting: %s" % (dt.datetime.now()-d_start))
with open("model_30_t0diff_maxdepth10.pkl","wb") as file: pickle.dump(model_30,file,protocol=-1)
plot_feature_importance(model_30,30)
