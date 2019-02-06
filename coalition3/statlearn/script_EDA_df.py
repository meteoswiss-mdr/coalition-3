# coding: utf-8
""" [COALITION3] This script contains code for some EDA of the training df (Pandas)"""
    
## Import packages and define functions:
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import datetime as dt
import dask.dataframe as dd
import matplotlib.pylab as plt

import coalition3.inout.paths as pth

## ============================================================================
## Open pandas training dataframe:
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_df = pth.file_path_reader("pandas training dataframe",user_argv_path)
df = pd.read_hdf(path_to_df,key="df")
"""
## Make analysis of how many values are missing per column:
df_nan_count_sort = df.isna().sum().sort_values(ascending=False)
df_nan_count_sort[:6].plot(drawstyle="steps", linewidth=2)
plt.grid(); plt.ylabel("NaN count"); plt.xlabel("Features (sorted)")
plt.title("Number of missing values per feature")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
"""
## Analyse pearson correlation between features (VERY TIME CONSUMING!):
print("Start calculating Pearson correlation")
d_start = dt.datetime.now()
df_pears_corr_feat = df.corr()
print("  Elapsed time for calculating Pearson correlation: %s" % (dt.datetime.now()-d_start))
save_path = "%s_pcorr.h5" % os.path.splitext(path_to_df)[0]
df_pears_corr_feat.to_hdf(save_path,key="pearson_corr",mode="w",complevel=0)
del(df_pears_corr_feat)

print("Start calculating Spearmans rank correlation")
d_start = dt.datetime.now()
df_rank_corr_feat = df.corr(method="spearman")
print("  Elapsed time for calculating rank correlation: %s" % (dt.datetime.now()-d_start))
df_rank_corr_feat.to_hdf(save_path,key="rank_corr",mode="w",complevel=0)
del(df_rank_corr_feat)

"""
percentage_corr = [(df_pears_corr_feat>corr_thr).sum().sum()/((10099**2)/2.) for corr_thr in np.arange(1,0.1,-0.025)]
"""











"""
## Delete rows with nan-entries:
df_nonnan = df.dropna(0,'any')
df.to_hdf("df_23km_nonnan.h5",key="df_nonnan",mode="w",complevel=0)

## Choose columns not containing TRT
X = df_nonnan[[Xcol for Xcol in df_nonnan.columns if "TRT" not in Xcol]]
y_30 = df_nonnan[["TRT_Rank|30"]]


os.environ['KMP_DUPLICATE_LIB_OK']='True'
#model_30 = XGBRegressor(n_jobs=6,nthreads=6) #objective=reg:linear
model_30 = xgb.XGBRegressor(n_jobs=6,nthreads=6) #objective=reg:linear
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
"""
