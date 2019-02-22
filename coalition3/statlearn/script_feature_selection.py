# coding: utf-8
""" [COALITION3] This script contains code for the selection of features using XGBoost"""
    
## Import packages and define functions:
from __future__ import division
from __future__ import print_function

import os
import sys

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import datetime as dt
import matplotlib.pylab as plt
import matplotlib.colors as mcolors

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import coalition3.inout.paths as pth
from coalition3.visualisation.TRTcells import contour_of_2dHist

def mse_n_feat(X_train, X_test, y_train, y_test, top_features, n_feat):
    print("  Working on %3i features" % n_feat, end="")
    sys.stdout.flush()
    t0 = dt.datetime.now()
    model = xgb.XGBRegressor(max_depth=6,silent=True,n_jobs=6,nthreads=6)
    model.fit(X_train[top_features.index[:n_feat]], y_train)
    print(" (Time needed: %s)" % (dt.datetime.now()-t0), end="\r")
    sys.stdout.flush()
    return(mean_squared_error(y_test, model.predict(X_test[top_features.index[:n_feat]])))
    
def plot_feature_importance(model,X,delta_t):
    sort_ind = np.argsort(model.feature_importances_)[::-1]
    df_featimp = pd.DataFrame(np.array([X.columns[sort_ind[:500]],
                                        model.feature_importances_[sort_ind[:500]]]).T,
                              columns=["Feature","Importance"])
    df_featimp.plot(drawstyle="steps", linewidth=2)
    plt.grid(); plt.ylabel("Feature importance"); plt.xlabel("Features (sorted)")
    plt.title("Feature importance - TRT t+%imin" % delta_t)
    #plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path,"Feature_importance_%imin_long.pdf" % delta_t),orientation="portrait")

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
    plt.savefig(os.path.join(figure_path,"Feature_importance_%imin.pdf" % delta_t),orientation="portrait")

def get_X_col(colname):
    ## No future TRT observations in input:
    use_col = ("TRT" not in colname or \
               "-" in colname or "|0" in colname) and \
              ("TRT_Rank_diff" not in colname)
    return(use_col)
    
## ============================================================================
figure_path = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/statistical_learning/feature_selection/plots/diam_23km/"
model_path = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/statistical_learning/feature_selection/models/diam_23km/"
col10 = '#E69F00'
col30 = '#D55E00'

"""
GET PATH TO NONNAN FROM USER INPUT
GET DIAM FROM USER INPUT
"""

## Open pandas training dataframe:
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_df = pth.file_path_reader("pandas training dataframe",user_argv_path)
df = pd.read_hdf(path_to_df,key="df")


df_nonnan = pd.read_hdf("df_23km_nonnan.h5","df_nonnan")
df_nonnan_nonzerot0t10 = df_nonnan.loc[(df_nonnan["TRT_Rank|10"]>=0.15) & (df_nonnan["TRT_Rank|0"]>=0.15)]
df_nonnan_nonzerot0t30 = df_nonnan.loc[(df_nonnan["TRT_Rank|30"]>=0.15) & (df_nonnan["TRT_Rank|0"]>=0.15)]
del(df_nonnan)
    
    
"""
## Fit XGBoost Regressor model
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
xgb_model_10 = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6) #objective=reg:linear
X_10 = df_nonnan_nonzerot0t10[[Xcol for Xcol in df_nonnan_nonzerot0t10.columns if get_X_col(Xcol)]]
y_10 = df_nonnan_nonzerot0t10[["TRT_Rank_diff|10"]]
del(df_nonnan_nonzerot0t10)

d_start = dt.datetime.now()
xgb_model_10.fit(X_10, y_10, verbose=True)
print("  Elapsed time for XGBoost model fitting: %s" % (dt.datetime.now()-d_start))
with open(os.path.join(model_path,"model_10_t0diff_maxdepth6.pkl"),"wb") as file: pickle.dump(xgb_model_10,file,protocol=-1)
plot_feature_importance(xgb_model_10,X_10,10)

xgb_model_30 = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6) #objective=reg:linear
X_30 = df_nonnan_nonzerot0t30[[Xcol for Xcol in df_nonnan_nonzerot0t30.columns if get_X_col(Xcol)]]
y_30 = df_nonnan_nonzerot0t30[["TRT_Rank_diff|30"]]
del(df_nonnan_nonzerot0t30)
d_start = dt.datetime.now()
xgb_model_30.fit(X_30, y_30)
print("  Elapsed time for XGBoost model fitting: %s" % (dt.datetime.now()-d_start))
with open(os.path.join(model_path,"model_30_t0diff_maxdepth6.pkl"),"wb") as file: pickle.dump(xgb_model_30,file,protocol=-1)
plot_feature_importance(xgb_model_30,X_30,30)
"""

with open(os.path.join(model_path,"model_10_t0diff_maxdepth6.pkl"),"rb") as file: xgb_model_10 = pickle.load(file)
with open(os.path.join(model_path,"model_30_t0diff_maxdepth6.pkl"),"rb") as file: xgb_model_30 = pickle.load(file)

## Extract feature importance measures and sort in descending order:
top_features_weight_10 = pd.DataFrame.from_dict(xgb_model_10.get_booster().get_score(importance_type='weight'),orient="index",columns=["F_score"]).sort_values(by=['F_score'], ascending=False)
top_features_gain_10   = pd.DataFrame.from_dict(xgb_model_10.get_booster().get_score(importance_type='gain'),orient="index",columns=["F_score"]).sort_values(by=['F_score'], ascending=False)
top_features_weight_30 = pd.DataFrame.from_dict(xgb_model_30.get_booster().get_score(importance_type='weight'),orient="index",columns=["F_score"]).sort_values(by=['F_score'], ascending=False)
top_features_gain_30   = pd.DataFrame.from_dict(xgb_model_30.get_booster().get_score(importance_type='gain'),orient="index",columns=["F_score"]).sort_values(by=['F_score'], ascending=False)

## Split into training and testing dataset:
#X_train, X_test, y_train, y_test = train_test_split(X, pd.concat([y_10, y_30], axis=1), test_size=0.2, random_state=42)
X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(X_10, y_10, test_size=0.2, random_state=42)
X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(X_30, y_30, test_size=0.2, random_state=42)

## Create list of number of features to select for the fitting:
n_feat_arr = np.concatenate([np.arange(10)+1,
                             np.arange(12,50,2),
                             np.arange(50,100,10),
                             np.arange(100,520,20)])#,
                             #np.arange(500,2000,50),
                             #np.arange(2000,5000,100),
                             #np.arange(5000,11000,1000)])

pred_gain_10_ls = [mse_n_feat(X_train, X_test, y_train[["TRT_Rank_diff|10"]], y_test[["TRT_Rank_diff|10"]], top_features_gain_10, n_feat) for n_feat in n_feat_arr]
pred_gain_30_ls = [mse_n_feat(X_train, X_test, y_train[["TRT_Rank_diff|30"]], y_test[["TRT_Rank_diff|30"]], top_features_gain_30, n_feat) for n_feat in n_feat_arr]

df_mse_feat_count = pd.DataFrame({"Feature Count": n_feat_arr,
                                  #"MSE 10min": pred_gain_10_ls}) #,
                                  "MSE 30min": pred_gain_30_ls})
with open(os.path.join(model_path,"MSE_feature_count_gain.pkl"),"wb") as file: pickle.dump(df_mse_feat_count,file,protocol=-1)
with open(os.path.join(model_path,"MSE_feature_count_gain.pkl"),"rb") as file: df_mse_feat_count = pickle.load(file)

fig = plt.figure(figsize = [10,5])
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(n_feat_arr,df_mse_feat_count[["MSE 10min"]], col10, label='10min')
ax2.plot(n_feat_arr,df_mse_feat_count[["MSE 30min"]], col30, label='30min')
#ax1.semilogx(n_feat_arr,df_mse_feat_count[["MSE 10min"]], '#E69F00', label='10min')
#ax2.semilogx(n_feat_arr,df_mse_feat_count[["MSE 30min"]], '#56B4E9', label='30min')
#df_mse_feat_count.plot(ax=ax1,x="Feature Count",y="MSE 10min", style='b-', secondary_y=False)
#df_mse_feat_count.plot(ax=ax2,x="Feature Count",y="MSE 30min", style='g-', secondary_y=True)
ax1.set_title("Mean square error (MSE) as function of feature count")
ax1.set_xlabel(r"Number of features")
ax1.set_ylabel(r"MSE - 10min prediction", color=col10)
ax2.set_ylabel(r"MSE - 30min prediction", color=col30)
ax1.grid()
#ax2.legend([ax1.get_lines()[0], ax2.right_ax.get_lines()[0]], ['A','B'], bbox_to_anchor=(1.5, 0.5))
#plt.show()
plt.savefig(os.path.join(figure_path,"MSE_feature_count.pdf"), orientation="portrait")


## Make new model with just the 200 "best" features:
model_weight_10 = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)
model_weight_10.fit(X_train_10[top_features_weight_10.index[:200]], y_train_10[["TRT_Rank_diff|10"]])
with open(os.path.join(model_path,"model_10_t0diff_maxdepth6_200feat_weight.pkl"),"wb") as file: pickle.dump(model_weight_10,file,protocol=-1)
model_gain_10   = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)
model_gain_10.fit(X_train_10[top_features_gain_10.index[:200]], y_train_10[["TRT_Rank_diff|10"]])
with open(os.path.join(model_path,"model_10_t0diff_maxdepth6_200feat_gain.pkl"),"wb") as file: pickle.dump(model_gain_10,file,protocol=-1)
model_weight_30 = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)
model_weight_30.fit(X_train_30[top_features_weight_30.index[:300]], y_train_30[["TRT_Rank_diff|30"]])
with open(os.path.join(model_path,"model_30_t0diff_maxdepth6_300feat_weight.pkl"),"wb") as file: pickle.dump(model_weight_30,file,protocol=-1)
model_gain_30   = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)
model_gain_30.fit(X_train_30[top_features_gain_30.index[:300]], y_train_30[["TRT_Rank_diff|30"]])
with open(os.path.join(model_path,"model_30_t0diff_maxdepth6_300feat_gain.pkl"),"wb") as file: pickle.dump(model_gain_30,file,protocol=-1)

with open(os.path.join(model_path,"model_10_t0diff_maxdepth6_200feat_gain.pkl"),"rb") as file: model_gain_30 = pickle.load(file)
with open(os.path.join(model_path,"model_30_t0diff_maxdepth6_300feat_gain.pkl"),"rb") as file: model_gain_30 = pickle.load(file)


## Predict using the top 200 features in each model:
pred_weight_10 = model_weight_10.predict(X_test_10[top_features_weight_10.index[:200]])
pred_gain_10   = model_gain_10.predict(X_test_10[top_features_gain_10.index[:200]])
pred_weight_30 = model_weight_30.predict(X_test_30[top_features_weight_30.index[:300]])
pred_gain_30   = model_gain_30.predict(X_test_30[top_features_gain_30.index[:300]])

## Calculate the respective root mean squared error:
mse_weight_10 = mean_squared_error(y_test_10[["TRT_Rank_diff|10"]],pred_weight_10)
mse_gain_10   = mean_squared_error(y_test_10[["TRT_Rank_diff|10"]],pred_gain_10)
mse_weight_30 = mean_squared_error(y_test_30[["TRT_Rank_diff|30"]],pred_weight_30)
mse_gain_30   = mean_squared_error(y_test_30[["TRT_Rank_diff|30"]],pred_gain_30)

## Plot 2D histogram of predicted and observed TRT Rank changes:
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[10,8])
axes.set_ylabel('Predicted TRT Rank difference')
counts,ybins,xbins,image = axes.hist2d(y_test_30[["TRT_Rank_diff|30"]].values[:,0],pred_gain_30,
                                       bins=200,range=[[-2.,2.],[-2.,2.]],cmap="magma",norm=mcolors.LogNorm())
cbar = fig.colorbar(image, ax=axes, extend='max')
#cbar.set_label('Number of cells per bin of size [0.02, 0.02]', rotation=90)
cont2d_1, lvl = contour_of_2dHist(counts,smooth=True)
axes.grid()
#axes.fill_between([-0.2,0.2],y1=[-1.5,-1.5], y2=[1.5,1.5], facecolor="none", hatch="X", edgecolor="darkred", linewidth=0.5)
axes.plot([-2,2],[-2,2],'w--',linewidth=2) #,facecolor="w",linewidth=2,linestyle='--')
CS = axes.contour(cont2d_1,levels=lvl,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=2,cmap="YlGn_r")
CS_lab = axes.clabel(CS, inline=1, fontsize=10, fmt='%i%%', colors="black")
#[txt.set_backgroundcolor('white') for txt in CS_lab]
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.3, boxstyle='round', alpha=0.7)) for txt in CS_lab] #pad=0,
axes.set_xlabel(r'Observed TRT Rank difference $t_{+30min}$'); axes.set_title('TRT Ranks differences (23km diameter)\nTime delta: 30min')
axes.set_aspect('equal'); axes.patch.set_facecolor('0.7')
str_n_cells = "Total number of cells = %i" % np.sum(counts)
axes.text(0.4,3.6,str_n_cells)
plt.savefig(os.path.join(figure_path,"Pred_scatter_30.pdf"), orientation="portrait")
















