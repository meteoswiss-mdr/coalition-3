# coding: utf-8
""" [COALITION3] This script contains code for some EDA of the training df (Pandas)"""
    
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
import dask.dataframe as dd
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
figure_path = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/statistical_learning/feature_selection/plots"
model_path = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/statistical_learning/feature_selection/models/diam_23km/"

## Open pandas training dataframe:
"""
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_df = pth.file_path_reader("pandas training dataframe",user_argv_path)
df = pd.read_hdf(path_to_df,key="df")
"""
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

"""
## Delete rows with nan-entries:
df_nonnan = df.dropna(0,'any')
df_nonnan.to_hdf("df_23km_nonnan.h5",key="df_nonnan",mode="w",complevel=0)
"""
df_nonnan = pd.read_hdf("df_23km_nonnan.h5","df_nonnan")

## Choose columns not containing TRT
X = df_nonnan[[Xcol for Xcol in df_nonnan.columns if "TRT" not in Xcol]]
y_10 = df_nonnan[["TRT_Rank|10"]]
y_30 = df_nonnan[["TRT_Rank|30"]]

## Plot histogram of Rank changes:
fig = plt.figure(figsize = [10,5])
plt.hist([y_10.values,y_30.values],
         bins=50,
         color=['#E69F00','#56B4E9'],
         label=['10min Rank difference', '30min Rank difference'])
plt.grid()
plt.show()

fig = plt.figure(figsize = [10,5])
axes = fig.add_subplot(1,1,1)
sns.kdeplot(y_10.values[:,0], shade=True, kernel="gau", bw=0.03, color='#E69F00', label='10min Rank difference')
sns.kdeplot(y_30.values[:,0], shade=True, kernel="gau", bw=0.03, color='#56B4E9', label='30min Rank difference')
plt.xlabel("TRT Rank difference")
plt.title("Kerndel density estimation of TRT Rank difference")
plt.grid()
axes.get_yaxis().set_visible(False)
plt.savefig(os.path.join(figure_path,"KDE_TRT_Rank_diff.pdf"), orientation="portrait")

print("Count of absolute TRT Rank differences > 0.2 after 10min: %5i (%2d.1%%)" % (np.sum(np.abs(y_10.values)>0.2), 100.*np.sum(np.abs(y_10.values)>0.2)/len(y_10)))
print("Count of absolute TRT Rank differences > 0.5 after 10min: %5i (%2d.1%%)" % (np.sum(np.abs(y_10.values)>0.5), 100.*np.sum(np.abs(y_10.values)>0.5)/len(y_10)))
print("Count of absolute TRT Rank differences > 0.2 after 30min: %5i (%2d.1%%)" % (np.sum(np.abs(y_30.values)>0.2), 100.*np.sum(np.abs(y_30.values)>0.2)/len(y_30)))
print("Count of absolute TRT Rank differences > 0.5 after 30min: %5i (%2d.1%%)" % (np.sum(np.abs(y_30.values)>0.5), 100.*np.sum(np.abs(y_30.values)>0.5)/len(y_30)))

    
"""
## Fit XGBoost Regressor model
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
with open("model_30_t0diff_maxdepth10_TRTdiff02.pkl","wb") as file: pickle.dump(model_30,file,protocol=-1)
plot_feature_importance(model_30,30)
"""

with open(os.path.join(model_path,"model_10_t0diff_maxdepth10.pkl"),"rb") as file: xgb_model_10 = pickle.load(file)
with open(os.path.join(model_path,"model_30_t0diff_maxdepth10.pkl"),"rb") as file: xgb_model_30 = pickle.load(file)

## Extract feature importance measures and sort in descending order:
top_features_weight_10 = pd.DataFrame.from_dict(xgb_model_10.get_booster().get_score(importance_type='weight'),orient="index",columns=["F_score"]).sort_values(by=['F_score'], ascending=False)
top_features_gain_10   = pd.DataFrame.from_dict(xgb_model_10.get_booster().get_score(importance_type='gain'),orient="index",columns=["F_score"]).sort_values(by=['F_score'], ascending=False)
top_features_weight_30 = pd.DataFrame.from_dict(xgb_model_30.get_booster().get_score(importance_type='weight'),orient="index",columns=["F_score"]).sort_values(by=['F_score'], ascending=False)
top_features_gain_30   = pd.DataFrame.from_dict(xgb_model_30.get_booster().get_score(importance_type='gain'),orient="index",columns=["F_score"]).sort_values(by=['F_score'], ascending=False)

## Split into training and testing dataset:
X_train, X_test, y_train, y_test = train_test_split(X, pd.concat([y_10, y_30], axis=1), test_size=0.2, random_state=42)

## Create list of number of features to select for the fitting:
n_feat_arr = np.concatenate([np.arange(10)+1,
                             np.arange(12,50,2),
                             np.arange(50,100,10),
                             np.arange(100,520,20)])#,
                             #np.arange(500,2000,50),
                             #np.arange(2000,5000,100),
                             #np.arange(5000,11000,1000)])

pred_gain_10_ls = [mse_n_feat(X_train, X_test, y_train[["TRT_Rank|10"]], y_test[["TRT_Rank|10"]], top_features_gain_10, n_feat) for n_feat in n_feat_arr]
pred_gain_30_ls = [mse_n_feat(X_train, X_test, y_train[["TRT_Rank|30"]], y_test[["TRT_Rank|30"]], top_features_gain_30, n_feat) for n_feat in n_feat_arr]

df_mse_feat_count = pd.DataFrame({"Feature Count": n_feat_arr,
                                  "MSE 10min": pred_gain_10_ls,
                                  "MSE 30min": pred_gain_30_ls})
with open(os.path.join(model_path,"MSE_feature_count_gain.pkl"),"wb") as file: pickle.dump(df_mse_feat_count,file,protocol=-1)
with open(os.path.join(model_path,"MSE_feature_count_gain.pkl"),"rb") as file: df_mse_feat_count = pickle.load(file)

fig = plt.figure(figsize = [10,5])
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(n_feat_arr,df_mse_feat_count[["MSE 10min"]], '#E69F00', label='10min')
ax2.plot(n_feat_arr,df_mse_feat_count[["MSE 30min"]], '#56B4E9', label='30min')
#ax1.semilogx(n_feat_arr,df_mse_feat_count[["MSE 10min"]], '#E69F00', label='10min')
#ax2.semilogx(n_feat_arr,df_mse_feat_count[["MSE 30min"]], '#56B4E9', label='30min')
#df_mse_feat_count.plot(ax=ax1,x="Feature Count",y="MSE 10min", style='b-', secondary_y=False)
#df_mse_feat_count.plot(ax=ax2,x="Feature Count",y="MSE 30min", style='g-', secondary_y=True)
ax1.set_title("Mean square error (MSE) as function of feature count")
ax1.set_xlabel(r"Number of features")
ax1.set_ylabel(r"MSE - 10min prediction", color='#E69F00')
ax2.set_ylabel(r"MSE - 30min prediction", color='#56B4E9')
ax1.grid()
#ax2.legend([ax1.get_lines()[0], ax2.right_ax.get_lines()[0]], ['A','B'], bbox_to_anchor=(1.5, 0.5))
#plt.show()
plt.savefig(os.path.join(figure_path,"MSE_feature_count.pdf"), orientation="portrait")


## Make new model with just the 200 "best" features:
model_weight_10 = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)
model_weight_10.fit(X_train[top_features_weight_10.index[:200]], y_train[["TRT_Rank|10"]])
with open(os.path.join(model_path,"model_10_t0diff_maxdepth6_200feat_weight.pkl"),"wb") as file: pickle.dump(model_weight_10,file,protocol=-1)
model_gain_10   = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)
model_gain_10.fit(X_train[top_features_gain_10.index[:200]], y_train[["TRT_Rank|10"]])
with open(os.path.join(model_path,"model_10_t0diff_maxdepth6_200feat_gain.pkl"),"wb") as file: pickle.dump(model_gain_10,file,protocol=-1)
model_weight_30 = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)
model_weight_30.fit(X_train[top_features_weight_30.index[:200]], y_train[["TRT_Rank|30"]])
with open(os.path.join(model_path,"model_30_t0diff_maxdepth6_200feat_weight.pkl"),"wb") as file: pickle.dump(model_weight_30,file,protocol=-1)
model_gain_30   = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)
model_gain_30.fit(X_train[top_features_gain_30.index[:200]], y_train[["TRT_Rank|30"]])
with open(os.path.join(model_path,"model_30_t0diff_maxdepth6_200feat_gain.pkl"),"wb") as file: pickle.dump(model_gain_30,file,protocol=-1)

with open(os.path.join(model_path,"model_30_t0diff_maxdepth10_200feat_gain.pkl"),"rb") as file: model_gain_30 = pickle.load(file)


## Predict using the top 200 features in each model:
pred_weight_10 = model_weight_10.predict(X_test[top_features_weight_10.index[:200]])
pred_gain_10   = model_gain_10.predict(X_test[top_features_gain_10.index[:200]])
pred_weight_30 = model_weight_30.predict(X_test[top_features_weight_30.index[:200]])
pred_gain_30   = model_gain_30.predict(X_test[top_features_gain_30.index[:200]])

## Calculate the respective root mean squared error:
mse_weight_10 = mean_squared_error(y_test[["TRT_Rank|10"]],pred_weight_10)
mse_gain_10   = mean_squared_error(y_test[["TRT_Rank|10"]],pred_gain_10)
mse_weight_30 = mean_squared_error(y_test[["TRT_Rank|30"]],pred_weight_30)
mse_gain_30   = mean_squared_error(y_test[["TRT_Rank|30"]],pred_gain_30)

## Plot 2D histogram of predicted and observed TRT Rank changes:
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[10,8])
axes.set_ylabel('Predicted TRT Rank difference')
counts,ybins,xbins,image = axes.hist2d(y_test[["TRT_Rank|30"]].values[:,0],pred_gain_30,
                                       bins=200,range=[[-1.5,1.5],[-1.5,1.5]],cmap="magma",norm=mcolors.LogNorm())
cbar = fig.colorbar(image, ax=axes, extend='max')
#cbar.set_label('Number of cells per bin of size [0.02, 0.02]', rotation=90)
cont2d_1, lvl = contour_of_2dHist(counts,smooth=True)
axes.grid()
axes.fill_between([-0.2,0.2],y1=[-1.5,-1.5], y2=[1.5,1.5], facecolor="none", hatch="X", edgecolor="darkred", linewidth=0.5)
axes.plot([-1.5,1.5],[-1.5,1.5],'w--',linewidth=2) #,facecolor="w",linewidth=2,linestyle='--')
CS = axes.contour(cont2d_1,levels=lvl,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=2,cmap="YlGn_r")
CS_lab = axes.clabel(CS, inline=1, fontsize=10, fmt='%i%%', colors="black")
#[txt.set_backgroundcolor('white') for txt in CS_lab]
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.3, boxstyle='round', alpha=0.7)) for txt in CS_lab] #pad=0,
axes.set_xlabel('Observed TRT Rank difference'); axes.set_title('TRT Ranks differences (23km diameter)')
axes.set_aspect('equal'); axes.patch.set_facecolor('0.7')
str_n_cells = "Total number of cells = %i" % np.sum(counts)
axes.text(0.4,3.6,str_n_cells)
plt.show()















