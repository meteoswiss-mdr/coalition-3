# coding: utf-8
""" [COALITION3] This script contains code for comparing the MSE & R2 scores
    between the ANN and the XGBoost Model."""
    
## Import packages and define functions:
import sys
import os
import pickle
import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pylab as plt

import coalition3.inout.paths as pth
import coalition3.inout.readconfig as cfg
import coalition3.statlearn.feature as feat
import coalition3.statlearn.inputprep as ipt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
## ============================================================================
## Get config info:
cfg_tds = cfg.get_config_info_tds()
cfg_op, __, __ = cfg.get_config_info_op()

## Load training dataframe:
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_df = pth.file_path_reader("pandas training dataframe (nonnan)",user_argv_path)
print("\nLoading nonnan dataframe into RAM")
df_nonnan  = pd.read_hdf(path_to_df,key="df_nonnan")

## Load list with models:
model_path_xgb = pth.file_path_reader("XGBoost model list")
model_path_mlp = pth.file_path_reader("MLP model list")
with open(model_path_xgb,"rb") as file: ls_models_xgb = pickle.load(file)
with open(model_path_mlp,"rb") as file: ls_models_mlp = pickle.load(file)

## Load XGBoost model fitted with all features:
#model_path_xgb_feat = pth.file_path_reader("XGBoost model fitted with all features")
#with open(model_path_xgb_feat,"rb") as file: xgb_model_feat = pickle.load(file)
#top_features_gain = pd.DataFrame.from_dict(xgb_model_feat.get_booster().get_score(importance_type='gain'),
#                                           orient="index",columns=["F_score"]).sort_values(by=['F_score'],
#                                           ascending=False)

## Get prediction leadtime from model:
pred_dt = feat.get_pred_dt_ls("the model comparison", cfg_op["timestep"],cfg_op["n_integ"])[0]

## Get features of largest models (ANN and XGB)
top_features_gain = features = ls_models_xgb[-1].get_booster().feature_names
xgb_500  = ls_models_xgb[-1]
mlp_500  = ls_models_mlp[-1].best_estimator_

## Get scores for the following number of features:
n_feat_arr = np.concatenate([np.arange(10)+1,
                             np.arange(12,50,2),
                             np.arange(50,100,10),
                             np.arange(100,520,20)])

## Get training and testing data (non-normalised for XGBoost model) and the scores:
X_train_nonnorm, X_test_nonnorm, y_train_nonnorm, \
        y_test_nonnorm = ipt.get_model_input(df_nonnan, del_TRTeqZero_tpred=True, split_Xy_traintest=True, pred_dt = pred_dt, X_normalise=False)
pred_xgb_500 = xgb_500.predict(X_test_nonnorm[features])
mse_xgb_500  = sklearn.metrics.mean_squared_error(y_test_nonnorm, pred_xgb_500)
r2_xgb_500   = sklearn.metrics.r2_score(y_test_nonnorm, pred_xgb_500)
feat.plot_pred_vs_obs_core(y_test_nonnorm,pred_xgb_500,pred_dt,mse_xgb_500,r2_xgb_500,"_xgb500",cfg_tds)
MSE_r2_ls_xgb = [feat.mse_r2_n_feat(X_test_nonnorm, y_test_nonnorm, top_features_gain, n_feat, model) for n_feat, model in zip(n_feat_arr[9:],ls_models_xgb[9:])]
del(X_train_nonnorm, X_test_nonnorm, y_train_nonnorm, y_test_nonnorm)

## Get training and testing data (normalised for ANN model) and the scores:
X_train_norm, X_test_norm, y_train_norm, \
    y_test_norm, scaler = ipt.get_model_input(df_nonnan, del_TRTeqZero_tpred=True, split_Xy_traintest=True, pred_dt = pred_dt, X_normalise=True)
pred_mlp_500 = mlp_500.predict(X_test_norm[features])
mse_mlp_500  = sklearn.metrics.mean_squared_error(y_test_norm, pred_mlp_500)
r2_mlp_500   = sklearn.metrics.r2_score(y_test_norm, pred_mlp_500)
feat.plot_pred_vs_obs_core(y_test_norm,pred_mlp_500,pred_dt,mse_mlp_500,r2_mlp_500,"_mlp500",cfg_tds)
MSE_r2_ls_mlp = [feat.mse_r2_n_feat(X_test_norm, y_test_norm, top_features_gain, n_feat, model) for n_feat, model in zip(n_feat_arr[9:],ls_models_mlp)]
del(X_train_norm, X_test_norm, y_train_norm, y_test_norm)

## Get scores into dataframe:
df_mse_r2_feat_count_mlp = pd.DataFrame.from_dict({"Feature Count": n_feat_arr[9:],
    "MSE %imin%s" % (pred_dt,"_mlp"): [score[0] for score in MSE_r2_ls_mlp],
     "R2 %imin%s" % (pred_dt,"_mlp"): [score[1] for score in MSE_r2_ls_mlp]}).set_index("Feature Count")
df_mse_r2_feat_count_xgb = pd.DataFrame.from_dict({"Feature Count": n_feat_arr[9:],
     "MSE %imin%s" % (pred_dt,"_xgb"): [score[0] for score in MSE_r2_ls_xgb],
      "R2 %imin%s" % (pred_dt,"_xgb"): [score[1] for score in MSE_r2_ls_xgb]}).set_index("Feature Count")
df_mse_r2_feat_count = pd.concat([df_mse_r2_feat_count_mlp,df_mse_r2_feat_count_xgb],
                                 axis=1)
df_mse_r2_feat_count.columns = [colname.replace("_"," (")+")" for colname in df_mse_r2_feat_count.columns]

## Plot scores side-by-side:
fig = plt.figure(figsize = [10,7])
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
df_mse_r2_feat_count.iloc[:,[0,2]].plot.line(ax=ax1)
df_mse_r2_feat_count.iloc[:,[1,3]].plot.line(ax=ax2)
for ax in [ax1, ax2]:
    ax.grid()
    ax.set_xlabel("Feature Count")
ax1.set_ylabel(r"MSE (Mean Square Error)")
ax2.set_ylabel(r"Coeff of determination R$^2$")
plt.tight_layout()
plt.tight_layout()
plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Score_comparison_ANN_XGB_%imin.pdf" % (pred_dt)),orientation="portrait")




