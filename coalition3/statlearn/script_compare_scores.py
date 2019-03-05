# coding: utf-8
import sys
import os
import pickle
import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pylab as plt

import coalition3.inout.readconfig as cfg
import coalition3.statlearn.feature as ftr
import coalition3.statlearn.inputprep as ipt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

pred_dt = 30
cfg_tds = cfg.get_config_info_tds()

ls_models_xgb = pickle.load(open("../MeteoSwiss_Data/all_samples/models_30_t0diff_maxdepth6_nfeat.pkl","rb"))
ls_models_mlp = pickle.load(open("mlp_selfeat_ls.pkl","rb"))
features = ls_models_xgb[-1].get_booster().feature_names
xgb_500 = ls_models_xgb[-1]
mlp_500 = ls_models_mlp[-1].best_estimator_


df = pd.read_hdf("Combined_stat_pixcount_df_t0diff_nonnan.h5")
n_feat_arr = np.concatenate([np.arange(10)+1,
                             np.arange(12,50,2),
                             np.arange(50,100,10),
                             np.arange(100,520,20)])

xgb_model = pickle.load(open("../MeteoSwiss_Data/all_samples/model_30_t0diff_maxdepth6.pkl","rb"))
top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                               orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                               ascending=False)

X_train, X_test, y_train, y_test = ipt.get_model_input(df, del_TRTeqZero_tpred=True, split_Xy_traintest=True, pred_dt = pred_dt, X_normalise=False)
pred_xgb_500 = xgb_500.predict(X_test[features])
mse_xgb_500  = sklearn.metrics.mean_squared_error(y_test, pred_xgb_500)
r2_xgb_500   = sklearn.metrics.r2_score(y_test, pred_xgb_500)
ftr.plot_pred_vs_obs_core(y_test,pred_xgb_500,pred_dt,mse_xgb_500,r2_xgb_500,"_xgb500",cfg_tds)

X_train, X_test, y_train, y_test, scaler = ipt.get_model_input(df, del_TRTeqZero_tpred=True, split_Xy_traintest=True, pred_dt = pred_dt, X_normalise=True)
pred_mlp_500 = mlp_500.predict(X_test[features])
mse_mlp_500  = sklearn.metrics.mean_squared_error(y_test, pred_mlp_500)
r2_mlp_500   = sklearn.metrics.r2_score(y_test, pred_mlp_500)
ftr.plot_pred_vs_obs_core(y_test,pred_mlp_500,pred_dt,mse_mlp_500,r2_mlp_500,"_mlp500",cfg_tds)

MSE_r2_ls_xgb = [ftr.mse_r2_n_feat(X_test, y_test, top_features_gain, n_feat, model) for n_feat, model in zip(n_feat_arr[9:],ls_models_xgb[9:])]
MSE_r2_ls_mlp = [ftr.mse_r2_n_feat(X_test, y_test, top_features_gain, n_feat, model) for n_feat, model in zip(n_feat_arr[9:],ls_models_mlp)]
df_mse_r2_feat_count_mlp = pd.DataFrame.from_dict({"Feature Count": n_feat_arr[9:],
    "MSE %imin%s" % (pred_dt,"_mlp"): [score[0] for score in MSE_r2_ls_mlp],
     "R2 %imin%s" % (pred_dt,"_mlp"): [score[1] for score in MSE_r2_ls_mlp]}).set_index("Feature Count")
df_mse_r2_feat_count_xgb = pd.DataFrame.from_dict({"Feature Count": n_feat_arr[9:],
             "MSE %imin%s" % (pred_dt,"_xgb"): [score[0] for score in MSE_r2_ls_xgb],
              "R2 %imin%s" % (pred_dt,"_xgb"): [score[1] for score in MSE_r2_ls_xgb]}).set_index("Feature Count")
df_mse_r2_feat_count = pd.concat([df_mse_r2_feat_count_mlp,df_mse_r2_feat_count_xgb],
                                 axis=1)
df_mse_r2_feat_count.columns = [colname.replace("_"," (")+")" for colname in df_mse_r2_feat_count.columns]

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
plt.show()
