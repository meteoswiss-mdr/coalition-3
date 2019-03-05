# coding: utf-8
import sys
import os
import numpy as np

import pandas as pd
import xgboost as xgb

import coalition3.inout.readconfig as cfg
import coalition3.statlearn.inputprep as ipt

import sklearn.metrics as met
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

cfg_tds = cfg.get_config_info_tds()
df = pd.read_hdf("Combined_stat_pixcount_df_t0diff_nonnan.h5","df_nonnan")
pred_dt = 30
X_train, X_test, y_train, y_test, scaler = ipt.get_model_input(df,
    del_TRTeqZero_tpred=True, split_Xy_traintest=True, X_normalise=True,
    pred_dt=pred_dt)

mlp_allfeat = MLPRegressor(hidden_layer_sizes=(100,50), verbose=True)
mlp_allfeat.fit(X_train,y_train)
pickle.dump(mlp_allfeat,open("mlp_allfeat","wb"))
pred_mlp_allfeat = mlp_allfeat.predict(X_test)
mse_mlp_allfeat = met.mean_squared_error(y_test,pred_mlp_allfeat)
r2_mlp_allfeat = met.r2_score(y_test,pred_mlp_allfeat)
plot_pred_vs_obs_core(y_test,pred_mlp_allfeat,pred_dt,mse_mlp_allfeat,
                      r2_mlp_allfeat,"_mlp-allfeat",cfg_tds)

xgb_model = pickle.load(open("../MeteoSwiss_Data/all_samples/model_30_t0diff_maxdepth6_300feat_gain.pkl","rb"))
features = xgb_model.get_booster().feature_names
X_train, X_test, y_train, y_test, scaler = ipt.get_model_input(df,
    del_TRTeqZero_tpred=True, split_Xy_traintest=True, X_normalise=True,
    pred_dt=pred_dt,X_feature_sel=features))

mlp_selfeat = MLPRegressor(hidden_layer_sizes=(100,50), verbose=True)
mlp_selfeat.fit(X_train,y_train)
pickle.dump(mlp_selfeat,open("mlp_selfeat","wb"))
pred_mlp_selfeat = mlp_selfeat.predict(X_test)
mse_gain_selfeat = met.mean_squared_error(y_test,pred_mlp_selfeat)
r2_gain_selfeat = met.r2_score(y_test,pred_mlp_selfeat)
plot_pred_vs_obs_core(y_test,pred_mlp_selfeat,pred_dt,mse_gain_selfeat,r2_gain_selfeat,"_mlp-selfeat",cfg_tds)

xgb_model = pickle.load(open("../MeteoSwiss_Data/all_samples/model_30_t0diff_maxdepth6.pkl","rb"))
top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                           orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                           ascending=False)
n_feat_arr = np.concatenate([np.arange(10,50,2),
                             np.arange(50,100,10),
                             np.arange(100,520,20)])
ls_models = [fit_model_n_feat(X_train, y_train, top_features_gain, n_feat, n_feat_arr, model="mlp") for n_feat in n_feat_arr]
