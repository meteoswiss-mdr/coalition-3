# coding: utf-8
""" [COALITION3] Script fitting ANN models using a grid-search over several
    hyper-parameters (ANN architecture and learning rate)."""
    
## Import packages and define functions:
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pylab as plt

import coalition3.inout.paths as pth
import coalition3.inout.readconfig as cfg
import coalition3.statlearn.feature as feat
import coalition3.statlearn.inputprep as ipt
    
import sklearn.metrics as met
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

## ============================================================================
## Get config info:
cfg_tds = cfg.get_config_info_tds()
cfg_op, __, __ = cfg.get_config_info_op()
mod_name = ""

## Open pandas training dataframe:
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_df = pth.file_path_reader("pandas training dataframe (nonnan)",user_argv_path)
model_path = pth.file_path_reader("model saving location")
print("\nLoading nonnan dataframe into RAM")
df_nonnan  = pd.read_hdf(path_to_df,key="df_nonnan")

## Get lead-time from user:
ls_pred_dt = feat.get_pred_dt_ls("the ANN fit", cfg_op["timestep"],cfg_op["n_integ"])

## Loop over time-deltas:
for pred_dt in ls_pred_dt:
    ## Get normalised training and testing data:
    X_train, X_test, y_train, y_test, scaler = ipt.get_model_input(df_nonnan,
        del_TRTeqZero_tpred=True, split_Xy_traintest=True, X_normalise=True,
        pred_dt=pred_dt)
    
    ## Fit ANN model with all features but only two hidden layers (100, 50):
    print("Fit ANN model with all features but only two hidden layers (100, 50)")
    mlp_allfeat = MLPRegressor(hidden_layer_sizes=(100,50), verbose=True)
    mlp_allfeat.fit(X_train,y_train)
    with open(os.path.join(model_path,"model_%i%s_t0diff_mlp_allfeat.pkl" % (pred_dt,mod_name)),"wb") as file:
        pickle.dump(mlp_allfeat,file,protocol=-1)
        
    pred_mlp_allfeat = mlp_allfeat.predict(X_test)
    mse_mlp_allfeat  = met.mean_squared_error(y_test,pred_mlp_allfeat)
    r2_mlp_allfeat   = met.r2_score(y_test,pred_mlp_allfeat)
    feat.plot_pred_vs_obs_core(y_test,pred_mlp_allfeat,pred_dt,mse_mlp_allfeat,
                               r2_mlp_allfeat,"_mlp-allfeat",cfg_tds)
    
    ## Fit ANN model with 300 selected features but only two hidden layers (100, 50):
    print("Fit ANN model with 300 features but only two hidden layers (100, 50)")
    xgb_model_path_300 = pth.file_path_reader("300 feature XGB model location (for feature selection)")
    with open(xgb_model_path_300,"rb") as file:
        xgb_model = pickle.load(file)
    features        = xgb_model.get_booster().feature_names
    X_train_selfeat = X_train[features]
    X_test_selfeat  = X_test[features]

    mlp_selfeat = MLPRegressor(hidden_layer_sizes=(100,50), verbose=True)
    with open(os.path.join(model_path,"model_%i%s_t0diff_mlp_300feat.pkl" % (pred_dt,mod_name)),"wb") as file:
        pickle.dump(mlp_selfeat,file,protocol=-1)
    
    mlp_selfeat.fit(X_train_selfeat,y_train)
    pred_mlp_selfeat = mlp_selfeat.predict(X_test_selfeat)
    mse_gain_selfeat = met.mean_squared_error(y_test,pred_mlp_selfeat)
    r2_gain_selfeat  = met.r2_score(y_test,pred_mlp_selfeat)
    feat.plot_pred_vs_obs_core(y_test,pred_mlp_selfeat,pred_dt,mse_gain_selfeat,r2_gain_selfeat,"_mlp-300feat",cfg_tds)
    
    ## Fit ANN models with 10 - 500 selected features with grid-search over hyperparameters:
    print("Fit ANN models to 10 - 500 features with grid-search over hyper-parameters")
    xgb_model_path_all = pth.file_path_reader("all-feature XGB model location (for feature selection)")
    with open(xgb_model_path_all,"rb") as file:
        xgb_model = pickle.load(file)
    top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                               orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                               ascending=False)
    n_feat_arr = np.concatenate([np.arange(10,50,2),
                                 np.arange(50,100,10),
                                 np.arange(100,520,20)])
    print("  *** Watch out, this takes veeeeeeeeeeeeery long!! ***")
    ls_models = [feat.fit_model_n_feat(X_train, y_train, top_features_gain, n_feat, n_feat_arr, model="mlp", verbose_bool=True) for n_feat in n_feat_arr]
    
    with open(os.path.join(model_path,"model_%i%s_t0diff_mlp_nfeat.pkl" % (pred_dt,mod_name)),"wb") as file:
        pickle.dump(ls_models,file,protocol=-1)
        
    ## Make plot showing model architecture:
    best_param_ls = [ele.best_params_ for ele in ls_models]
    model_arch_ls = np.array([ele['hidden_layer_sizes'][0]/float(nfeat) for ele, nfeat in zip(best_param_ls, n_feat_arr)])
    model_arch_ls[np.where(model_arch_ls>1.1)[0]] = 3
    model_arch_ls[np.where(np.isclose(model_arch_ls,1))[0]] = 2
    model_arch_ls[np.where(model_arch_ls<0.9)[0]] = 1
    
    learn_rate_ls = np.log10([ele['alpha'] for ele in best_param_ls])
    df_best_model = pd.DataFrame.from_dict({"Model Architecture": model_arch_ls, "Regularisation Exponent": learn_rate_ls})
    df_best_model = df_best_model.set_index(n_feat_arr)
        
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10,8])
    for ax in axes:
        ax.grid()
        ax.set_xlabel("Feature Count")
    df_best_model.iloc[:,0].plot.scatter(ax=ax[0])
    df_best_model.iloc[:,1].plot.scatter(ax=ax[1])
    """
    axes = df_best_model.plot.line(ylim=(-5,4),sharex=True,grid=True) # subplots=True
    axes.set_xlabel("Feature Count")       
    plt.show()
        
        
        
        
        
        
        
        
        