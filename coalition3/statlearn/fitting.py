""" [COALITION3] Functions used for the selection of features at different
    leadtimes."""

## Import packages and define functions:
from __future__ import division
from __future__ import print_function

import os
import sys

import pickle
import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import datetime as dt
import matplotlib.pylab as plt

from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from pysteps.postprocessing.probmatching import nonparam_match_empirical_cdf

import coalition3.statlearn.feature as feat
import coalition3.statlearn.inputprep as ipt

## =============================================================================
## FUNCTIONS:

## Get cv MLP model:
def cv_mlp_model(X_train, y_train, n_feat, verbose_bool=False):
    alpha_vals = [1e+0, 1e-3, 1e-6]
    hidden_layer_setup = [(int(np.ceil(n_feat*1.5)),int(np.ceil(n_feat*1.5))), # (n_feat,int(np.ceil(n_feat/2.))),
                          (int(np.ceil(n_feat*2/3.)),int(np.ceil(n_feat*1/3.)))]
    param_grid = {'alpha': alpha_vals, 'hidden_layer_sizes': hidden_layer_setup}
    gs_object = GridSearchCV(MLPRegressor(verbose=verbose_bool), param_grid=param_grid, cv=3)
    gs_object.fit(X_train.values, y_train.values)
    return gs_object

## Get Mean Square Error (MSE) for different number of n features:
def fit_model_n_feat(X_train, y_train, top_features, n_feat, n_feat_arr,
                     model="xgb", cv=True, verbose_bool=False, set_log_weight=False):
    perc_finished = np.sum(n_feat_arr[:np.where(n_feat_arr==n_feat)[0][0]],
                           dtype=np.float32) / np.sum(n_feat_arr,dtype=np.float32)
    ending = "\n" if verbose_bool else "\r"
    print("    Working on %3i features (~%4.1f%%)" % (n_feat,perc_finished*100),end=ending)
    sys.stdout.flush()
    if model=="xgb":
        if set_log_weight:
            if np.where(n_feat_arr==n_feat)[0][0] == 0:
                print("\n      Using log weights for XGB model fitting")
            s_weights = X_train["s_weight"].values
            X = X_train.drop(labels="s_weight", axis=1)
        else:
            s_weights = None
        model = xgb.XGBRegressor(max_depth=6,silent=True,n_jobs=6,nthreads=6)
        model.fit(X_train[top_features.index[:n_feat]], y_train,
                  sample_weight=s_weights)
    elif model=="mlp":
        if cv:
            model = cv_mlp_model(X_train[top_features.index[:n_feat]], y_train,
                                 n_feat, verbose_bool)
        else:
            model = MLPRegressor(hidden_layer_sizes=(n_feat,np.ceil(n_feat/2.)))
            model.fit(X_train[top_features.index[:n_feat]], y_train)
    return(model)
    #return(mean_squared_error(y_test, model.predict(X_test[top_features.index[:n_feat]])))

## Get Mean Square Error (MSE) and R^2 for different number of n features:
def mse_r2_n_feat(X_test, y_test, top_features, n_feat, model):
    if isinstance(model, sklearn.model_selection._search.GridSearchCV):
        model_pred = model.best_estimator_
    else:
        model_pred = model
    if isinstance(top_features, list):
        prediction = model_pred.predict(X_test[top_features[:n_feat]])
    else:
        prediction = model_pred.predict(X_test[top_features.index[:n_feat]])
    mse_val    = sklearn.metrics.mean_squared_error(y_test,prediction)
    r2_val     = sklearn.metrics.r2_score(y_test,prediction)
    return(mse_val, r2_val)


def get_n_feat_arr(model):
    n_feat_arr = np.concatenate([np.arange(10)+1,
                                 np.arange(12,40,2),
                                 np.arange(40,100,20),
                                 np.arange(100,200,25),
                                 np.arange(200,300,50),
                                 np.arange(300,500,100),
                                 np.arange(500,1250,250)])
    if model == "xgb":
        return n_feat_arr
    elif model in ["ann","mlp"]:
        return n_feat_arr[9:]

## Get Mean Square Error depending on number of features:
def get_mse_from_n_feat(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path,
                        mod_bound=None,mod_name="",delete_RADAR_t0=False,
                        set_log_weight=False):
    print("Get dependence of MSE on n features for lead time t0 + %imin" % pred_dt, end="")
    if mod_bound is not None:
        print(" (for %s)" % mod_name)
        mod_name = "_%s" % mod_name
    else:
        print(" (for all samples)")
    sys.stdout.flush()

    ## Check whether data on MSE already exists:
    calc_new_model = "y"
    if os.path.exists(os.path.join(model_path,"MSE_feature_count_gain_%i%s.pkl" % (pred_dt,mod_name))):
        calc_new_model = ""
        while (calc_new_model!="y" and calc_new_model!="n"):
            calc_new_model = raw_input("  MSE data exists alreay, get new one? [y/n] ")
        #if calc_new_model=="n":
        #    print("  Use existing one, return from this function")
        #    return

    ## Calculate sample weights for XGB fitting:
    if set_log_weight:
        df_nonnan_nonzerot0["s_weight"] = feat.calc_sample_weight(df_nonnan_nonzerot0["TRT_Rank|0"],
                                                                  df_nonnan_nonzerot0["TRT_Rank_diff|%i" % pred_dt])

    ## Delete rows with TRT Rank close to zero at lead time:
    print("  Delete rows with TRT Rank close to zero at lead time")
    if delete_RADAR_t0:
        print("  Get predictor matrix X without RADAR variables at t0")
        X_feature_sel = "no_radar_t0"
    else:
        print("  Get predictor matrix X with RADAR variables at t0")
        X_feature_sel = "all"
    X_train, X_test, y_train, y_test = ipt.get_model_input(df_nonnan_nonzerot0,
            del_TRTeqZero_tpred=True, split_Xy_traintest=True,
            pred_dt=pred_dt, TRTRankt0_bound=mod_bound,check_for_nans=False,
            X_feature_sel=X_feature_sel)

    ## Load XGBmodel:
    print("  Load XGBmodel")
    with open(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6.pkl" % (pred_dt,mod_name)),"rb") as file:
        xgb_model = pickle.load(file)

    ## Order features by importance (gain):
    top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                               orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                               ascending=False)

    ## Create list of number of features to select for the fitting:
    n_feat_arr = get_n_feat_arr(model="xgb")

    ## Get models fitted with n top features:
    if calc_new_model=="y":
        print("  Get models fitted with n top features")
        ls_models = [fit_model_n_feat(X_train, y_train, top_features_gain, n_feat, n_feat_arr, set_log_weight=set_log_weight) for n_feat in n_feat_arr]
        print("    Save list of models as pickle to disk")
        with open(os.path.join(model_path,"models_%i%s_t0diff_maxdepth6_nfeat.pkl" % (pred_dt,mod_name)),"wb") as file:
            pickle.dump(ls_models, file, protocol=2)
    else:
        print("  Load existing models fitted with n top features")
        with open(os.path.join(model_path,"models_%i%s_t0diff_maxdepth6_nfeat.pkl" % (pred_dt,mod_name)),"rb") as file:
            ls_models = pickle.load(file)

    ## Get mean square error of models with n features:
    print("  Get mean square error of models with n features")
    MSE_r2_ls = [mse_r2_n_feat(X_test, y_test, top_features_gain, n_feat, model) \
                   for n_feat, model in zip(n_feat_arr,ls_models)]
    df_mse_feat_count = pd.DataFrame.from_dict({"Feature Count": n_feat_arr,
        "MSE %imin%s" % (pred_dt,mod_name): [score[0] for score in MSE_r2_ls],
         "R2 %imin%s" % (pred_dt,mod_name): [score[1] for score in MSE_r2_ls]})
    df_mse_feat_count.set_index("Feature Count",inplace=True)
    print("    Save dataframe with MSE to disk")
    with open(os.path.join(model_path,"MSE_feature_count_gain_%i%s.pkl" % (pred_dt,mod_name)),"wb") as file:
        pickle.dump(df_mse_feat_count,file,protocol=2)

    ## Append MSE values to existing HDF5 file (if existing):
    print("  Append MSE values to HDF5 file")
    df_mse_feat_count.to_hdf(os.path.join(model_path,"MSE_feature_count_gain.h5"),
                             key="MSE_%imin%s" % (pred_dt,mod_name), mode="a", format="t", append=True)

## Plot dependence of MSE on feature number:
def plot_mse_from_n_feat(ls_pred_dt,cfg_tds,model_path,thresholds=None,ls_model_names=[""]):
    print("Plot MSE as function of number of features used")
    if ls_model_names != [""]:
        ls_model_names = ["_%s" % name for name in ls_model_names]

    ## Read in MSE values:
    print("  Reading MSE values from disk")
    ls_df_mse_feat_count = []
    for model_name in ls_model_names:
        ls_df_mse_feat_count += [pd.read_hdf(os.path.join(model_path,"MSE_feature_count_gain.h5"),
                                             key="MSE_%imin%s" % (pred_dt,model_name), mode="r") for pred_dt in ls_pred_dt]
    df_mse_feat_count = pd.concat(ls_df_mse_feat_count, axis=1)

    ## Plot the figure:
    print("  Plotting the figure")
    col10 = '#E69F00'
    col30 = '#D55E00'

    if (len(ls_pred_dt)>0 and ls_model_names == [""]):
        #df_mse_feat_count_norm = df_mse_feat_count/df_mse_feat_count.mean()
        df_mse = df_mse_feat_count[[MSE_col for MSE_col in df_mse_feat_count.columns if "MSE" in MSE_col]]
        df_r2  = df_mse_feat_count[[R2_col  for R2_col  in df_mse_feat_count.columns if "R2"  in R2_col]]
        df_mse.columns = [colname[3:] for colname in df_mse.columns]
        df_r2.columns  = [colname[3:] for colname in df_r2.columns]
        fig, axes = plt.subplots(1, 2, figsize = (8,6))
        df_mse.plot(ax = axes[0], linestyle="-", cmap="viridis_r",legend=False)
        axes[0].set_ylabel(r'Mean square error MSE')
        df_r2.plot(ax = axes[1], linestyle="-", cmap="viridis_r")
        axes[1].set_ylabel(r'Coeff of determination $\mathregular{R^2}$')
        plt.tight_layout()
        for ax in axes:
            ax.grid()
        plt.savefig(os.path.join(cfg_tds["fig_output_path"],
                                         "MSE_R2_feature_count.pdf"),
                    orientation="portrait")
    elif (len(ls_pred_dt) == 3 and ls_model_names == [""]):
        fig = plt.figure(figsize = [10,5])
        ax = fig.add_subplot(1,1,1)
        df_mse_feat_count_norm = df_mse_feat_count/df_mse_feat_count.mean()
        df_mse_feat_count_norm
        df_mse_feat_count_norm.plot(ax = ax, linestyle="-", cmap="Set1")
        ax.set_ylabel("Normalised MSE")
        ax.set_xlabel("Number of features")
        ax.set_title("Normalised mean square error (MSE) as function of feature count")
        ax.grid()
        plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count.pdf"), orientation="portrait")
    elif (len(ls_pred_dt) == 2 and ls_model_names == [""]):
        fig = plt.figure(figsize = [10,5])
        ax1 = fig.add_subplot(1,1,1)
        ax2 = ax1.twinx()
        ax1.plot(df_mse_feat_count.index,df_mse_feat_count[["MSE %imin" % ls_pred_dt[0]]], col10, label="%imin" % ls_pred_dt[0])
        ax2.plot(df_mse_feat_count.index,df_mse_feat_count[["MSE %imin" % ls_pred_dt[1]]], col30, label="%imin" % ls_pred_dt[1])
        #ax1.semilogx(n_feat_arr,df_mse_feat_count[["MSE 10min"]], '#E69F00', label='10min')
        #ax2.semilogx(n_feat_arr,df_mse_feat_count[["MSE 30min"]], '#56B4E9', label='30min')
        #df_mse_feat_count.plot(ax=ax1,x="Feature Count",y="MSE 10min", style='b-', secondary_y=False)
        #df_mse_feat_count.plot(ax=ax2,x="Feature Count",y="MSE 30min", style='g-', secondary_y=True)
        ax1.set_title("Mean square error (MSE) as function of feature count")
        ax1.set_xlabel(r"Number of features")
        ax1.set_ylabel("MSE - %imin prediction" % ls_pred_dt[0], color=col10)
        ax2.set_ylabel("MSE - %imin prediction" % ls_pred_dt[1], color=col30)
        ax1.grid()
        #ax2.legend([ax1.get_lines()[0], ax2.right_ax.get_lines()[0]], ['A','B'], bbox_to_anchor=(1.5, 0.5))
        #plt.show()
        if thresholds is not None:
            props = dict(boxstyle='round', facecolor='white')
            ax2.axvline(x=thresholds[0], color=col10, linestyle='--', linewidth=2)
            ax2.axvline(x=thresholds[1], color=col30, linestyle='--', linewidth=2)
            ax2.text(thresholds[0], df_mse_feat_count[["MSE %imin" % ls_pred_dt[1]]].values[3],
                     "Treshold %imin" % ls_pred_dt[0], verticalalignment='top', bbox=props)
            ax2.text(thresholds[1], df_mse_feat_count[["MSE %imin" % ls_pred_dt[1]]].values[3],
                     "Treshold %imin" % ls_pred_dt[1], verticalalignment='top', bbox=props)
            plt.pause(8)
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count_thresh.pdf"), orientation="portrait")
        else:
            plt.pause(8)
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count_%i.pdf"), orientation="portrait")
            plt.close()
    elif (len(ls_pred_dt) == 2 and ls_model_names != [""]):
        df_mse_feat_count_norm = df_mse_feat_count/df_mse_feat_count.mean()
        df_mse_feat_count_norm.columns = [colname.replace("_", " (")+")" for colname in df_mse_feat_count_norm.columns]
        fig = plt.figure(figsize = [10,5])
        ax = fig.add_subplot(1,1,1)
        if len(ls_model_names)==2:
            df_mse_feat_count_norm.iloc[:,:2].plot(ax = ax, color=[col10,col30])
            df_mse_feat_count_norm.iloc[:,2:].plot(ax = ax, linestyle="--", color=[col10,col30])
        else:
            ax = df_mse_feat_count_norm.iloc[:,::2].plot(cmap="Paired")
            df_mse_feat_count_norm.iloc[:,1::2].plot(ax = ax, linestyle="--", cmap="Paired")
        ax.set_ylabel("Normalised MSE")
        ax.set_xlabel("Number of features")
        ax.set_title("Normalised mean square error (MSE) as function of feature count")
        ax.grid()
        if thresholds is not None and len(ls_model_names)==2:
            ps = dict(boxstyle='round', facecolor='white')
            ax.axvline(x=thresholds[0], color=col10, linestyle='solid', linewidth=2)
            ax.axvline(x=thresholds[1], color=col30, linestyle='solid', linewidth=2)
            ax.axvline(x=thresholds[2], color=col10, linestyle='--', linewidth=2)
            ax.axvline(x=thresholds[3], color=col30, linestyle='--', linewidth=2)
            plt.pause(8)
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count_thresh.pdf"), orientation="portrait")
            plt.close()
        else:
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count.pdf"), orientation="portrait")
    elif (len(ls_pred_dt) == 1 and ls_model_names != [""]):
        df_mse_feat_count_norm = df_mse_feat_count/df_mse_feat_count.mean()
        df_mse_feat_count_norm.columns = [colname.replace("_", " (")+")" for colname in df_mse_feat_count_norm.columns]
        fig = plt.figure(figsize = [10,5])
        ax = fig.add_subplot(1,1,1)
        if len(ls_model_names)==1:
            df_mse_feat_count_norm.iloc[:,:2].plot(ax = ax, color=[col10,col30])
            #df_mse_feat_count_norm.iloc[:,2:].plot(ax = ax, linestyle="--", color=[col10,col30])
        else:
            ax = df_mse_feat_count_norm.iloc[:,::2].plot(cmap="Paired")
            df_mse_feat_count_norm.iloc[:,1::2].plot(ax = ax, linestyle="--", cmap="Paired")
        ax.set_ylabel("Normalised MSE")
        ax.set_xlabel("Number of features")
        ax.set_title("Normalised mean square error (MSE) as function of feature count")
        ax.grid()
        if thresholds is not None and len(ls_model_names)==2:
            ps = dict(boxstyle='round', facecolor='white')
            ax.axvline(x=thresholds[0], color=col10, linestyle='solid', linewidth=2)
            #ax.axvline(x=thresholds[1], color=col30, linestyle='solid', linewidth=2)
            #ax.axvline(x=thresholds[2], color=col10, linestyle='--', linewidth=2)
            #ax.axvline(x=thresholds[3], color=col30, linestyle='--', linewidth=2)
            plt.pause(8)
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count_thresh.pdf"), orientation="portrait")
            plt.close()
        else:
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count.pdf"), orientation="portrait")
    else:
        df_mse_feat_count.plot.line()
        plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count.pdf"), orientation="portrait")

## Get observed and forecasted TRT Ranks in different files:
def get_obs_fcst_TRT_Rank(TRT_t0, TRT_diff_pred, TRT_diff_obs, TRT_tneg5):
    obs           = TRT_t0 + TRT_diff_obs
    pred_mod      = TRT_t0 + TRT_diff_pred
    pred_mod.name = "Rank model prediction"
    
    pred_mod_PM = pd.Series(nonparam_match_empirical_cdf(pred_mod.values,obs.values),
                            index=pred_mod.index, name="Rank model prediction (PM)")

    pred_pers      = TRT_t0.copy()
    pred_pers.name = "Rank persistency prediction"
    pred_pers_PM   = pd.Series(nonparam_match_empirical_cdf(pred_pers.values,obs.values),
                               index=pred_pers.index, name="Rank persistency prediction (PM)")
    
    pred_diff      = TRT_t0 + (TRT_t0-TRT_tneg5)
    pred_diff.name = "Rank constant gradient prediction"
    
    diff_pred      = pd.Series(TRT_diff_pred, index=TRT_t0.index, name="TRT rank difference model prediction")
    return(obs, pred_mod, pred_mod_PM, pred_pers, pred_pers_PM, pred_diff, diff_pred)
    
        
## Fit model with threshold number of features (selected by the user), 
## save to disk and make a plot:
def selected_model_fit(df_nonnan_nonzerot0,pred_dt,n_feat_ls,cfg_tds,
                       model_path,ls_mod_bound=[None],ls_model_names=[""]):
    if len(ls_mod_bound)>1:
        y_test_ls = []
        TRT_diff_pred_ls = []

    for mod_bound, n_feat, mod_name in zip(ls_mod_bound,n_feat_ls,ls_model_names):
        print("\nGet selected XGB model for prediction of lead time %imin" % (pred_dt),end="")
        if mod_bound is not None:
            print(" (%i features for model '%s')" % (n_feat,mod_name))
            mod_name = "_%s" % mod_name
        else:
            print(" (%i features for all samples)" % n_feat)
        sys.stdout.flush()

        ## Delete rows with TRT Rank close to zero at lead time:
        print("  Delete rows with TRT Rank close to zero at lead time")
        X_train, X_test, y_train, y_test = ipt.get_model_input(df_nonnan_nonzerot0,
                del_TRTeqZero_tpred=True, split_Xy_traintest=True,
                pred_dt=pred_dt, TRTRankt0_bound=mod_bound,check_for_nans=False,
                )

        precalc_n_feat = get_n_feat_arr("xgb")

        if n_feat not in precalc_n_feat:
            ## Load XGBmodel:
            print("  Load XGBmodel")
            with open(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6.pkl" % (pred_dt,mod_name)),"rb") as file:
                xgb_model = pickle.load(file)

            ## Order features by importance (gain):
            top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                                       orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                                       ascending=False)

            ## Fit model:
            model    = fit_model_n_feat(X_train, y_train, top_features_gain, n_feat, np.array(n_feat))
        else:
            with open(os.path.join(model_path,"models_%i%s_t0diff_maxdepth6_nfeat.pkl" % (pred_dt,mod_name)),"rb") as file:
                model = pickle.load(file)[np.where(precalc_n_feat==n_feat)[0][0]]

        ## Save the model to disk:
        model_saving_name = "model_%i%s_t0diff_maxdepth6_%ifeat_gain.pkl" % (pred_dt,mod_name,n_feat)
        with open(os.path.join(model_path,model_saving_name),"wb") as file:
            pickle.dump(model,file,protocol=2)
            
        ## Get features:
        features  = model.get_booster().feature_names

        ## Make prediction and get skill scores:
        TRT_diff_pred = model.predict(X_test[features])
        
        ## Append to list of results for combined plot:
        if len(ls_mod_bound)>1:
            y_test_ls.append(y_test)
            TRT_diff_pred_ls.append(TRT_diff_pred)

    ## Make combined plot:
    if len(ls_mod_bound)>1:
        y_test_combi = pd.concat(y_test_ls,axis=0)
        pred_gain_combi = np.concatenate(TRT_diff_pred_ls)
        mse_gain  = sklearn.metrics.mean_squared_error(y_test_combi,pred_gain_combi)
        r2_gain   = sklearn.metrics.r2_score(y_test_combi,pred_gain_combi)
        plot_pred_vs_obs_core(y_test_combi, pred_gain_combi,
                              pred_dt,"_%s" % "|".join(ls_model_names),cfg_tds)
    
    ## Return model for and put into dictionary:
    if len(ls_mod_bound)>1:
        raise ImplementationError("Not yet implemented to used models fitted with" + \
                                  "TRT Rank subset for prediction, not returned")
    else:
        return model

## Save dictionary with selected models to disk:
def save_selected_models(dict_sel_model, model_path, cfg_op):
    print("\nSave dictionary with selected models to the disk.")
    train_path_name = os.path.join(model_path,"model_dict_t0diff_maxdepth6_selfeat_gain.pkl")
    with open(train_path_name,"wb") as file:
        pickle.dump(dict_sel_model,file,protocol=2)
    print("  saved dict to 'model_path' location:\n    %s" % train_path_name)
    op_path_name = os.path.join(cfg_op["XGB_model_path"],
                                       "model_dict_t0diff_maxdepth6_selfeat_gain.pkl")
    with open(op_path_name,"wb") as file:
        pickle.dump(dict_sel_model,file,protocol=2)
    print("  saved dict to 'XGB_model_path' location:\n    %s" % op_path_name)
    prt_txt = """
    ---------------------------------------------------------------------------------
        Now it is YOUR responsibility to set the soft link 'pred_model_ls' in the
        directory '%s'
        to the new collection of models 'model_dict_t0diff_maxdepth6_selfeat_gain.pkl'
        if these should be used for prediction!
    ---------------------------------------------------------------------------------\n""" % (cfg_op["XGB_model_path"])
    print(prt_txt)
    


















