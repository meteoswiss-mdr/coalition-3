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
import matplotlib.colors as mcolors
from pandas.api.types import CategoricalDtype

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

import coalition3.statlearn.inputprep as ipt
from coalition3.visualisation.TRTcells import contour_of_2dHist

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
                     model="xgb", cv=True, verbose_bool=False):
    perc_finished = np.sum(n_feat_arr[:np.where(n_feat_arr==n_feat)[0][0]],
                           dtype=np.float32) / np.sum(n_feat_arr,dtype=np.float32)
    ending = "\n" if verbose_bool else "\r"
    print("    Working on %3i features (~%4.1f%%)" % (n_feat,perc_finished*100),end=ending)
    sys.stdout.flush()
    if model=="xgb":
        model = xgb.XGBRegressor(max_depth=6,silent=True,n_jobs=6,nthreads=6)
        model.fit(X_train[top_features.index[:n_feat]], y_train)
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

## Get prediction time from user:
def get_pred_dt_ls(input_str, timestep=None,n_integ=None):
    ls_pred_dt = [-5]
    if (timestep is not None and n_integ is not None):
        poss_fcst_steps = np.arange(timestep,
                                    timestep*n_integ,
                                    timestep)
        print("\nFor which time delta (%i, %i, .., %i) should %s be made?" % \
              (poss_fcst_steps[0], poss_fcst_steps[1], poss_fcst_steps[-1], input_str))
    else:
        poss_fcst_steps = np.arange(-100,100)
        print("\nFor which time delta should %s be made?" % input_str)
    
    while not np.all([pred_dt_i in poss_fcst_steps for pred_dt_i in ls_pred_dt]):
        ls_pred_dt = raw_input("  Select forecast step (if several, split with comma): ").split(",")
        ls_pred_dt = [int(pred_dt_i.strip()) for pred_dt_i in ls_pred_dt]
    print("  Perform %s at lead times %s min" % (input_str, ', '.join([str(pred_dt_i) for pred_dt_i in ls_pred_dt])))
    return ls_pred_dt
    
## Plotting procedure for feature importance:
def plot_feature_importance(model,X,delta_t,cfg_tds,mod_name):
    sort_ind = np.argsort(model.feature_importances_)[::-1]
    df_featimp = pd.DataFrame(np.array([X.columns[sort_ind[:500]],
                                        model.feature_importances_[sort_ind[:500]]]).T,
                              columns=["Feature","Importance"])
    df_featimp.plot(drawstyle="steps", linewidth=2)
    plt.grid(); plt.ylabel("Feature importance"); plt.xlabel("Features (sorted)")
    plt.title("Feature importance - TRT t+%imin" % delta_t)
    #plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Feature_importance_%imin%s_long.pdf" % (delta_t,mod_name)),orientation="portrait")

    fig = plt.figure(figsize = [10,10])
    ax1 = fig.add_subplot(2,1,1)
    xgb.plot_importance(model,ax1,max_num_features=20,importance_type="weight",
                        title="Feature importance (Weight) - TRT t+%imin" % delta_t)
    ax2 = fig.add_subplot(2,1,2)
    xgb.plot_importance(model,ax2,max_num_features=20,importance_type="gain",
                        title="Feature importance (Gain) - TRT t+%imin" % delta_t,show_values=False)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Feature_importance_%imin%s.pdf" % (delta_t,mod_name)),orientation="portrait")

## Get feature ranking for the complete dataset:
def get_feature_importance(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path,mod_bound=None,mod_name="",delete_RADAR_t0=False):
    print("Get features for lead time t0 + %imin" % pred_dt, end="")
    if mod_bound is not None:
        if mod_name=="":
            raise ValueError("Model name required")
        else:
            print(" (for %s)" % mod_name)
            mod_name = "_%s" % mod_name
        if len(mod_bound)!=2:
            raise ValueError("Model boundary list must have length 2")
    else:
        print(" (for all samples)")
    sys.stdout.flush()

    ## Check whether model already exists:
    if os.path.exists(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6.pkl" % (pred_dt,mod_name))):
        use_existing = ""
        while (use_existing!="y" and use_existing!="n"):
            use_existing = raw_input("  Model exists alreay, fit a new one? [y/n] ")
        if use_existing=="n":
            print("  Use existing one, return from this function")
            return

    ## Delete rows with TRT Rank close to zero at lead time:
    print("  Delete rows with TRT Rank close to zero at lead time")
    if delete_RADAR_t0:
        print("  Get predictor matrix X without RADAR variables at t0")
        X_feature_sel = "no_radar_t0" 
    else:
        print("  Get predictor matrix X with RADAR variables at t0")
        X_feature_sel = "all" 
    X, y = ipt.get_model_input(df_nonnan_nonzerot0, del_TRTeqZero_tpred=True,
            split_Xy=True, pred_dt=pred_dt, TRTRankt0_bound=mod_bound, X_feature_sel=X_feature_sel)
    del(df_nonnan_nonzerot0)
    if len(X)>80000:
        print("   *** Warning: Dataframe X probably to big to be converted, reduced to 80'000 rows! ***")
        X = X.sample(n=80000,random_state=42)
        y = y.sample(n=80000,random_state=42)
    #X = X.values
    #X = X.astype(np.float16, order='C', copy=False)

    ## Setup model:
    print("  Setup XGBmodel with max_depth = 6")
    xgb_model = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)

    ## Train model:
    print("  Train XGBmodel")
    d_start = dt.datetime.now()
    xgb_model.fit(X, y, verbose=True)
    print("    Elapsed time for XGBoost model fitting: %s" % (dt.datetime.now()-d_start))

    ## Save model to disk:
    print("  Save XGBmodel to disk")
    with open(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6.pkl" % (pred_dt,mod_name)),"wb") as file:
        pickle.dump(xgb_model,file,protocol=-1)

    ## Plot feature importance:
    print("  Plot feature importance")
    plot_feature_importance(xgb_model,X,pred_dt,cfg_tds,mod_name)

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
def get_mse_from_n_feat(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path,mod_bound=None,mod_name=""):
    print("Get dependence of MSE on n features for lead time t0 + %imin" % pred_dt, end="")
    if mod_bound is not None:
        print(" (for %s)" % mod_name)
        mod_name = "_%s" % mod_name
    else:
        print(" (for all samples)")
    sys.stdout.flush()

    ## Check whether data on MSE already exists:
    if os.path.exists(os.path.join(model_path,"MSE_feature_count_gain_%i%s.pkl" % (pred_dt,mod_name))):
        use_existing = ""
        while (use_existing!="y" and use_existing!="n"):
            use_existing = raw_input("  MSE data exists alreay, get new one? [y/n] ")
        if use_existing=="n":
            print("  Use existing one, return from this function")
            return

    ## Delete rows with TRT Rank close to zero at lead time:
    print("  Delete rows with TRT Rank close to zero at lead time")
    X_train, X_test, y_train, y_test = ipt.get_model_input(df_nonnan_nonzerot0,
            del_TRTeqZero_tpred=True, split_Xy_traintest=True,
            pred_dt=pred_dt, TRTRankt0_bound=mod_bound)

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
    print("  Get models fitted with n top features")
    ls_models = [fit_model_n_feat(X_train, y_train, top_features_gain, n_feat, n_feat_arr) for n_feat in n_feat_arr]
    print("    Save list of models as pickle to disk")
    with open(os.path.join(model_path,"models_%i%s_t0diff_maxdepth6_nfeat.pkl" % (pred_dt,mod_name)),"wb") as file:
        pickle.dump(ls_models, file, protocol=-1)

    ## Get mean square error of models with n features:
    print("  Get mean square error of models with n features")
    MSE_r2_ls = [mse_r2_n_feat(X_test, y_test, top_features_gain, n_feat, model) \
                   for n_feat, model in zip(n_feat_arr,ls_models)]
    df_mse_feat_count = pd.DataFrame.from_dict({"Feature Count": n_feat_arr,
        "MSE %imin%s" % (pred_dt,mod_name): [score[0] for score in MSE_r2_ls],
         "R2 %imin%s" % (pred_dt,mod_name): [score[1] for score in MSE_r2_ls]})
    df_mse_feat_count.set_index("Feature Count",inplace=True)
    print("    Save dataframe with MSE to disk")
    with open(os.path.join(model_path,"MSE_feature_count_gain_%i%s.pkl" % (pred_dt,mod_name)),"wb") as file: pickle.dump(df_mse_feat_count,file,protocol=-1)

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
    if (len(ls_pred_dt) == 3 and ls_model_names == [""]):
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
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count.pdf"), orientation="portrait")
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

## Fit model with threshold number of features (selected by the user):
def plot_pred_vs_obs(df_nonnan_nonzerot0,pred_dt,n_feat_ls,cfg_tds,model_path,
                     ls_mod_bound=[None],ls_model_names=[""]):
    if len(ls_mod_bound)>1:
        y_test_ls = []
        pred_gain_ls = []

    for mod_bound, n_feat, mod_name in zip(ls_mod_bound,n_feat_ls,ls_model_names):
        print("\nPlot predicted vs. observed testing samples for time delta %imin" % (pred_dt),end="")
        if mod_bound is not None:
            print(" (%i features for model '%s')" % (n_feat,mod_name))
            mod_name = "_%s" % mod_name
        else:
            print(" (%i features for all samples)" & n_feat)
        sys.stdout.flush()

        ## Delete rows with TRT Rank close to zero at lead time:
        print("  Delete rows with TRT Rank close to zero at lead time")
        X_train, X_test, y_train, y_test = ipt.get_model_input(df_nonnan_nonzerot0,
                del_TRTeqZero_tpred=True, split_Xy_traintest=True,
                pred_dt=pred_dt, TRTRankt0_bound=mod_bound)
        """
        if mod_bound is not None:
            df_nonnan_nonzerot0_mod = df_nonnan_nonzerot0.loc[(df_nonnan_nonzerot0["TRT_Rank|0"]>=mod_bound[0]) & \
                                                              (df_nonnan_nonzerot0["TRT_Rank|0"]<mod_bound[1])]
        X = df_nonnan_nonzerot0_mod.loc[df_nonnan_nonzerot0_mod["TRT_Rank|%i" % pred_dt]>=0.15,
                                    [Xcol for Xcol in df_nonnan_nonzerot0_mod.columns if ipt.get_X_col(Xcol)]]
        y = df_nonnan_nonzerot0_mod.loc[df_nonnan_nonzerot0_mod["TRT_Rank|%i" % pred_dt]>=0.15,
                                    ["TRT_Rank_diff|%i" % pred_dt]]
        del(df_nonnan_nonzerot0_mod)

        ## Split training/testing data:
        print("  Split into training/testing data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        """
        del(X,y)

        precalc_n_feat = np.concatenate([np.arange(10)+1, np.arange(12,50,2),
                                         np.arange(50,100,10), np.arange(100,520,20)])
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

        ## Get features:
        features  = model.get_booster().feature_names

        ## Make prediction and get skill scores:
        pred_gain = model.predict(X_test[features])
        mse_gain  = sklearn.metrics.mean_squared_error(y_test[["TRT_Rank_diff|%i" % pred_dt]], pred_gain)
        r2_gain   = sklearn.metrics.r2_score(y_test[["TRT_Rank_diff|%i" % pred_dt]], pred_gain)

        ## Save the model to disk:
        with open(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6_%ifeat_gain.pkl" % (pred_dt,mod_name,n_feat)),"wb") as file:
            pickle.dump(model,file,protocol=-1)

        ## Make the plot:
        plot_pred_vs_obs_core(y_test,pred_gain,pred_dt,mse_gain,r2_gain,mod_name,cfg_tds)

        ## Append to list of results for combined plot:
        if len(ls_mod_bound)>1:
            y_test_ls.append(y_test)
            pred_gain_ls.append(pred_gain)

    ## Make combined plot:
    if len(ls_mod_bound)>1:
        y_test_combi = pd.concat(y_test_ls,axis=0)
        pred_gain_combi = np.concatenate(pred_gain_ls)
        mse_gain  = sklearn.metrics.mean_squared_error(y_test_combi[["TRT_Rank_diff|%i" % pred_dt]],pred_gain_combi)
        r2_gain   = sklearn.metrics.r2_score(y_test_combi[["TRT_Rank_diff|%i" % pred_dt]],pred_gain_combi)
        plot_pred_vs_obs_core(y_test_combi, pred_gain_combi,
                              pred_dt,mse_gain,r2_gain,"_%s" % "|".join(ls_model_names),cfg_tds)


def plot_pred_vs_obs_core(y_test,pred_gain,pred_dt,mse_gain,r2_gain,mod_name,cfg_tds):
    print("  Making the plot")
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[10,8])
    if len(y_test)>1000:
        counts,ybins,xbins,image = axes.hist2d(y_test.values,pred_gain,
                                               bins=200,range=[[-2.5,2.5],[-2.5,2.5]],cmap="magma",norm=mcolors.LogNorm())
        #counts,ybins,xbins,image = axes.hist2d(y_test[["TRT_Rank_diff|%i" % pred_dt]].values[:,0],pred_gain,
        #                                       bins=200,range=[[-2.5,2.5],[-2.5,2.5]],cmap="magma",norm=mcolors.LogNorm())
        cbar = fig.colorbar(image, ax=axes, extend='max')
    else:
        axes.scatter(y_test[["TRT_Rank_diff|%i" % pred_dt]].values[:,0],pred_gain,
                     marker="+", color="black", s=8)
    axes.set_xlim([-2.5,2.5]); axes.set_ylim([-2.5,2.5])
    #cbar.set_label('Number of cells per bin of size [0.02, 0.02]', rotation=90)
    axes.grid()
    #axes.fill_between([-0.2,0.2],y1=[-1.5,-1.5], y2=[1.5,1.5], facecolor="none", hatch="X", edgecolor="darkred", linewidth=0.5)
    axes.plot([-4,4],[-4,4],'w--',linewidth=2) #,facecolor="w",linewidth=2,linestyle='--')
    if len(y_test)>1000:
        cont2d_1, lvl = contour_of_2dHist(counts,smooth=True)
        CS = axes.contour(cont2d_1,levels=lvl,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=2,cmap="YlGn_r")
        CS_lab = axes.clabel(CS, inline=1, fontsize=10, fmt='%i%%', colors="black")
        #[txt.set_backgroundcolor('white') for txt in CS_lab]
        [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.3, boxstyle='round', alpha=0.71)) for txt in CS_lab] #pad=0,
    axes.set_xlabel(r'Observed TRT Rank difference t$\mathregular{_{+%imin}}$' % pred_dt)
    axes.set_ylabel(r'Predicted TRT Rank difference t$\mathregular{_{+%imin}}$' % pred_dt)
    model_title = "" if mod_name == "" else r" | Mod$\mathregular{_{%s}}$" % mod_name[1:]
    title_str = 'TRT Ranks differences\nTime delta: %imin' % pred_dt
    title_str += model_title
    axes.set_title(title_str)
    axes.set_aspect('equal'); axes.patch.set_facecolor('0.71')
    str_n_cells  = "Mean Squared Error (MSE): %.2f\n" % (mse_gain)
    str_n_cells += r"Coeff of determination ($R^2$): %.2f" % (r2_gain)
    props = dict(boxstyle='round', facecolor='white')
    axes.text(-2,2, str_n_cells, bbox=props)
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Pred_scatter_%i%s.pdf" % (pred_dt,mod_name.replace("|","-"))), orientation="portrait")
    print("    Saved the plot")

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z
    
def plot_feat_source_dt_gainsum(path_xgb, cfg_op, cfg_tds, pred_dt_ls = None):
    print("Plot feature source and time delta importance")
    if pred_dt_ls is None:
        pred_dt_ls  = np.arange(cfg_op["timestep"],cfg_op["n_integ"]*cfg_op["timestep"],cfg_op["timestep"])
        print("  This script assumes that for all future time steps %imin - %imin, " % (pred_dt_ls[0],pred_dt_ls[-1]) +
              "XGB models with all features have been fitted")
    else:
        print("  This script only plot the feature source/time step importance for the time steps:\n    %s" % pred_dt_ls)
    source_dict = merge_two_dicts(cfg_op["source_dict"], cfg_op["source_dict_combi"])

    score_ls    = []
    model_ls    = []
    for i, pred_dt in enumerate(pred_dt_ls):
        model_path_xgb = os.path.join(path_xgb,"model_%i_t0diff_maxdepth6.pkl" % pred_dt)
        with open(model_path_xgb,"rb") as file:
            xgb_dt = pickle.load(file)
        score_ls.append(pd.DataFrame.from_dict(xgb_dt.get_booster().get_score(importance_type='gain'),
                                               orient="index",columns=["%imin" % pred_dt]))
        model_ls.append(xgb_dt)

    feature_names = model_ls[0].get_booster().feature_names
    df_gain = pd.concat([pd.DataFrame([],index=feature_names)]+score_ls,axis=1,sort=True).fillna(0, inplace=False).astype(np.float32)

    feat_ls = list(df_gain.index)
    feat_var_ls = [feat.split("|")[0] for feat in feat_ls]
    feat_dt_ls  = [int(feat.split("|")[1]) if "|" in feat else 0 for feat in feat_ls]

    Time_feat      = [u'SOLAR_TIME_COS', u'SOLAR_TIME_SIN']
    TRT_Rank_feat  = [u'RANK', u'RANKr', u'TRT_Rank']
    TRT_light_feat = [u'CG', u'CG_minus', u'CG_plus', u'perc_CG_plus']
    TRT_radar_feat = [u'CZC_lt57dBZ',  u'ET15', u'ET15m', u'ET45', u'ET45m', u'POH', u'VIL',  u'maxH', u'maxHm']
    TRT_loc_feat   = [u'Border_cell', u'Dvel_x', u'Dvel_y', u'angle', u'area', u'det', u'ell_L', u'ell_S', u'iCH', u'jCH', u'lat', u'lon', u'vel_x', u'vel_y']

    feat_src_ls = [source_dict[feat_var.split("_stat")[0]] if "_stat" in feat_var else feat_var for feat_var in feat_var_ls]
    feat_src_ls = [source_dict[feat_var.split("_pixc")[0]] if "_pixc" in feat_var else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["TRT_Rank" if feat_var in TRT_Rank_feat  else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["RADAR"    if feat_var in TRT_radar_feat else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["THX"      if feat_var in TRT_light_feat else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["LOC_AREA" if feat_var in TRT_loc_feat   else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["TIME"     if feat_var in Time_feat      else feat_var for feat_var in feat_src_ls]
        
    df_gain = pd.concat([df_gain, pd.DataFrame.from_dict({"TIME_DELTA": feat_dt_ls, "VARIABLE": feat_var_ls, "SOURCE": feat_src_ls}).set_index(df_gain.index)], axis=1)
    df_gain["SOURCE"]   = df_gain["SOURCE"].astype('category')
    df_gain["VARIABLE"] = df_gain["VARIABLE"].astype('category')

    df_sum_source      = df_gain.groupby("SOURCE").sum().transpose().drop("TIME_DELTA")
    df_sum_source_norm = df_sum_source.div(df_sum_source.sum(axis=1), axis=0)
    df_sum_source_norm = df_sum_source_norm.iloc[:,[3,4,0,5,2,7,1,6]]

    df_sum_dtime       = df_gain.groupby("TIME_DELTA").sum().transpose()
    df_sum_dtime_norm  = df_sum_dtime.div(df_sum_dtime.sum(axis=1), axis=0)
    df_sum_dtime_norm.columns = ["%imin" % colname for colname in df_sum_dtime_norm.columns]

    fig = plt.figure(figsize = [10,13])
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    df_sum_source_norm.plot.line(ax=ax1, cmap="Set1", linewidth=2)
    df_sum_dtime_norm.plot.line(ax=ax2, cmap="viridis_r", linewidth=2) 
    for title,ax in zip(["Feature source","Past time step"],[ax1,ax2]):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, # + box.height * 0.1,
                         box.width, box.height * 0.8])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                  fancybox=True, shadow=True, ncol=5,title=title,fontsize="medium")
        ax.set_xlabel("Lead time")
        ax.grid()
    ax1.set_ylabel("Relative feature source importance")
    ax2.set_ylabel("Relative time step importance")
    #plt.tight_layout()
    plt_saving_location = os.path.join(cfg_tds["fig_output_path"],"Feature_source_time_step_importance.pdf")
    plt.savefig(plt_saving_location,orientation="portrait")
    print("  Plot saved in:\n    %s" % plt_saving_location)
        
    
    
    
    