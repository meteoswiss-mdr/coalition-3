""" [COALITION3] Functions used for the selection of features at different
    leadtimes."""

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
import sklearn.metrics as met
import matplotlib.pylab as plt
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split

from coalition3.visualisation.TRTcells import contour_of_2dHist

## =============================================================================
## FUNCTIONS:
  
## Get Mean Square Error (MSE) for different number of n features:
def fit_model_n_feat(X_train, y_train, top_features, n_feat, n_feat_arr):
    perc_finished = np.sum(n_feat_arr[:np.where(n_feat_arr==n_feat)[0][0]],
                           dtype=np.float32) / np.sum(n_feat_arr,dtype=np.float32)
    print("    Working on %3i features (~%4.1f%%)" % (n_feat,perc_finished*100),end="\r")
    sys.stdout.flush()
    model = xgb.XGBRegressor(max_depth=6,silent=True,n_jobs=6,nthreads=6)
    model.fit(X_train[top_features.index[:n_feat]], y_train)
    return(model)
    #return(mean_squared_error(y_test, model.predict(X_test[top_features.index[:n_feat]])))
    
## Get Mean Square Error (MSE) for different number of n features:
def mse_n_feat(X_test, y_test, top_features, n_feat, model):
    return(met.mean_squared_error(y_test, model.predict(X_test[top_features.index[:n_feat]])))
    
## Plotting procedure for feature importance:
def plot_feature_importance(model,X,delta_t,cfg_tds):
    sort_ind = np.argsort(model.feature_importances_)[::-1]
    df_featimp = pd.DataFrame(np.array([X.columns[sort_ind[:500]],
                                        model.feature_importances_[sort_ind[:500]]]).T,
                              columns=["Feature","Importance"])
    df_featimp.plot(drawstyle="steps", linewidth=2)
    plt.grid(); plt.ylabel("Feature importance"); plt.xlabel("Features (sorted)")
    plt.title("Feature importance - TRT t+%imin" % delta_t)
    #plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Feature_importance_%imin_long.pdf" % delta_t),orientation="portrait")

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
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Feature_importance_%imin.pdf" % delta_t),orientation="portrait")

## Get columns not containing TRT of the future:
def get_X_col(colname):
    ## No future TRT observations in input:
    use_col = ("TRT" not in colname or \
               "-" in colname or "|0" in colname) and \
              ("TRT_Rank_diff" not in colname)
    return(use_col)

## Get feature ranking for the complete dataset:
def get_feature_importance(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path):
    print("Get features for lead time t0 + %imin" % pred_dt)
    
    ## Check whether model already exists:
    if os.path.exists(os.path.join(model_path,"model_%i_t0diff_maxdepth6.pkl" % pred_dt)):
        use_existing = ""
        while (use_existing!="y" and use_existing!="n"):
            use_existing = raw_input("  Model exists alreay, fit a new one? [y/n] ")
        if use_existing=="n":
            print("  Use existing one, return from this function")
            return
    
    ## Delete rows with TRT Rank close to zero at lead time:
    print("  Delete rows with TRT Rank close to zero at lead time")
    X = df_nonnan_nonzerot0.loc[df_nonnan_nonzerot0["TRT_Rank|%i" % pred_dt]>=0.15,
                                [Xcol for Xcol in df_nonnan_nonzerot0.columns if get_X_col(Xcol)]]
    y = df_nonnan_nonzerot0.loc[df_nonnan_nonzerot0["TRT_Rank|%i" % pred_dt]>=0.15,
                                ["TRT_Rank_diff|%i" % pred_dt]]
    del(df_nonnan_nonzerot0)
    
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
    with open(os.path.join(model_path,"model_%i_t0diff_maxdepth6.pkl" % pred_dt),"wb") as file:
        pickle.dump(xgb_model,file,protocol=-1)
        
    ## Plot feature importance:
    print("  Plot feature importance")
    plot_feature_importance(xgb_model,X,pred_dt,cfg_tds)

## Get Mean Square Error depending on number of features:
def get_mse_from_n_feat(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path):
    print("Get dependence of MSE on n features for lead time t0 + %imin" % pred_dt)
    
    ## Check whether data on MSE already exists:
    if os.path.exists(os.path.join(model_path,"MSE_feature_count_gain_%i.pkl" % pred_dt)):
        use_existing = ""
        while (use_existing!="y" and use_existing!="n"):
            use_existing = raw_input("  MSE data exists alreay, get new one? [y/n] ")
        if use_existing=="n":
            print("  Use existing one, return from this function")
            return
            
    ## Delete rows with TRT Rank close to zero at lead time:
    print("  Delete rows with TRT Rank close to zero at lead time")
    X = df_nonnan_nonzerot0.loc[df_nonnan_nonzerot0["TRT_Rank|%i" % pred_dt]>=0.15,
                                [Xcol for Xcol in df_nonnan_nonzerot0.columns if get_X_col(Xcol)]]
    y = df_nonnan_nonzerot0.loc[df_nonnan_nonzerot0["TRT_Rank|%i" % pred_dt]>=0.15,
                                ["TRT_Rank_diff|%i" % pred_dt]]
    del(df_nonnan_nonzerot0)
    
    ## Split training/testing data:
    print("  Split into training/testing data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del(X,y)

    ## Load XGBmodel:
    print("  Load XGBmodel")
    with open(os.path.join(model_path,"model_%i_t0diff_maxdepth6.pkl" % pred_dt),"rb") as file:
        xgb_model = pickle.load(file)

    ## Order features by importance (gain):
    top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                               orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                               ascending=False)

    ## Create list of number of features to select for the fitting:
    n_feat_arr = np.concatenate([np.arange(10)+1,
                                 np.arange(12,50,2),
                                 np.arange(50,100,10),
                                 np.arange(100,520,20)])
    
    ## Get models fitted with n top features:
    print("  Get models fitted with n top features")
    ls_models = [fit_model_n_feat(X_train, y_train, top_features_gain, n_feat, n_feat_arr) for n_feat in n_feat_arr]
    print("    Save list of models as pickle to disk")
    with open(os.path.join(model_path,"models_%i_t0diff_maxdepth6_nfeat.pkl" % pred_dt),"wb") as file:
        pickle.dump(ls_models, file, protocol=-1)
    #with open(os.path.join(model_path,"models_%i_t0diff_maxdepth6_nfeat.pkl" % pred_dt),"rb") as file:
    #    ls_models = pickle.load(file)
    
    ## Get mean square error of models with n features:
    print("  Get mean square error of models with n features")
    MSE_gain_ls = [mse_n_feat(X_test, y_test, top_features_gain, n_feat, model) for n_feat, model in zip(n_feat_arr,ls_models)]    
    df_mse_feat_count = pd.DataFrame.from_dict({"Feature Count": n_feat_arr,
                                                "MSE %imin" % pred_dt: MSE_gain_ls})
    df_mse_feat_count.set_index("Feature Count",inplace=True)
    print("    Save dataframe with MSE to disk")
    with open(os.path.join(model_path,"MSE_feature_count_gain_%i.pkl" % pred_dt),"wb") as file: pickle.dump(df_mse_feat_count,file,protocol=-1)
    
    ## Append MSE values to existing HDF5 file (if existing):
    print("  Append MSE values to HDF5 file")
    df_mse_feat_count.to_hdf(os.path.join(model_path,"MSE_feature_count_gain.h5"),
                             key="MSE_%imin" % pred_dt, mode="a", format="t", append=True)
    #if not os.path.exists(os.path.join(model_path,"MSE_feature_count_gain.pkl")):
    #    df_mse_feat_count.to_hdf(os.path.join(model_path,"MSE_feature_count_gain.pkl"),
    #                             key="MSE",mode="w",complevel=0,format='table')
    #else:
    #    store = pd.HDFStore(os.path.join(model_path,"MSE_feature_count_gain.pkl"))
    #    store.append(df_mse_feat_count, ohlcv_candle, format='t',  data_columns=True)
        
## Plot dependence of MSE on feature number:
def plot_mse_from_n_feat(ls_pred_dt,cfg_tds,model_path,thresholds=None):
    print("Plot MSE as function of number of features used")
    
    ## Read in MSE values:
    print("  Reading MSE values from disk")
    ls_df_mse_feat_count = [pd.read_hdf(os.path.join(model_path,"MSE_feature_count_gain.h5"),
                                        key="MSE_%imin" % pred_dt, mode="r") for pred_dt in ls_pred_dt]
    df_mse_feat_count = pd.concat(ls_df_mse_feat_count, axis=1)
    
    ## Plot the figure:
    print("  Plotting the figure")
    if len(ls_pred_dt) == 2:
        col10 = '#E69F00'
        col30 = '#D55E00'
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
    else:
        df_mse_feat_count.plot.lines()
        plt.show()
        
## Fit model with threshold number of features (selected by the user):
def plot_pred_vs_obs(df_nonnan_nonzerot0,pred_dt,n_feat,cfg_tds,model_path):
    print("\nPlot predicted vs. observed testing samples for time delta %imin (%i features)" % (pred_dt,n_feat))
    
    ## Delete rows with TRT Rank close to zero at lead time:
    print("  Delete rows with TRT Rank close to zero at lead time")
    X = df_nonnan_nonzerot0.loc[df_nonnan_nonzerot0["TRT_Rank|%i" % pred_dt]>=0.15,
                                [Xcol for Xcol in df_nonnan_nonzerot0.columns if get_X_col(Xcol)]]
    y = df_nonnan_nonzerot0.loc[df_nonnan_nonzerot0["TRT_Rank|%i" % pred_dt]>=0.15,
                                ["TRT_Rank_diff|%i" % pred_dt]]
    del(df_nonnan_nonzerot0)
    
    ## Split training/testing data:
    print("  Split into training/testing data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del(X,y)
    
    precalc_n_feat = np.concatenate([np.arange(10)+1, np.arange(12,50,2),
                                     np.arange(50,100,10), np.arange(100,520,20)])
    if n_feat not in precalc_n_feat:
        ## Load XGBmodel:
        print("  Load XGBmodel")
        with open(os.path.join(model_path,"model_%i_t0diff_maxdepth6.pkl" % pred_dt),"rb") as file:
            xgb_model = pickle.load(file)

        ## Order features by importance (gain):
        top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                                   orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                                   ascending=False)
                                                   
        ## Fit model:
        model    = fit_model_n_feat(X_train, y_train, top_features_gain, n_feat, np.array(n_feat))
    else:
        with open(os.path.join(model_path,"models_%i_t0diff_maxdepth6_nfeat.pkl" % pred_dt),"rb") as file:
            model = pickle.load(file)[np.where(precalc_n_feat==n_feat)[0][0]]
        
    ## Get features:
    features  = model.get_booster().feature_names
    
    ## Make prediction and get skill scores:
    pred_gain = model.predict(X_test[features])
    mse_gain  = met.mean_squared_error(y_test[["TRT_Rank_diff|%i" % pred_dt]],pred_gain)
    r2_gain   = met.r2_score(y_test[["TRT_Rank_diff|%i" % pred_dt]],pred_gain)
    
    ## Save the model to disk:
    with open(os.path.join(model_path,"model_%i_t0diff_maxdepth6_%ifeat_gain.pkl" % (pred_dt,n_feat)),"wb") as file:
        pickle.dump(model,file,protocol=-1)
    
    ## Make the plot:
    print("  Making the plot")
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[10,8])
    axes.set_ylabel('Predicted TRT Rank difference')
    counts,ybins,xbins,image = axes.hist2d(y_test[["TRT_Rank_diff|%i" % pred_dt]].values[:,0],pred_gain,
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
    axes.set_xlabel(r'Observed TRT Rank difference $t_{+%imin}$' % pred_dt)
    axes.set_title('TRT Ranks differences\nTime delta: %imin' % pred_dt)
    axes.set_aspect('equal'); axes.patch.set_facecolor('0.7')
    str_n_cells  = "Mean Squared Error (MSE): %.2f\n" % (mse_gain)
    str_n_cells += r"Coeff of determination ($R^2$): %.2f" % (r2_gain)
    props = dict(boxstyle='round', facecolor='white')
    axes.text(-1.5,1.5, str_n_cells, bbox=props)
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Pred_scatter_%i.pdf" % pred_dt), orientation="portrait")
    print("    Saved the plot")


    
    
    
    