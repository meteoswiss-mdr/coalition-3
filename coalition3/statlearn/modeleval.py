""" [COALITION3] Code to get long lasting TRT cells """
from __future__ import division
from __future__ import print_function

import os
import pickle
import sklearn
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import skill_metrics as sm
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from scipy import stats
from collections import Counter
from sklearn.neural_network import MLPRegressor

import coalition3.inout.readconfig as cfg
import coalition3.statlearn.inputprep as ipt

from pysteps.postprocessing.probmatching import nonparam_match_empirical_cdf
from coalition3.visualisation.TRTcells import truncate_cmap, plot_var_time_series_dt0_multiquant, contour_of_2dHist

## Uncomment when running on Mac OS:
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plot_var_time_series(TRT_ID_sel, df_nonnan, var, stat, past_dt, dt_highlight=None, TRT_var=None, ax=None):
    date_of_cell = dt.datetime.strptime(TRT_ID_sel["TRT_ID"][:12], "%Y%m%d%H%M")
    if isinstance(dt_highlight, int):
        dt_highlight = [dt_highlight]
    return_ax = False if ax is None else True
    
    ## Find cells where the there are loads of similar TRT Ranks:
    DTI_sel  = [dti for dti in df_nonnan.index.values if dti[13:] in TRT_ID_sel["TRT_ID"]]
    cell_sel = df_nonnan.loc[DTI_sel]
    cell_sel.set_index(pd.to_datetime([dt.datetime.strptime(date[:12],"%Y%m%d%H%M") for date in cell_sel.index]),
                       drop=True,append=False,inplace=True)
                       
    df_var = shift_values_to_timestep(cell_sel,var_name=var,stat_name=stat,TRT_var=TRT_var,dt_sel=past_dt,shift=False)
    if 0 not in past_dt:
        df_var *= -1
    
    if ax is None:
        fig = plt.figure(figsize = [10,5])
        ax  = fig.add_subplot(1,1,1)
    cmap = plt.get_cmap('inferno')
    past_dt_cmap = truncate_cmap(cmap, 0.0, 0.8)
    df_var[[col for col in df_var.columns if var in col]].plot(ax=ax,cmap=past_dt_cmap,linewidth=1,style='.-',alpha=0.8)
    if dt_highlight is not None:
        for line_i, dt_i in zip(ax.lines, past_dt):
            alpha_i = 0.8 if dt_i in dt_highlight else 0.1
            line_i.set_alpha(alpha_i)
            line_i.set_alpha(alpha_i)
    ax.set_ylabel("%s (%s)" % (var,stat))
    ax.grid()
    if return_ax:
        ax.get_legend().remove()
        ax.set_title("%s%s | %s" % (var, " | "+", ".join(str(dt_i)+"min" for dt_i in dt_highlight), stat))
        return ax
    else:
        ax.legend(["%imin" % dt_i for dt_i in past_dt],
                  ncol=3, fontsize="small", title ="%s (%s)" % (var,stat), loc="upper right")
        ax.set_title("TRT Cell %s\n(%i values)" % (TRT_ID_sel["TRT_ID"],TRT_ID_sel["Count"]))
        ax.set_xlabel("Time %s" % dt.datetime.strftime(date_of_cell,"%d.%m.%Y"))
        plt.savefig(os.path.join(cfg_tds["fig_output_path"],"%s_%s_Series_%s.pdf" % (var, stat, TRT_ID_sel["TRT_ID"])))
        plt.close()
                           
def plot_pred_time_series(TRT_ID_sel, df_nonnan, pred_mod_ls, ls_pred_dt, cfg_tds, path_addon="", title_addon=""):
    if len(pred_mod_ls)!=len(ls_pred_dt):
        raise ValueError("Variables 'pred_mod_ls' and 'ls_pred_dt' must have the same length")
    if path_addon is not "":
        path_addon = path_addon+"_"
    
    date_of_cell = dt.datetime.strptime(TRT_ID_sel["TRT_ID"][:12], "%Y%m%d%H%M")

    ## Find cells where the there are loads of similar TRT Ranks:
    DTI_sel  = [dti for dti in df_nonnan.index.values if dti[13:] in TRT_ID_sel["TRT_ID"]]
    cell_sel = df_nonnan.loc[DTI_sel]
    cell_sel.set_index(pd.to_datetime([dt.datetime.strptime(date[:12],"%Y%m%d%H%M") for date in cell_sel.index]),
                       drop=True,append=False,inplace=True)

    df_TRT_shift = shift_values_to_timestep(cell_sel,"TRT_Rank",TRT_var="RANKr")
    df_TRT_shift["RANKr"]/=10.

    for i_pred, pred_dt in enumerate(ls_pred_dt):
        cell_sel["TRT_Rank_pred|%i" % pred_dt] = pred_mod_ls[i_pred].loc[DTI_sel].values
    #with open("TRT_Rank_Pred_%s%s.pkl" % (path_addon,TRT_ID_sel["TRT_ID"]),"wb") as file:
    #    pickle.dump(cell_sel, file, protocol=2)
    df_TRT_pred_shift = shift_values_to_timestep(cell_sel,"TRT_Rank_pred")

    fig = plt.figure(figsize = [10,5])
    ax  = fig.add_subplot(1,1,1)
    fcst_lines = df_TRT_pred_shift.plot(ax=ax,cmap="viridis_r",linewidth=1,style='.-') #,legend=False)
    for line_i, alpha_i in zip(fig.gca().lines, [.1,.8,.1,.8,.1,.8,.1,.1,.1]):
        line_i.set_alpha(alpha_i)
    
    cmap_pred_dt = plt.cm.get_cmap('viridis_r')
    for pred_dt in [10,20,30]:
        for timepoint in cell_sel.index:
            line_col = cmap_pred_dt((pred_dt-5.)/45)
            #line_col = [rgb_val for rgb_val in cmap_pred_dt(float(pred_dt)/90)[:3]]
            #line_col.append(0.8)
            t0 = timepoint
            t_pred = t0+dt.timedelta(minutes=pred_dt)
            TRT_0 = df_TRT_shift["TRT_Rank|0"].loc[t0]
            TRT_pred = df_TRT_pred_shift["TRT_Rank_pred|%i" % pred_dt].loc[t_pred]
            df_line = pd.DataFrame([TRT_0,TRT_pred],
                                    index=pd.DatetimeIndex([t0,t_pred],freq='%iT' % pred_dt))
            df_line.plot.line(ax=ax,color=line_col,linewidth=0.5,legend=False)
    
    df_TRT_shift[["RANKr"]].plot.line(ax=ax, color="darkgrey",linewidth=1.5,style='.-')
    df_TRT_shift[["TRT_Rank|0"]].plot.line(ax=ax, color="black",linewidth=2,style='.-')
    legend_TRT = plt.legend(ax.get_lines()[-2:],[r"TRT Rank $t_0$ (TRT)",r"TRT Rank $t_0$ (COALITION3)"], fontsize="small", loc="lower right")
    
    
    ax.axhspan(1.2, 1.5, 0, 0.02, edgecolor="grey", facecolor='white') #, alpha=0.15)
    ax.axhspan(1.5, 2.5, 0, 0.02, edgecolor="grey", facecolor='green') #, alpha=0.15)
    ax.axhspan(2.5, 3.5, 0, 0.02, edgecolor="grey", facecolor='yellow') #, alpha=0.15)
    ax.axhspan(3.5, 4.0, 0, 0.02, edgecolor="grey", facecolor='red') #, alpha=0.15)
    ax.set_ylim([0, np.min([4,np.max([df_TRT_shift.max().max(), df_TRT_pred_shift.max().max()])*1.1+0.1])])
    
    ax.legend(ax.get_lines()[:9],["%imin" % pred_dt for pred_dt in ls_pred_dt],
              ncol=3, fontsize="small", title ="Forecast leadtime", loc="upper right")
    plt.gca().add_artist(legend_TRT)
    plt.ylabel("TRT Rank")
    plt.xlabel("Time %s" % dt.datetime.strftime(date_of_cell,"%d.%m.%Y"))
    plt.grid()
    plt.title("TRT Cell %s%s\n(%i values)" % (TRT_ID_sel["TRT_ID"],title_addon,TRT_ID_sel["Count"]))
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"TRT_Series_%s%s.pdf" % (path_addon,TRT_ID_sel["TRT_ID"])))
    plt.close()

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
    
## Get colorbar for Taylor-Diagram:
def get_time_delta_colorbar(fig, ls_pred_dt, cmap_pred_dt, loc):
    cmaplist = [cmap_pred_dt(i) for i in range(cmap_pred_dt.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('TimeDel_cmap', cmaplist, cmap_pred_dt.N)
    bounds = np.diff(ls_pred_dt)#np.arange(2.5,52.5,5)
    diff = np.diff(ls_pred_dt)
    diff = np.concatenate([np.array([diff[0],]), diff])
    bounds =  ls_pred_dt-diff/2.
    bounds = np.append(bounds,ls_pred_dt[-1]+diff[-1]/2.)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax2 = fig.add_axes(loc)
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
    spacing='proportional', ticks=ls_pred_dt, boundaries=bounds, format='%1i')
    ax2.set_ylabel('Lead time [min]') #, size=12)
   
def shift_values_to_timestep(df, var_name, stat_name = "", TRT_var=None, dt_sel=None, shift=True):
    all_timesteps = pd.date_range(start = df.index[ 0] - dt.timedelta(minutes=45),
                                  end   = df.index[-1] + dt.timedelta(minutes=45),
                                  freq = '5min')
    df = pd.concat([pd.DataFrame([], index = all_timesteps),df], axis=1, sort=True)

    df_var = df[[colname_var for colname_var in df.columns if (var_name+"|" in colname_var) and (stat_name in colname_var)]]
    if df_var.shape[1]==0:
        print("   *** Warning: No variable %s found in column names, returning empty df ***" % var_name)
        return df_var

    ls_var_new = []
    if TRT_var is not None:
        ls_var_new.append(df[TRT_var])
        
    time_delta = np.arange(-45,50,5) if dt_sel is None else dt_sel
    if shift:
        for i, time_del in enumerate(time_delta):
            df_var_dt = df_var[[colname_dt for colname_dt in df_var.columns if "|%i" % time_del in colname_dt]]
            if df_var_dt.shape[1]!=0:
                df_var_dt_short = df_var_dt.iloc[9:-9,:].values[:,0]
                ls_var_new.append(pd.DataFrame(df_var_dt_short, columns=df_var_dt.columns,
                                               index=df_var_dt.index[i:i+len(df_var_dt_short)]))
        df_var_shift = pd.concat(ls_var_new, axis=1, sort=True)
    else:
        df_var_shift = df_var[[colname_dt for colname_dt in df_var.columns if np.any(["|%i" % time_del in colname_dt for time_del in time_delta])]]
    return df_var_shift
    
def get_R2_param(obs, pred):
    slope1, intercept, r_value, p_value, std_err = stats.linregress(pred,obs)
    slope2, intercept, r_value, p_value, std_err = stats.linregress(obs,pred)
    columns = ["R^2_fun","corr","s_o","s_y","s_y/s_o","bias","(beta(x=pred,y=obs)-1)^2",
               "beta(x=obs,y=pred)","MSE","MSE_clim","R^2_self","BSS_pre"]
    params = [
        sklearn.metrics.r2_score(obs,pred),
        np.corrcoef(obs,pred)[0,1],
        np.std(obs),
        np.std(pred),
        np.std(pred)/np.std(obs),
        (np.mean(pred)-np.mean(obs)), #intercept**2,
        (slope1-1)**2,
        slope2,
        np.mean((obs-pred)**2),
        np.mean((obs-np.mean(obs))**2),
        1 - (np.sum((obs-pred)**2)/np.sum((obs-np.mean(obs))**2)),
        1 - np.sum((obs-pred)**2)/(len(obs) * np.std(obs)**2)
      ]
    df_param = pd.DataFrame(columns=columns)
    df_param.loc[0] = params
    return(df_param)

def plot_stats(df_R2_param, df_name, cfg_tds):
    df_R2_param["BSS"] = df_R2_param["corr"]**2 - \
                              (df_R2_param["s_y/s_o"]**2 * df_R2_param["(beta(x=pred,y=obs)-1)^2"]) - \
                              (df_R2_param["bias"]**2 / df_R2_param["s_o"]**2)
                              #(df_R2_param["corr"] - df_R2_param["s_y/s_o"])**2 -\

    fig = plt.figure(figsize = [12,8])
    ax_BSS = fig.add_subplot(2,3,1)
    df_R2_param[["BSS"]].plot(ax=ax_BSS,style=".-")
    ax_corr = fig.add_subplot(2,3,2)
    df_R2_param[["corr"]].plot(ax=ax_corr,style=".-")
    ax_std = fig.add_subplot(2,3,3)
    df_R2_param[["s_o","s_y","s_y/s_o"]].plot(ax=ax_std,style=".-")
    ax_beta = fig.add_subplot(2,3,4)
    df_R2_param[["beta(x=obs,y=pred)"]].plot(ax=ax_beta,style=".-")
    df_R2_param[["(beta(x=pred,y=obs)-1)^2"]].plot(ax=ax_beta,style=".-", secondary_y=True)
    ax_bias = fig.add_subplot(2,3,5)
    df_R2_param[["bias"]].plot(ax=ax_bias,style=".-")
    ax_SS = fig.add_subplot(2,3,6)
    df_R2_param[["MSE","MSE_clim"]].plot(ax=ax_SS,style=".-")
    for ax in [ax_BSS,ax_corr,ax_std,ax_beta,ax_bias,ax_SS]:
        ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Stats_%s.pdf" % df_name))
    plt.close()
    

def plot_stats_nice(df_R2_param, df_name, cfg_tds):
    df_R2_param["corr^2"] = df_R2_param["corr"].values**2
    df_R2_param["corr^2_pen"] = df_R2_param["corr^2"].values - \
                                (df_R2_param["corr"].values - df_R2_param["s_y/s_o"].values)**2
    
    fig = plt.figure(figsize = [8,6])
    ax_corr = fig.add_subplot(2,2,1)
    df_R2_param[["corr^2", "corr^2_pen", "R^2_fun"]].plot(ax=ax_corr,style=".-", cmap="copper_r") #,label=r"$\rho^2$")
    #df_R2_param[["corr^2_pen"]].plot(ax=ax_corr,style=".-",label=r"$\rho^2$ (cond bias penalised)")
    #df_R2_param[["R^2_fun"]].plot(ax=ax_corr,style=".-",label=r"$R^2$ (Brier Score)")
    ax_corr.legend([r"$\rho^2$", r"$\rho^2$ (cond bias penalised)", r"$R^2$ (Brier Score)"]);

    ax_std = fig.add_subplot(2,2,2)
    cmap_std = plt.cm.get_cmap('tab10')
    df_R2_param[["s_o","s_y","s_y/s_o"]].plot(ax=ax_std,style=".-", color=[cmap_std(0), cmap_std(3), cmap_std(4)])
    ax_std.legend([r"$s_{obs}$",r"$s_{pred}$","$s_{pred} / s_{obs}$"]);
    
    ax_MSE = fig.add_subplot(2,2,3)
    df_R2_param[["MSE","MSE_clim"]].plot(ax=ax_MSE,style=".-")
    ax_MSE.legend([r"$MSE$",r"$MSE_{clim}$"]);
    
    ax_bias = fig.add_subplot(2,2,4)
    df_R2_param["bias"].plot(ax=ax_bias,style=".-")
    ax_bias.legend([r"Bias (pred - obs)"]);

    cmap_xgrid = plt.cm.get_cmap('viridis_r')
    for ax in [ax_corr,ax_std,ax_MSE,ax_bias]:
        ax.set_xlabel("Lead time")
        ax.grid()
        ax.grid(axis='x',linewidth=4,alpha=0.5)
        n_xticks = range(len(ax.get_xticklines()))
        for i, tick, label, gridl in zip(n_xticks, ax.get_xticklines(), ax.get_xticklabels(), ax.get_xgridlines()):
            col_tick = cmap_xgrid(float(i)/len(ax.get_xticklines()))
            tick.set_color(col_tick)
            label.set_color(col_tick)
            gridl.set_color(col_tick)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Stats_nice_%s.pdf" % df_name))
    plt.close()
    

def plot_hist_probmatch(Rank_pred_XGB_df, Rank_pred_XGB_PM_df):
    df_pred_10    = Rank_pred_XGB_df["TRT_Rank_pred|5"]
    df_pred_30    = Rank_pred_XGB_df["TRT_Rank_pred|45"]
    df_pred_PM_10 = Rank_pred_XGB_PM_df["TRT_Rank_pred_PM|5"]
    df_pred_PM_30 = Rank_pred_XGB_PM_df["TRT_Rank_pred_PM|45"]
    #Stack the data
    fig = plt.figure(figsize=[7,5])
    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[1,1],
                           height_ratios=[1,1,20]
                           )
    gs.update(wspace=.325, hspace=0.05)
    ax0 = plt.subplot(gs[0])
    ax0.boxplot(df_pred_10[df_pred_10.notnull()], vert=False, flierprops = dict(markeredgecolor="white", alpha=0),whis=[5,95])
    ax0.axvline(np.median(df_pred_10[df_pred_10.notnull()]), color=cmap_pred_dt(1/9), linewidth=1.5)
    ax0.axis('off')
    ax0.set_title("Predicted TRT Rank\n ")
    ax0.set_ylabel("+5min")

    ax1 = plt.subplot(gs[1])
    ax1.boxplot(df_pred_PM_10[df_pred_PM_10.notnull()], vert=False, flierprops = dict(markeredgecolor="white", alpha=0),whis=[5,95])
    ax1.axvline(np.median(df_pred_PM_10[df_pred_PM_10.notnull()]), color=cmap_pred_dt(1/9), linewidth=1.5)
    ax1.axis('off')
    ax1.set_title("Predicted TRT Rank\nProbability matched (PM)")
    ax1.set_ylabel("+5min")

    ax2 = plt.subplot(gs[2])
    ax2.boxplot(df_pred_30[df_pred_30.notnull()], vert=False, flierprops = dict(markeredgecolor="white", alpha=0),whis=[5,95])
    ax2.axvline(np.median(df_pred_30[df_pred_30.notnull()]), color=cmap_pred_dt(9/9), linewidth=1.5)
    ax2.axis('off')
    ax2.set_ylabel("+45min")

    ax3 = plt.subplot(gs[3])
    ax3.boxplot(df_pred_PM_30[df_pred_PM_30.notnull()], vert=False, flierprops = dict(markeredgecolor="white", alpha=0),whis=[5,95])
    ax3.axvline(np.median(df_pred_PM_30[df_pred_PM_30.notnull()]), color=cmap_pred_dt(9/9), linewidth=1.5)
    ax3.axis('off')
    ax3.set_ylabel("+45min")

    ax4 = plt.subplot(gs[4])
    ax4.hist([df_pred_10[df_pred_10.notnull()],df_pred_30[df_pred_30.notnull()]], bins=20, color=[cmap_pred_dt(1/9),cmap_pred_dt(9/9)])

    ax5 = plt.subplot(gs[5])
    ax5.hist([df_pred_PM_10[df_pred_PM_10.notnull()],df_pred_PM_30[df_pred_PM_30.notnull()]], bins=20, color=[cmap_pred_dt(1/9),cmap_pred_dt(9/9)])
    for ax in [ax4,ax5]:
        ax.set_xlabel("TRT Rank")
        ax.set_ylabel("Count")
        ax.legend(['+5min', '+45min'])
        ax.grid()
    #plt.tight_layout()
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Hist_TRT_pred_5_45.pdf")) 

## Core function producing scatter plot of future observed and predicted TRT Rank (changes):
def plot_pred_vs_obs_core(y_test,pred_gain,pred_dt,mod_name,cfg_tds,outtype="TRT_Rank_diff"):
    print("  Making the plot")
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[10,8])
    if outtype=="TRT_Rank_diff":
        fig_range = [-2.5,2.5]
        print_str = "difference "
        save_str = "_diff"
    elif outtype=="TRT_Rank":
        fig_range = [0,4]
        print_str = ""
        save_str = "_rank"

    if len(y_test)>1000:
        counts,ybins,xbins,image = axes.hist2d(y_test.values,pred_gain,
                                               bins=200,range=[fig_range,fig_range],cmap="magma",norm=mcolors.LogNorm())
        #counts,ybins,xbins,image = axes.hist2d(y_test[["TRT_Rank_diff|%i" % pred_dt]].values[:,0],pred_gain,
        #                                       bins=200,range=[[-2.5,2.5],[-2.5,2.5]],cmap="magma",norm=mcolors.LogNorm())
        cbar = fig.colorbar(image, ax=axes, extend='max')
    else:
        axes.scatter(y_test[["%s|%i" % (outtype,pred_dt)]].values[:,0],pred_gain,
                     marker="+", color="black", s=8)
    axes.set_xlim(fig_range); axes.set_ylim(fig_range)
    #cbar.set_label('Number of cells per bin of size [0.02, 0.02]', rotation=90)
    axes.grid()
    #axes.fill_between([-0.2,0.2],y1=[-1.5,-1.5], y2=[1.5,1.5], facecolor="none", hatch="X", edgecolor="darkred", linewidth=0.5)
    axes.plot(fig_range,fig_range,'w--',linewidth=2) #,facecolor="w",linewidth=2,linestyle='--')
    if len(y_test)>1000:
        cont2d_1, lvl = contour_of_2dHist(counts,smooth=True)
        CS = axes.contour(cont2d_1,levels=lvl,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=2,cmap="YlGn_r")
        CS_lab = axes.clabel(CS, inline=1, fontsize=10, fmt='%i%%', colors="black")
        #[txt.set_backgroundcolor('white') for txt in CS_lab]
        [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.3, boxstyle='round', alpha=0.71)) for txt in CS_lab] #pad=0,
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test.values,pred_gain)
    axes.plot(fig_range,np.array(fig_range)*slope+intercept,'darkred',linewidth=2)
    
    axes.set_xlabel(r'Observed TRT Rank %st$\mathregular{_{+%imin}}$' % (print_str,pred_dt))
    axes.set_ylabel(r'Predicted TRT Rank %st$\mathregular{_{+%imin}}$' % (print_str,pred_dt))
    model_title = "" if mod_name == "" else r" | Mod$\mathregular{_{%s}}$" % mod_name[1:]
    title_str = 'TRT Ranks %s\nTime delta: %imin' % (print_str,pred_dt)
    title_str += model_title
    axes.set_title(title_str)
    axes.set_aspect('equal'); axes.patch.set_facecolor('0.71')
    
    mse_gain = sklearn.metrics.mean_squared_error(y_test.values,pred_gain)
    r2_gain  = sklearn.metrics.r2_score(y_test.values,pred_gain)
    str_n_cells  = "Mean Squared Error (MSE): %.2f\n" % (mse_gain)
    str_n_cells += r"Coeff of determination (R$\mathregular{^2}$): %.2f" % (r2_gain); str_n_cells += "\n"
    str_n_cells += r"Regression intercept ($\mathregular{\beta_0}$): %.2f" % (intercept); str_n_cells += "\n"
    str_n_cells += r"Regression slope ($\mathregular{\beta_1}$): %.2f" % (slope)
    props = dict(boxstyle='round', facecolor='white')
    axes.text(fig_range[0]+0.25, fig_range[1]-0.25, str_n_cells, bbox=props,
              horizontalalignment='left',verticalalignment='top')
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Pred_scatter%s_%i%s.pdf" % (save_str,pred_dt,mod_name.replace("|","-"))), orientation="portrait")
    print("    Saved the plot")
    plt.close()
    
## Wrapper function for all the model evaluations:
def make_model_evaluation(df_nonnan, model_path, ls_pred_dt, cfg_tds, cfg_op):
    X_test_ls = []; y_test_ls = []
    cmap_pred_dt = plt.cm.get_cmap('viridis_r')
    
    ## Import dictionary with selected models:
    train_path_name = os.path.join(model_path,"model_dict_t0diff_maxdepth6_selfeat_gain.pkl")
    with open(train_path_name,"rb") as file:
        dict_sel_model = pickle.load(file)

    plt.close()
    fig = plt.figure(num=1, figsize=(7, 6))
    
    ## Loop over lead times:
    for i, pred_dt in enumerate(ls_pred_dt):

        if i==0:
            xgb_model_ls = []; pred_model_ls = []; Rank_obs_ls = []
            top_features_ls = []
            df_param_ls_diff = []
            df_param_ls_rank = []
            df_param_ls_rank_PM = []
            df_param_ls_rank_pers = []
            Rank_pred_XGB_ls = []
            Rank_pred_XGB_PM_ls = []
        
        if len(X_test_ls)==len(ls_pred_dt) and len(y_test_ls)==len(ls_pred_dt):
            X_test = X_test_ls[i]
            y_test = y_test_ls[i]
        else:
            if i==0:
                X_test_ls = []; y_test_ls = []
            X_train, X_test, y_train, y_test = ipt.get_model_input(df_nonnan,
                del_TRTeqZero_tpred=True, split_Xy_traintest=True, X_normalise=False,
                pred_dt=pred_dt, check_for_nans=False, verbose=True)
            del(X_train, y_train)
            X_test_ls.append(X_test)
            y_test_ls.append(y_test)
        
        ## Load XGB model fitted to all features:
        with open(os.path.join(model_path,"model_%i_t0diff_maxdepth6.pkl" % pred_dt),"rb") as file:
            xgb_model_feat = pickle.load(file)
        xgb_model_ls.append(xgb_model_feat)
                
        top_features = pd.DataFrame.from_dict(xgb_model_feat.get_booster().get_score(importance_type='gain'),
                                                 orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                                 ascending=False)
        top_features_ls.append(top_features)
        
        ## Get specific predictive model for this leadtime:
        pred_model = dict_sel_model["pred_mod_%i" % pred_dt]
        pred_model_ls.append(pred_model)
    
        ## Check that features agree:
        features_pred_model = pred_model.get_booster().feature_names
        n_features = len(features_pred_model)
        if set(features_pred_model)!=set(top_features.index[:n_features]):
            raise ValueError("Features of predictive model and top features of model fitted with all features do not agree")
    
        ## Make prediction of TRT Rank differences:
        TRT_diff_pred = pred_model.predict(X_test[features_pred_model]) 
        
        ## Get set of different TRT Rank predictions:
        Rank_obs, Rank_pred_XGB, Rank_pred_XGB_PM, Rank_pred_pers, Rank_pred_pers_PM, \
            Rank_pred_diff, Diff_pred_XGB = get_obs_fcst_TRT_Rank(X_test["TRT_Rank|0"], TRT_diff_pred, y_test, X_test["TRT_Rank|-5"])
        Rank_obs_ls.append(Rank_obs)
        Rank_pred_XGB_ls.append(Rank_pred_XGB)
        Rank_pred_XGB_PM_ls.append(Rank_pred_XGB_PM)
    
        ## Plot scatterplots obs vs. predicted:
        plot_pred_vs_obs_core(y_test,Diff_pred_XGB.values,pred_dt,             "_XGB%i" % n_features,           cfg_tds,outtype="TRT_Rank_diff")
        plot_pred_vs_obs_core(Rank_obs,Rank_pred_XGB.values,pred_dt,    "_XGB%i" % n_features,           cfg_tds,outtype="TRT_Rank")
        plot_pred_vs_obs_core(Rank_obs,Rank_pred_XGB_PM.values,pred_dt, "_XGB%i-ProbMatch" % n_features, cfg_tds,outtype="TRT_Rank")
        plot_pred_vs_obs_core(Rank_obs,Rank_pred_pers.values,pred_dt,   "_Pers",                         cfg_tds,outtype="TRT_Rank")
        plot_pred_vs_obs_core(Rank_obs,Rank_pred_pers_PM.values,pred_dt,"_Pers-ProbMatch",               cfg_tds,outtype="TRT_Rank")
        plot_pred_vs_obs_core(Rank_obs,Rank_pred_diff.values,pred_dt,   "_ConstDiff",                    cfg_tds,outtype="TRT_Rank")
        
        ## Calculate different term elements for R^2 / Brier Score calculation:
        df_param_ls_diff.append(get_R2_param(y_test.values, Diff_pred_XGB.values))
        df_param_ls_rank.append(get_R2_param(Rank_obs.values, Rank_pred_XGB.values))
        df_param_ls_rank_PM.append(get_R2_param(Rank_obs.values, Rank_pred_XGB_PM.values))
        df_param_ls_rank_pers.append(get_R2_param(Rank_obs.values, Rank_pred_pers.values))
    
        ## Calculate statistics for Taylor Diagram:
        stat_pred_XGB          = sm.taylor_statistics(predicted=Rank_pred_XGB.values, reference=Rank_obs.values)
        stat_pred_XGB_PM       = sm.taylor_statistics(predicted=Rank_pred_XGB_PM.values, reference=Rank_obs.values)
        stat_pred_pred_pers    = sm.taylor_statistics(predicted=Rank_pred_pers.values, reference=Rank_obs.values)
        stat_pred_pred_diff    = sm.taylor_statistics(predicted=Rank_pred_diff.values, reference=Rank_obs.values)
        stat_pred_pred_pers_PM = sm.taylor_statistics(predicted=Rank_pred_pers_PM.values, reference=Rank_obs.values)
        
        sdev  = np.array([stat_pred_XGB['sdev'][0], stat_pred_XGB['sdev'][1], stat_pred_XGB_PM['sdev'][1], stat_pred_pred_pers['sdev'][1]])
        crmsd = np.array([stat_pred_XGB['crmsd'][0], stat_pred_XGB['crmsd'][1], stat_pred_XGB_PM['crmsd'][1], stat_pred_pred_pers['crmsd'][1]])
        ccoef = np.array([stat_pred_XGB['ccoef'][0], stat_pred_XGB['ccoef'][1], stat_pred_XGB_PM['ccoef'][1], stat_pred_pred_pers['ccoef'][1]])
        #sdev  = np.array([stat_pred_XGB['sdev'][0], stat_pred_XGB['sdev'][1], stat_pred_XGB_PM['sdev'][1], stat_pred_pred_pers['sdev'][1], stat_pred_pred_diff['sdev'][1]])
        #crmsd = np.array([stat_pred_XGB['crmsd'][0], stat_pred_XGB['crmsd'][1], stat_pred_XGB_PM['crmsd'][1], stat_pred_pred_pers['crmsd'][1], stat_pred_pred_diff['crmsd'][1]])
        #ccoef = np.array([stat_pred_XGB['ccoef'][0], stat_pred_XGB['ccoef'][1], stat_pred_XGB_PM['ccoef'][1], stat_pred_pred_pers['ccoef'][1], stat_pred_pred_diff['ccoef'][1]])

        ## Plot Taylor Diagram:
        col_point = cmap_pred_dt(float(i)/len(ls_pred_dt))
        col_point = (col_point[0], col_point[1], col_point[2], 0.8)

        plot_markerLabel = ["Obs","+%imin" % pred_dt,"",""] 
        plot_markerLabelColor = "black"
        if i == 0:
            plot_markerLegend = 'on'
            plot_overlay = 'off'
        else:
            plot_markerLegend = "on"
            plot_overlay = 'on'
            #plot_markerLabelColor = None
            if i == len(ls_pred_dt)-1:
                plot_markerLabelColor = None
                plot_markerLabel = ["Obs","XGB","XGB (PM)","Persistance"]
                
        sm.taylor_diagram(sdev/sdev[0], crmsd, ccoef, styleOBS = '-', colOBS = 'darkred', markerobs = 'o', titleOBS = 'Obs',
                            markerLabel = plot_markerLabel, markerLabelColor = plot_markerLabelColor, alpha = 0.1, 
                            markerColor = col_point, markerLegend = plot_markerLegend, axismax = 1.2, markerSize = 5,
                            colRMS = 'grey', styleRMS = '--',  widthRMS = 0.8, rincRMS = 0.25, tickRMS = np.arange(0.25, 1.5, 0.25), #titleRMSangle = 110,
                            colSTD = 'grey', styleSTD = '-.', widthSTD = 0.8,
                            colCOR = 'grey', styleCOR = ':', widthCOR = 0.8, 
                            overlay = plot_overlay)

    ## Save Taylor Diagram:
    get_time_delta_colorbar(fig, ls_pred_dt, cmap_pred_dt, [0.7, 0.5, 0.05, 0.3])
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Taylor_Diagram_cmap.pdf"))
    plt.close()

    ## Plot histogram showing the effect of probability matching:
    print("Save dataframe with observed, predicted, and predicted & PM TRT Ranks")
    Rank_obs_df                 = pd.concat(Rank_obs_ls, axis=1, sort=True)
    Rank_obs_df.columns         = ["TRT_Rank_obs|%i" % pred_dt for pred_dt in ls_pred_dt]
    Rank_pred_XGB_df            = pd.concat(Rank_pred_XGB_ls, axis=1, sort=True)
    Rank_pred_XGB_df.columns    = ["TRT_Rank_pred|%i" % pred_dt for pred_dt in ls_pred_dt]
    Rank_pred_XGB_PM_df         = pd.concat(Rank_pred_XGB_PM_ls, axis=1, sort=True)
    Rank_pred_XGB_PM_df.columns = ["TRT_Rank_pred_PM|%i" % pred_dt for pred_dt in ls_pred_dt]
    #plot_hist_probmatch(Rank_pred_XGB_df, Rank_pred_XGB_PM_df)
    Rank_obs_pred_df = pd.concat([Rank_obs_df,Rank_pred_XGB_df,Rank_pred_XGB_PM_df],axis=1,sort=True)

    ## Get dataframe with observed, predicted, and predicted & PM TRT Ranks for operational PM:
    op_path_name = os.path.join(cfg_op["XGB_model_path"],
                                       "TRT_Rank_obs_pred.pkl")
    with open(op_path_name,"wb") as file:
        pickle.dump(Rank_obs_pred_df,file,protocol=2)
    print("  saved dict to 'XGB_model_path' location:\n    %s" % op_path_name)
    prt_txt = """
    ---------------------------------------------------------------------------------
        The file 'TRT_Rank_obs_pred.pkl' in the
        directory '%s'
        is now used for the operational probability matching procedure, be aware of
        that!
    ---------------------------------------------------------------------------------\n""" % (cfg_op["XGB_model_path"])
    print(prt_txt)
    
    ## Plot skill scores as function of lead-time:
    df_R2_param_rank      = pd.concat(df_param_ls_rank,axis=0).set_index(np.array(ls_pred_dt))
    df_R2_param_rank_PM   = pd.concat(df_param_ls_rank_PM,axis=0).set_index(np.array(ls_pred_dt))
    df_R2_param_diff      = pd.concat(df_param_ls_diff,axis=0).set_index(np.array(ls_pred_dt))
    df_R2_param_rank_pers = pd.concat(df_param_ls_rank_pers,axis=0).set_index(np.array(ls_pred_dt))
    plot_stats(df_R2_param_rank, "TRT_Rank", cfg_tds)
    plot_stats(df_R2_param_diff, "TRT_Rank_diff", cfg_tds)
    plot_stats_nice(df_R2_param_rank, "TRT_Rank", cfg_tds)
    plot_stats_nice(df_R2_param_diff, "TRT_Rank_diff", cfg_tds)
    plot_stats_nice(df_R2_param_rank_pers, "TRT_Rank_pers", cfg_tds)
    plot_stats_nice(df_R2_param_rank_PM, "TRT_Rank_PM", cfg_tds)

    ## Print IDs of long TRT cells in testing dataset:
    print("\nThese are the IDs of long TRT cells (>25 time steps) in the testing dataset:")
    TRT_ID = X_test_ls[-1].index
    TRT_ID = [TRT_ID_i[13:] for TRT_ID_i in TRT_ID.values]
    TRT_ID_count = Counter(TRT_ID)
    TRT_ID_count_sort = [(k, TRT_ID_count[k]) for k in sorted(TRT_ID_count, key=TRT_ID_count.get, reverse=True)]
    TRT_ID_count_sort_pd = pd.DataFrame(np.array(TRT_ID_count_sort),columns=["TRT_ID","Count"])
    TRT_ID_count_sort_pd["Count"] = TRT_ID_count_sort_pd["Count"].astype(np.uint16,inplace=True)
    TRT_ID_long = TRT_ID_count_sort_pd.loc[TRT_ID_count_sort_pd["Count"]>25]
    print(TRT_ID_long)
    
    TRT_ID_casestudy = ["2018080721250094","2018080721300099","2018080711400069","2018080710200036"]
    print("  Making analysis for TRT IDs (hardcoded!): %s" % TRT_ID_casestudy)
    
    TRT_ID_long_sel = TRT_ID_long.loc[TRT_ID_long['TRT_ID'].isin(TRT_ID_casestudy)]
    df_feature_ts_plot = pd.DataFrame.from_dict({"Radar":     ["CZC_lt57dBZ|-45|SUM","CZC_lt57dBZ|-45|SUM","CZC_lt57dBZ|-45|SUM"],
                                                 "Satellite": ["IR_097_stat|-20|PERC05","IR_097_stat|-15|PERC01","IR_097_stat|-20|MIN"],
                                                 "COSMO":     ["CAPE_MU_stat|-10|PERC50","CAPE_MU_stat|-5|PERC75","CAPE_ML_stat|0|SUM"],
                                                 "Lightning": ["THX_densIC_stat|-30|SUM","THX_curr_pos_stat|-40|SUM","THX_curr_pos_stat|-30|SUM"]})
    for i_sel in range(len(TRT_ID_long_sel)):
        print("    Working on cell %s" % TRT_ID_long_sel.iloc[i_sel]["TRT_ID"])
        plot_pred_time_series(TRT_ID_long_sel.iloc[i_sel], df_nonnan, Rank_pred_XGB_ls, ls_pred_dt, cfg_tds)
        plot_pred_time_series(TRT_ID_long_sel.iloc[i_sel], df_nonnan, Rank_pred_XGB_PM_ls, ls_pred_dt, cfg_tds, path_addon="PM", title_addon=" (PM)")
        
        plot_var_time_series_dt0_multiquant(TRT_ID_long_sel.iloc[i_sel], df_nonnan, cfg_tds)
            
        for i_pred_dt, pred_dt in enumerate([10,20,30]):
            fig = plt.figure(figsize = [10,6])
            ax_rad  = fig.add_subplot(2,2,1)
            ax_sat  = fig.add_subplot(2,2,2)
            ax_cos  = fig.add_subplot(2,2,3)
            ax_thx  = fig.add_subplot(2,2,4)
            ax_ls = [ax_rad, ax_sat, ax_cos, ax_thx]
            #fig, axes = plt.subplots(2,2)
            #fig.set_size_inches(8,6)
            for i_source, source in enumerate(["Radar","Satellite","COSMO","Lightning"]):
                ls_feat_param = df_feature_ts_plot[source].iloc[i_pred_dt].split("|")
                past_dt = np.arange(-45,0,5) if int(ls_feat_param[1])!=0 else [0]
                ax_ls[i_source] = plot_var_time_series(TRT_ID_long_sel.iloc[i_sel], df_nonnan,
                                                      ls_feat_param[0], ls_feat_param[2], past_dt = past_dt,
                                                      dt_highlight=int(ls_feat_param[1]), ax=ax_ls[i_source])
            plt.tight_layout()
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Feat_series_%i_%s.pdf" % (pred_dt, TRT_ID_long_sel.iloc[i_sel]["TRT_ID"])))
            plt.close()
        


    
    
    
    
    
    