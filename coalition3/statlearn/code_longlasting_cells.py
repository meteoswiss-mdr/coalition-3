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
import matplotlib.gridspec as gridspec

from scipy import stats
from collections import Counter
from sklearn.neural_network import MLPRegressor

import coalition3.inout.readconfig as cfg
import coalition3.statlearn.inputprep as ipt

from coalition3.visualisation.TRTcells import truncate_cmap, plot_var_time_series_dt0_multiquant
from pysteps.postprocessing.probmatching import nonparam_match_empirical_cdf

os.environ['KMP_DUPLICATE_LIB_OK']='True'



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
                           
def plot_pred_time_series(TRT_ID_sel, df_nonnan, pred_mod_ls, ls_pred_dt, path_addon="", title_addon=""):
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
    with open("TRT_Rank_Pred_%s%s.pkl" % (path_addon,TRT_ID_sel["TRT_ID"]),"wb") as file:
        pickle.dump(cell_sel, file, protocol=2)
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
    
    


X_test_45 = X_test_ls[-1]

TRT_ID = X_test_45.index
TRT_ID = [TRT_ID_i[13:] for TRT_ID_i in TRT_ID.values]

TRT_ID_count = Counter(TRT_ID)
TRT_ID_count_sort = [(k, TRT_ID_count[k]) for k in sorted(TRT_ID_count, key=TRT_ID_count.get, reverse=True)]
TRT_ID_count_sort_pd = pd.DataFrame(np.array(TRT_ID_count_sort),columns=["TRT_ID","Count"])
TRT_ID_count_sort_pd["Count"] = TRT_ID_count_sort_pd["Count"].astype(np.uint16,inplace=True)
TRT_ID_count_sort_pd.info()                
TRT_ID_long = TRT_ID_count_sort_pd.loc[TRT_ID_count_sort_pd["Count"]>25]

TRT_ID_casestudy = ["2018080721250094","2018080721300099","2018080711400069","2018080710200036"]
TRT_ID_long_sel = TRT_ID_long.loc[TRT_ID_long['TRT_ID'].isin(TRT_ID_casestudy)]
df_feature_ts_plot = pd.DataFrame.from_dict({"Radar":     ["CZC_lt57dBZ|-45|SUM","CZC_lt57dBZ|-45|SUM","CZC_lt57dBZ|-45|SUM"],
                                             "Satellite": ["IR_097_stat|-20|PERC05","IR_097_stat|-15|PERC01","IR_097_stat|-20|MIN"],
                                             "COSMO":     ["CAPE_MU_stat|-10|PERC50","CAPE_MU_stat|-5|PERC75","CAPE_ML_stat|0|SUM"],
                                             "Lightning": ["THX_densIC_stat|-30|SUM","THX_curr_pos_stat|-40|SUM","THX_curr_pos_stat|-30|SUM"]})
for i_sel in range(len(TRT_ID_long_sel)): #[len(TRT_ID_long)-11]: #range(len(TRT_ID_long)): #[12]: #
    print("Working on cell %s" % TRT_ID_long_sel.iloc[i_sel]["TRT_ID"])
    #plot_pred_time_series(TRT_ID_long_sel.iloc[i_sel], df_nonnan, Rank_pred_XGB_ls, ls_pred_dt)
    #plot_pred_time_series(TRT_ID_long_sel.iloc[i_sel], df_nonnan, Rank_pred_XGB_PM_ls, ls_pred_dt, path_addon="PM", title_addon=" (PM)")
    
    #fig, axes = plt.subplots(2,2)
    #fig.set_size_inches(8,6)
    #for ax,var,stat in zip(axes.flatten(),
    #                       ["RZC_stat","IR_108_stat","CAPE_ML_stat","THX_dens_stat"],
    #                       ["PERC50","PERC01","MEAN","SUM"]):
    #    ax = plot_var_time_series(TRT_ID_long_sel.iloc[i_sel], df_nonnan,
    #                              var,stat,dt_highlight=[0],past_dt=[0], ax=ax)
    #plt.show()
    plot_var_time_series_dt0_multiquant(TRT_ID_long_sel.iloc[i_sel], df_nonnan)
        
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
        

