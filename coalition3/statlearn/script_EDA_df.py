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
import seaborn as sns
import datetime as dt
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.patches as patches

import coalition3.inout.paths as pth
import coalition3.inout.readconfig as cfg
import coalition3.training.preparation as prep
import coalition3.visualisation.TRTcells as visTRT

def get_X_col(colname):
    ## No future TRT observations in input:
    use_col = ("TRT" not in colname or \
               "-" in colname or "|0" in colname) and \
              ("TRT_Rank_diff" not in colname)
    return(use_col)

def plot_hist_TRT_Ranks(df_plot,cfg_tds):
    fig_hist, axes = plt.subplots(1, 1)
    fig_hist.set_size_inches(8, 5)

    ## Analyse distribution of ranks
    axes = visTRT.plot_band_TRT_col(axes,df_plot["RANKr"],18000,4000,arrow_start=25000)
    df_plot["RANKr"] = df_plot["RANKr"]/10.
    df_plot["RANKr"].hist(ax=axes,bins=np.arange(0,4.125,0.125),facecolor=(.7,.7,.7),alpha=0.75,grid=True)
    axes.set_xlabel("TRT rank")
    axes.set_title("TRT Rank Distribution")
    df_plot["RANKr"] = df_plot["RANKr"]*10.
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"TRT_Rank_hist_post.pdf"), orientation="landscape")
    
def plot_TRT_scatter(df_plot,list_min_plus,path_addon="",contour=False,TRTcol=False,model_borders_ls=None):
    df_plot_i = df_plot
    if isinstance(df_plot, list) and len(df_plot)!=len(list_min_plus):
        raise ValueError("If list of df is provided, the list of time deltas must be of the same length")
    fig, axes = plt.subplots(nrows=1, ncols=len(list_min_plus), figsize=[4*len(list_min_plus)+1,8])
    props = dict(boxstyle='round', facecolor='white') #, alpha=0.5

    for ax_i, min_plus in enumerate(list_min_plus):
        if isinstance(df_plot, list):
            df_plot_i = df_plot[ax_i]
            
        if TRTcol:
            axes[ax_i] = visTRT.plot_band_TRT_col(axes[ax_i],df_plot_i["TRT_Rank|0"]*10,2.6,0.4,arrow_start=2.2)
            
        counts,ybins,xbins,image = axes[ax_i].hist2d(df_plot_i["TRT_Rank|0"],df_plot_i["TRT_Rank_diff|%i" % min_plus],
                                                  bins=[200,400],range=[[0,4],[-4,4]],norm=mcolors.LogNorm(),cmap="magma")
        #cbar = fig.colorbar(image, ax=axes[ax_i], extend='max')
        #cbar.set_label('Number of cells per bin of size [0.02, 0.04]', rotation=90)
        axes[ax_i].grid()
        #axes[ax_i].plot([0,4],[0,-4],'w--',linewidth=2) #,facecolor="w",linewidth=2,linestyle='--')
        if contour:
            cont2d_1, lvl = visTRT.contour_of_2dHist(counts,smooth=True)
            CS = axes[ax_i].contour(cont2d_1,levels=lvl,linewidths=2,cmap="YlGn_r",extent=[0,4,-4,4]) #,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()])
            CS_lab = axes[ax_i].clabel(CS, inline=1, fontsize=10, fmt='%i%%', colors="black")
            #[txt.set_backgroundcolor('white') for txt in CS_lab]
            [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.3, boxstyle='round', alpha=0.7)) for txt in CS_lab] #pad=0,
        axes[ax_i].set_ylabel(r"TRT Rank change  t$\mathregular{_{+%imin}}$" % min_plus) #; axes.set_title('TRT Ranks (16km diameter)')
        axes[ax_i].set_xlabel(r"TRT Rank  $\mathregular{t_0}$")
        axes[ax_i].set_title(r"Time Delta +%imin" % min_plus)
        axes[ax_i].set_aspect('equal')
        axes[ax_i].patch.set_facecolor('0.7')
        axes[ax_i].fill_between([0,4], y1=[-4,-4], y2=[0,-4], facecolor="none", hatch="X", edgecolor="darkred", linewidth=0.5)
        axes[ax_i].fill_between([0,4], y1=[ 4, 0], y2=[4, 4], facecolor="none", hatch="X", edgecolor="darkred", linewidth=0.5)
        str_n_cells = "Total number of\ncells:   %i" % np.sum(counts)
        axes[ax_i].text(0.05, 0.08, str_n_cells, transform=axes[ax_i].transAxes, #fontsize=8,
                        verticalalignment='top', bbox=props)
        #axes[ax_i].text(0.4,3.6,str_n_cells)
            
        if model_borders_ls is not None:
            model_names = ["low","med","high"]
            model_borders = [0]+model_borders_ls+[4]
            for bord_i, border in enumerate(model_borders):
                #if border not in [0,4]:
                #    axes[ax_i].axvline(border,color="white",linestyle="--",zorder=1)
                if border!=0:
                    n_samples = np.sum(np.logical_and(df_plot_i["TRT_Rank|0"].values<border,
                                                      df_plot_i["TRT_Rank|0"].values>=model_borders[bord_i-1]))
                    axes[ax_i].arrow(model_borders[bord_i-1],3.5,border-model_borders[bord_i-1],0,
                                     color="white", length_includes_head=True, head_width=0.07)
                    axes[ax_i].arrow(border,3.5,model_borders[bord_i-1]-border,0,
                                     color="white", length_includes_head=True, head_width=0.07)
                    text_loc = border-(border-model_borders[bord_i-1])/2.
                    if len(model_borders)==4:
                        text_str = r"Mod$\mathregular{_{%s}}$" % model_names[bord_i-1]
                        text_str += "\n%i" % n_samples
                    else:
                        text_str = r"Mod$_{%1.1f}$" % border
                    axes[ax_i].text(text_loc/4., 7.5/8, text_str,
                        transform=axes[ax_i].transAxes, fontsize=8,
                        verticalalignment='center', horizontalalignment='center',
                        bbox=props)
                        
    plt.tight_layout()
    path_addon_num = "_".join([str(num) for num in list_min_plus])
    if len(path_addon)>0: path_addon = "_"+path_addon
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"TRT_diff_scatter_%s%s.pdf" % (path_addon_num,path_addon)), orientation="landscape")

## ============================================================================
## Get config info:
cfg_tds = cfg.get_config_info_tds()
col10 = '#E69F00'
col30 = '#D55E00'

## Open pandas training dataframe:
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_df = pth.file_path_reader("pandas training dataframe",user_argv_path)

import_nonnan = False
if os.path.exists("%s_nonnan.h5" % os.path.splitext(path_to_df)[0]):
    import_ans = ""
    while (import_ans!="y" and import_ans!="n"):
        import_ans = raw_input("  Dataframe ending '.. _nonnan.h5' already exists, import this one? [y/n] ")
    if import_ans=="y":
        df_nonnan = pd.read_hdf("%s_nonnan.h5" % os.path.splitext(path_to_df)[0],"df_nonnan")
        import_nonnan = True
if not import_nonnan:
    df = pd.read_hdf(path_to_df,key="df")

"""
## Make analysis of how many values are missing per column:
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
del(df_pears_corr_feat)
"""

## Delete rows with nan-entries:
if not import_nonnan:
    print("Dropping NaN values")
    df_nonnan = df.dropna(0,'any')
    df_nonnan.to_hdf("%s_nonnan.h5" % os.path.splitext(path_to_df)[0],
                     key="df_nonnan",mode="w",complevel=0)

## Get map of cell locations and distributions
df_nonnan = prep.change_append_TRT_cell_info(cfg_tds,df=df_nonnan)
df_nonnan["date"] = [DTI[:12] for DTI in df_nonnan.index.values]
plot_hist_TRT_Ranks(df_nonnan,cfg_tds)
#df_nonnan["date"] = df_nonnan["date"].astype(np.datetime64,copy=False)
prep.exploit_TRT_cell_info(cfg_tds,samples_df=df_nonnan)
df_nonnan["RANKr"] = df_nonnan["RANKr"]*10

## Construct selection criteria for input dataset:
print("Split in 10min and 30min forcast")
X = df_nonnan[[Xcol for Xcol in df_nonnan.columns if get_X_col(Xcol)]]
y_10 = df_nonnan[["TRT_Rank_diff|10"]]
y_30 = df_nonnan[["TRT_Rank_diff|30"]]

## Plot histogram of Rank changes:
print("Plot histograms of TRT Rank changes")
fig = plt.figure(figsize = [10,5])
plt.title("Histogram of TRT Rank difference")
plt.hist([y_10.values,y_30.values],
         bins=50,
         color=[col10,col30],
         label=['10min Rank difference', '30min Rank difference'])
plt.legend()
plt.grid()
plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Hist_TRT_Rank_diff.pdf"), orientation="portrait")

fig = plt.figure(figsize = [10,5])
axes = fig.add_subplot(1,1,1)
sns.kdeplot(y_10.values[:,0], shade=True, kernel="gau", bw=0.03, color=col10, label='10min Rank difference')
sns.kdeplot(y_30.values[:,0], shade=True, kernel="gau", bw=0.03, color=col30, label='30min Rank difference')
plt.xlabel("TRT Rank difference")
plt.title("Kernel density estimation of TRT Rank difference")
plt.grid()
axes.get_yaxis().set_visible(False)
plt.savefig(os.path.join(cfg_tds["fig_output_path"],"KDE_TRT_Rank_diff.pdf"), orientation="portrait")

print("Print amount of cells showing higher absolute rank changes than threshold:")
for TRT_Rank_thresh in [0.2,0.5,1.0,1.5,2.0,2.5,3.0,3.5]:
    print("  Count of absolute TRT Rank differences > %.1f after 10min: %6i (%2.1f%%)" % \
          (TRT_Rank_thresh,np.sum(np.abs(y_10.values)>TRT_Rank_thresh), 100.*np.sum(np.abs(y_10.values)>TRT_Rank_thresh)/len(y_10)))
    print("  Count of absolute TRT Rank differences > %.1f after 30min: %6i (%2.1f%%)" % \
          (TRT_Rank_thresh,np.sum(np.abs(y_30.values)>TRT_Rank_thresh), 100.*np.sum(np.abs(y_30.values)>TRT_Rank_thresh)/len(y_30)))

## Plot relationship TRT Rank difference with TRT Rank at t0:
print("Plot scatterplot (TRT Rank (t0) vs. TRT Rank change)")
plot_TRT_scatter(df_nonnan,[10,30],"all")
df_nonnan_nonzerot0 = df_nonnan.loc[df_nonnan["TRT_Rank|0"]>=0.15]
plot_TRT_scatter(df_nonnan_nonzerot0,[10,30],"nonzerot0")
df_nonnan_nonzerot0t10 = df_nonnan.loc[(df_nonnan["TRT_Rank|10"]>=0.15) & (df_nonnan["TRT_Rank|0"]>=0.15)]
df_nonnan_nonzerot0t30 = df_nonnan.loc[(df_nonnan["TRT_Rank|30"]>=0.15) & (df_nonnan["TRT_Rank|0"]>=0.15)]
plot_TRT_scatter([df_nonnan_nonzerot0t10,df_nonnan_nonzerot0t30],[10,30],path_addon="nonzerot0t10t30",contour=True,TRTcol=True,model_borders_ls=[1.2,2.3])




