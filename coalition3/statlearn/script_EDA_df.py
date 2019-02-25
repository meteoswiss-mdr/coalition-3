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

import coalition3.inout.paths as pth
import coalition3.inout.readconfig as cfg
from coalition3.visualisation.TRTcells import contour_of_2dHist

def get_X_col(colname):
    ## No future TRT observations in input:
    use_col = ("TRT" not in colname or \
               "-" in colname or "|0" in colname) and \
              ("TRT_Rank_diff" not in colname)
    return(use_col)
    
def plot_TRT_scatter(df_plot,list_min_plus,path_addon="",contour=False):
    df_plot_i = df_plot
    if isinstance(df_plot, list) and len(df_plot)!=len(list_min_plus):
        raise ValueError("If list of df is provided, the list of time deltas must be of the same length")
    fig, axes = plt.subplots(nrows=1, ncols=len(list_min_plus), figsize=[4*len(list_min_plus)+1,8])
    for ax_i, min_plus in enumerate(list_min_plus):
        if isinstance(df_plot, list):
            df_plot_i = df_plot[ax_i]
        counts,ybins,xbins,image = axes[ax_i].hist2d(df_plot_i["TRT_Rank|0"],df_plot_i["TRT_Rank_diff|%i" % min_plus],
                                                  bins=[200,400],range=[[0,4],[-4,4]],norm=mcolors.LogNorm(),cmap="magma")
        #cbar = fig.colorbar(image, ax=axes[ax_i], extend='max')
        #cbar.set_label('Number of cells per bin of size [0.02, 0.04]', rotation=90)
        axes[ax_i].grid()
        #axes[ax_i].plot([0,4],[0,-4],'w--',linewidth=2) #,facecolor="w",linewidth=2,linestyle='--')
        if contour:
            cont2d_1, lvl = contour_of_2dHist(counts,smooth=True)
            CS = axes[ax_i].contour(cont2d_1,levels=lvl,linewidths=2,cmap="YlGn_r",extent=[0,4,-4,4]) #,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()])
            CS_lab = axes[ax_i].clabel(CS, inline=1, fontsize=10, fmt='%i%%', colors="black")
            #[txt.set_backgroundcolor('white') for txt in CS_lab]
            [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.3, boxstyle='round', alpha=0.7)) for txt in CS_lab] #pad=0,
        axes[ax_i].set_ylabel(r"TRT Rank change  $t_{+%imin}$" % min_plus) #; axes.set_title('TRT Ranks (16km diameter)')
        axes[ax_i].set_xlabel(r"TRT Rank  $t_0$")
        axes[ax_i].set_title(r"Time Delta +%imin" % min_plus)
        axes[ax_i].set_aspect('equal')
        axes[ax_i].patch.set_facecolor('0.7')
        axes[ax_i].fill_between([0,4], y1=[-4,-4], y2=[0,-4], facecolor="none", hatch="X", edgecolor="darkred", linewidth=0.5)
        axes[ax_i].fill_between([0,4], y1=[ 4, 0], y2=[4, 4], facecolor="none", hatch="X", edgecolor="darkred", linewidth=0.5)
        str_n_cells = "Total number of\ncells:   %i" % np.sum(counts)
        props = dict(boxstyle='round', facecolor='white') #, alpha=0.5
        axes[ax_i].text(0.05, 0.08, str_n_cells, transform=axes[ax_i].transAxes, #fontsize=8,
                        verticalalignment='top', bbox=props)
        #axes[ax_i].text(0.4,3.6,str_n_cells)

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
print("Dropping NaN values")
df_nonnan = df.dropna(0,'any')
df_nonnan.to_hdf(os.path.join(os.path.dirname(path_to_df),"df_23km_nonnan.h5"),
                 key="df_nonnan",mode="w",complevel=0)
#df_nonnan = pd.read_hdf("df_23km_nonnan.h5","df_nonnan")

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
plot_TRT_scatter([df_nonnan_nonzerot0t10,df_nonnan_nonzerot0t30],[10,30],path_addon="nonzerot0t10t30",contour=True)




