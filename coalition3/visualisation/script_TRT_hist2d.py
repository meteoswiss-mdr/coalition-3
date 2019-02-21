# coding: utf-8
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import os

import coalition3.inout.paths as pth
import coalition3.inout.readxr as rxr
import coalition3.operational.statistics as stat
from coalition3.visualisation.TRTcells import contour_of_2dHist


path_to_xarray = pth.file_path_reader("xarray training dataset")
xr_new_TRT = rxr.xarray_file_loader(path_to_xarray)
#xr_new_TRT = xr.open_mfdataset("Combined_stat_pixcount.nc")

Rank_TRT = xr_new_TRT["RANKr"]/10.
Rank_TRT_rand = Rank_TRT+np.random.uniform(-0.1,0.1,len(Rank_TRT))

Rank_COAL3_new    = xr_new_TRT["TRT_Rank"]
Rank_COAL3_allmed = stat.calc_TRT_Rank(xr_new_TRT,ET_option="all_median")["TRT_Rank"]
Rank_COAL3_allmax = stat.calc_TRT_Rank(xr_new_TRT,ET_option="all_max")["TRT_Rank"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[15,4.2])
axes[0].set_ylabel('TRT Rank (COAL3 - Cond. Median ET45)')
axes[1].set_ylabel('TRT Rank (COAL3 - Median ET45)')
axes[2].set_ylabel('TRT Rank (COAL3 - Max ET45)')
hist2d_1 = axes[0].hist2d(Rank_TRT_rand.values,Rank_COAL3_new.sel(time_delta=0).values.flatten(),bins=150,range=[[0,4],[0,4]],norm=mcolors.LogNorm(),cmap="magma")
#fig.colorbar(hist2d_1[3], ax=axes[0], extend='max')
hist2d_2 = axes[1].hist2d(Rank_TRT_rand.values,Rank_COAL3_allmed.sel(time_delta=0).values.flatten(),bins=150,range=[[0,4],[0,4]],norm=mcolors.LogNorm(),cmap="magma")
#fig.colorbar(hist2d_2[3], ax=axes[1], extend='max')
hist2d_3 = axes[2].hist2d(Rank_TRT_rand.values,Rank_COAL3_allmax.sel(time_delta=0).values.flatten(),bins=150,range=[[0,4],[0,4]],norm=mcolors.LogNorm(),cmap="magma")
#fig.colorbar(hist2d_3[3], ax=axes[2], extend='max')

for ax_i in axes:
    ax_i.grid()
    ax_i.plot([0,4],[0,4],"r")
    ax_i.set_xlabel('TRT Rank (TRT)')
    ax_i.set_title(' ')
    ax_i.set_aspect('equal')
    ax_i.patch.set_facecolor('0.7')
axes[1].set_title('TRT Ranks (16km diameter)')

fig.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[10,8])
axes.set_ylabel('TRT Rank (COAL3 - Cond. Median ET45)')
counts,ybins,xbins,image = axes.hist2d(Rank_TRT_rand.values,Rank_COAL3_new.sel(time_delta=0).values.flatten(),
                                       bins=200,range=[[0,4],[0,4]],norm=mcolors.LogNorm(),cmap="magma")
cbar = fig.colorbar(image, ax=axes, extend='max')
cbar.set_label('Number of cells per bin of size [0.02, 0.02]', rotation=90)
cont2d_1, lvl = contour_of_2dHist(counts,smooth=True)
axes.grid()
axes.plot([0,4],[0,4],'w--',linewidth=2) #,facecolor="w",linewidth=2,linestyle='--')
CS = axes.contour(cont2d_1,levels=lvl,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=2,cmap="YlGn_r")
CS_lab = axes.clabel(CS, inline=1, fontsize=10, fmt='%i%%', colors="black")
#[txt.set_backgroundcolor('white') for txt in CS_lab]
[txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.3, boxstyle='round', alpha=0.7)) for txt in CS_lab] #pad=0,
axes.set_xlabel('TRT Rank (TRT)'); axes.set_title('TRT Ranks (16km diameter)')
axes.set_aspect('equal'); axes.patch.set_facecolor('0.7')
str_n_cells = "Total number of cells = %i" % np.sum(counts)
axes.text(0.4,3.6,str_n_cells)
plt.show()
