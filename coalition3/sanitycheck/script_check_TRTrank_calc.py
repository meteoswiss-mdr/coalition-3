""" [COALITION3] This code is only written to see whether it makes a
    difference whether for the calculation of the TRT ranks, based on
    the CCS4 radar fields, the median is taken of all pixels, or only
    of those with max echo >45dBZ. This lead to the additional radar
    variables with the '_nonmin' ending, e.g. EZC45_stat_nonmin, where
    the statistics are only calculated on the pixels with non-minimum
    values (which can differ, e.g. for RZC it's 0.05mm/h)."""

from __future__ import division
from __future__ import print_function

import sys
import pdb
import glob
import numpy as np
import xarray as xr
import datetime as datetime
import matplotlib.pylab as plt
from shapely.geometry import Point
from shapely.geometry import Polygon

import coalition3.inout.readccs4 as rccs
import coalition3.inout.readconfig as cfg
import coalition3.inout.paths as pth
import coalition3.visualisation.TRTcells as TRTvis


def calc_TRT_rank(lon_coord,lat_coord,ET45,CZC,LZC,RZC=None,plot_cells=False):
    """Calculate the TRT Rank based on the coordinates of the TRT cell shape and the fields ET45, CZC, LZC, RZC"""

    ## Convert lat/lon coordinates in Swiss coordinates (list of tuples):
    x_CH, y_CH = TRTvis.lonlat2xy(lon_coord,lat_coord)
    coord_tuples = [(x_CH_i,y_CH_i) for x_CH_i,y_CH_i in zip(x_CH,y_CH)]
    
    ## Get polygon:
    poly = Polygon(coord_tuples)
    
    ## Get boolean array with all pixels within polygon:
    bool_arr = np.reshape([poly.contains(Point(x_coord,y_coord)) for x_coord, y_coord in zip(ccs4_CH[0].flatten(),ccs4_CH[1].flatten())],(640,710))
    bool_arr = bool_arr[::-1,:]
    
    ## Plot cell over RZC field:
    if plot_cells:
        if RZC is None: raise ValueError("RZC value must be provided if plots should be created")
        plt.clf()
        plt.imshow(RZC[0,:,:],interpolation="None")
        plt.contour(bool_arr)
        plt.pause(0.5)

    ## Calc statistics of VIL, MaxEcho and Area of pix with >= 57dBZ
    VIL_scal  = np.max(LZC[:,bool_arr])
    ME_scal   = np.max(CZC[:,bool_arr])
    A57_scal  = np.sum(CZC[:,bool_arr]>=57.)
        
    ## Set min/max values of the variable:
    if VIL_scal>65.: VIL_scal = 65.   ## Max  VIL:       56 kg m-2
    if ME_scal<45.:  ME_scal  = 45.   ## Min MaxEcho:    45 dBZ
    if ME_scal>57.:  ME_scal  = 57.   ## Max MaxEcho:    57 dBZ
    if A57_scal>40.: A57_scal = 40.   ## Max pix >57dBZ: 40
        
    ## Scale variables to values between 0 and 4:    
    VIL_scal  = VIL_scal/65.*4
    ME_scal   = (ME_scal-45.)/12.*4
    A57_scal  = A57_scal/40.*4


    ## Get median of ET45 out of all pixels with MaxEcho >= 45dBZ
    ET45_vals = ET45[:,bool_arr]
    if len(ET45_vals[ET45_vals>0])<1: ET45_scal = 0
    else:
        ET45_scal = np.median(ET45_vals[ET45_vals>0])
        if ET45_scal>10.: ET45_scal = 10. ## Max EchoTop:    10 km
        ET45_scal = ET45_scal/10.*4
    
    ## Calculate TRT rank (with median of ET45 of selected pixels):
    TRT_Rank_sel = (2.*VIL_scal+2*ET45_scal+ME_scal+2.*A57_scal)/7.

    ## Get median of ET45 out of all pixels (including zero pixels)
    ET45_scal = np.median(ET45_vals)
    if ET45_scal>10.: ET45_scal = 10. ## Max EchoTop:    10 km
    ET45_scal = ET45_scal/10.*4
    
    ## Calculate TRT rank (with median of ET45 of all pixels):
    TRT_Rank_all = (2.*VIL_scal+2*ET45_scal+ME_scal+2.*A57_scal)/7.

    return TRT_Rank_sel, TRT_Rank_all

## Get time point from user:
user_time_point = sys.argv[1] if len(sys.argv)==2 else None
if user_time_point is not None:
    try: 
        user_time_point = datetime.datetime.strptime(user_time_point,"%Y%m%d%H%M")
    except ValueError: user_time_point = None
while user_time_point is None:
    try:
        user_time_point = datetime.datetime.strptime(raw_input("Please enter a time point [%Y%m%d%H%M]: "),
                                                     "%Y%m%d%H%M")
    except ValueError:
        print("  Date is not in the right format, please repeat")
        user_time_point = None
print("\nGet TRT values at time point %s" % user_time_point)

## Initialise empty lists for results:
TRT_rank_ls = []
COAL3_rank_median_sel_ls = []
COAL3_rank_median_all_ls = []

## Get config info
cfg_set_tds   = cfg.get_config_info_tds()
cfg_set_input, cfg_var, cfg_var_combi = cfg.get_config_info_op()

## Initialise fields (CCS4 meshgrid and VIL, EchoTop and MaxEcho observations):
ccs4_CH = np.meshgrid(np.arange(255000,965000,1000)+500,np.arange(-160000,480000,1000)+500)
ET45 = rccs.get_vararr_t(user_time_point, "EZC45", cfg_set_input)
CZC  = rccs.get_vararr_t(user_time_point, "CZC",  cfg_set_input)
LZC  = rccs.get_vararr_t(user_time_point, "LZC",  cfg_set_input)
RZC  = rccs.get_vararr_t(user_time_point, "RZC",  cfg_set_input)

## Get TRT file:
filename = pth.path_creator(user_time_point,"TRT","TRT",cfg_set_input)[0]
if len(filename) == 0:  raise IOError("No TRT file found")
elif len(filename) > 1: raise IOError("More than one TRT file found")
file = open(filename[0],"r")

## Go through TRT file:
for line in file:
    line2=line.strip()
    if len(line2) > 0:
        if line2.startswith("@") or line2.startswith("#"): pass
        else:
            data = line2.split(";")
            print_str = "  Working on TRT traj_ID: %s" % data[0]
            print('\r',print_str,end='')
            shape_coord = [float(coord) for coord in data[27:][:-1]]
            lon_coord   = shape_coord[::2]
            lat_coord   = shape_coord[1::2]
            TRT_Rank_sel, TRT_Rank_all = calc_TRT_rank(lon_coord,lat_coord,ET45,CZC,LZC,RZC)
            COAL3_rank_median_sel_ls.append(TRT_Rank_sel)
            COAL3_rank_median_all_ls.append(TRT_Rank_all)
            TRT_rank_ls.append(float(data[11])/10.)
            
## Make Scatter-Plot:
TRT_rank_ls = np.array(TRT_rank_ls)
COAL3_rank_median_all_ls = np.array(COAL3_rank_median_all_ls)
COAL3_rank_median_sel_ls = np.array(COAL3_rank_median_sel_ls)
is_finite_all = np.isfinite(COAL3_rank_median_all_ls)
is_finite_sel = np.isfinite(COAL3_rank_median_sel_ls)
corr_all = np.corrcoef(TRT_rank_ls[is_finite_all],COAL3_rank_median_all_ls[is_finite_all])[0,1].round(3)
corr_sel = np.corrcoef(TRT_rank_ls[is_finite_sel],COAL3_rank_median_sel_ls[is_finite_sel])[0,1].round(3)

label_all = 'All pixels (%s)' % corr_all 
label_sel = 'Only non-null pixels (%s)' % corr_sel

print("\nCorrelation with COALITION3-Rank calculated with median of ALL pixels:      %s" % corr_all)
print("Correlation with COALITION3-Rank calculated with median of NON-ZERO pixels: %s" % corr_sel)
title = "TRT Rank Comparison %s" % user_time_point
max_val = np.min([4,np.nanmax(np.concatenate([TRT_rank_ls,COAL3_rank_median_all_ls,COAL3_rank_median_sel_ls]))*1.1])
plt.xlim([0,max_val]); plt.ylim([0,max_val]); plt.plot([0,max_val],[0,max_val],'--k')
plt.plot(TRT_rank_ls,COAL3_rank_median_all_ls, 'x', label=label_all)
plt.plot(TRT_rank_ls,COAL3_rank_median_sel_ls, 'x', label=label_sel)
plt.legend(loc='upper left')
plt.xlabel("TRT Rank - TRT")
plt.ylabel("TRT Rank - COALITION3")
plt.title(title)
plt.grid()
plt.show()
