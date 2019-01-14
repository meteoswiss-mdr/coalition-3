""" Functions for NOSTRADAMUS_1_casestudy.py:

The following functions are assembled here for displacement case study
for the EUMETSAT Conference poster:
"""

from __future__ import division
from __future__ import print_function

import sys
import ast
import configparser
import datetime
import matplotlib.pylab as plt
import numpy as np
import pickle
import os
import pysteps as st
import pdb #pdb.set_trace()
from netCDF4 import Dataset
import warnings


sys.path.insert(0, '/data/COALITION2/database/radar/ccs4/python')
import metranet
#sys.path.insert(0, '/opt/users/jmz/monti-pytroll/packages/mpop')
#import mpop
#from mpop.satin import swisslightning_jmz

## ===============================================================================
## FUNCTIONS:

## Update cfg_set dictionary for case study:
def update_cfg_set(cfg_set,t_start,t_end,x_range,y_range,t_end_alt=None):
    """Update cfg_set dictionary with variables needed for case study.

    Parameters
    ----------
    
    cfg_set : dict
        Dictionary with the basic variables used throughout the code:
        
    t_start : datetime object
        Starting time of case study
        
    t_end : datetime object
        End time of case study
        
    t_end_alt : datetime object
        End time of displacement array (in filename).
        In case _orig and _disp arrays were created for another t0
        this values specifies the actual t0 (to find the datasets)
        
    x_range : touple
        Touple with x range values for window of interest
        
    y_range : touple
        Touple with y range values for window of interest
    """
    
    cfg_set["t_start"]     = t_start
    cfg_set["t_end"]       = t_end
    cfg_set["t_end_alt"]   = t_end_alt
    cfg_set["x_range"]     = np.arange(x_range[0],x_range[1]+1)
    cfg_set["y_range"]     = np.arange(y_range[0],y_range[1]+1)
    
    return(cfg_set)  

## Get time indices
def get_time_indices(cfg_set):
    """Get time indices for the subset of _orig and _disp datasets.
    
    cfg_set : dict
        Dictionary with the basic variables used throughout the code
    """
    
    ## Calculate range of indices between start and end time for case study
    t_diff = cfg_set["t_end"] - cfg_set["t_start"]
    t_diff_ind = (t_diff.seconds/60)/cfg_set["timestep"]
    
    ## In case displacement was done for other time point (later in time),
    ## indicated with alt_end_date
    if cfg_set["t_end_alt"] is None:
        t_delta0 = datetime.timedelta(minutes=0)
    else:
        t_delta0 = cfg_set["t_end_alt"] - cfg_set["t_end"]
    
    ## Calculate the respective additional indices not to be considered
    t_delta0_ind = (t_delta0.seconds/60)/cfg_set["timestep"]
    if t_delta0_ind!=0: print("       Ignore first %s temporal indices." % str(t_delta0_ind))
    
    ## Get range of indices
    t_ind = np.int8(np.arange(t_diff_ind+1)+t_delta0_ind)
    
    return(t_ind)
    
    
## Get data spatio-temporal subset from _disp arrays
def subset_disp_data(cfg_set, resid=False):
    """Get data from displaced (_disp) variables
    within timeframe and spatial frame.
    
    cfg_set : dict
        Dictionary with the basic variables used throughout the code
        
    resid : bool
        Use displaced variables corrected for residual movement.
        Default: False.
    """
    
    ## Date in filename:
    resid_str = "_resid" if resid else ""
    date_file = cfg_set["t_end"] if cfg_set["t_end_alt"] is None else cfg_set["t_end_alt"]
    filename_print = "%stmp/%s_<var>_<orig/disp>%s%s.npy" % (cfg_set["root_path"], date_file.strftime("%Y%m%d%H%M"),
                                                             resid_str, cfg_set["file_ext_verif"]) 
    
    print("Get subset of original and displaced data from files:\n %s ..." % filename_print)  
    ## Loop through variables to be read in:
    for var in cfg_set["var_list"]:
        print("   ... for variable %s" % var)
        filename_basic = "%stmp/%s_" % (cfg_set["root_path"], date_file.strftime("%Y%m%d%H%M"))
                                                                       
        #vararr  = np.load(filename_basic+var+"_orig"+cfg_set["file_ext_verif"]+".npy")
        disparr = np.load(filename_basic+var+"_disp"+resid_str+cfg_set["file_ext_verif"]+".npy")
        
        ## Get indices:
        time_indices = get_time_indices(cfg_set)
        x_indices = cfg_set["x_range"]
        y_indices = cfg_set["y_range"]
        
        ## Get subset based on indices:
        disparr_sub = disparr[time_indices[0]:time_indices[-1],
                              y_indices[0]:y_indices[-1],
                              x_indices[0]:x_indices[-1]]
        
        #title = "%s %s" % (var, time_indices[-1].strftime("%Y-%m-%d %H:%M"))
        timestamps = np.arange(cfg_set["t_start"], cfg_set["t_end"], datetime.timedelta(minutes=cfg_set["timestep"])).astype(datetime.datetime)
        st.plt.animate(disparr_sub[::-1,:,:],1,timestamps=timestamps,title_var=var,colorbar=False) #title=var)
        
        ## Save subset:
        filename = "%s%s_disp_sub_%s_%s%s%s.npy" % (filename_basic,var,cfg_set["t_start"].strftime("%H%M"),
                                                  cfg_set["t_end"].strftime("%H%M"),resid_str,cfg_set["file_ext_verif"])
        #filename = filename_basic+var+"_disp_sub"+cfg_set["file_ext_verif"]+".npy"
        #np.save(filename,disparr_sub)
        print("       subset of displaced %s saved." % var)
    sys.exit()
  
    
## Read in subset and calculate variable-specific statistics
def subset_stats(cfg_set):
    """Read in data subsets and calculate time-dependent and
    variable specific statistics.
    
    cfg_set : dict
        Dictionary with the basic variables used throughout the code
    """
    date_file = cfg_set["t_end"] if cfg_set["t_end_alt"] is None else cfg_set["t_end_alt"]
    print("Get subset data and calculate variable-specific, time-dependent statistics ...")
    plot_var_list = ["IR_108","Glac","CD","UD","RZC","LZC","EZC","BZC","MZC","THX"]
    
    ## Prepare big plot (as for TRT cells)
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c'])
    
    fig, axes = plt.subplots(3,4, figsize=(32, 10), dpi=600) #figsize=(28.731,8.978), (1.5*32, 1.5*10), dpi=600)

    ## Read in subsets:
    axes_arr = axes.reshape(-1)
    axes_arr_plot = np.delete(axes_arr,[7,11],0)
    
    for ax_i, var in zip(axes_arr_plot, plot_var_list):
        print("   ... for variable %s" % var)
        filename_basic = "%stmp/%s_" % (cfg_set["root_path"], date_file.strftime("%Y%m%d%H%M"))
        ## Set colour cycle:
        #fig.gca().set_color_cycle(['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c'])
                                     
        if var in ["RZC","LZC","BZC","MZC","THX","IR_108"]: #!="EZC":                                     
            filename = "%s%s_disp_sub_%s_%s%s.npy" % (filename_basic,var,cfg_set["t_start"].strftime("%H%M"),
                                                      cfg_set["t_end"].strftime("%H%M"),cfg_set["file_ext_verif"])
            disparr_sub = np.load(filename)
        
            ## Make plot of statistics:
            ax_subp = read_stats(disparr_sub,var,cfg_set,ax_i)
        elif var == "EZC":                                     
            filename15 = "%s%s15_disp_sub_%s_%s%s.npy" % (filename_basic,var,cfg_set["t_start"].strftime("%H%M"),
                                                          cfg_set["t_end"].strftime("%H%M"),cfg_set["file_ext_verif"])
            filename45 = "%s%s45_disp_sub_%s_%s%s.npy" % (filename_basic,var,cfg_set["t_start"].strftime("%H%M"),
                                                          cfg_set["t_end"].strftime("%H%M"),cfg_set["file_ext_verif"])
            disparr_sub15 = np.load(filename15)
            disparr_sub45 = np.load(filename45)
            disparr_sub_EZC = np.stack([disparr_sub15,disparr_sub45],axis=0)
        
            ## Make plot of statistics:
            #ax_i.set_prop_cycle(cycler(["#d62728","#ff7f0e","#1f77b4","#9467bd"]))
            ax_subp = read_stats(disparr_sub_EZC,var,cfg_set,ax_i)
            
        elif var == "Glac":
            filenameIR120 = "%sIR_120_disp_sub_%s_%s%s.npy" % (filename_basic,cfg_set["t_start"].strftime("%H%M"),
                                                          cfg_set["t_end"].strftime("%H%M"),cfg_set["file_ext_verif"])
            filenameIR108 = "%sIR_108_disp_sub_%s_%s%s.npy" % (filename_basic,cfg_set["t_start"].strftime("%H%M"),
                                                          cfg_set["t_end"].strftime("%H%M"),cfg_set["file_ext_verif"])
            disparr_subIR120 = np.load(filenameIR120)
            disparr_subIR108 = np.load(filenameIR108)
            disparr_glac = disparr_subIR120-disparr_subIR108
            
            ## Make plot of statistics:
            ax_subp = read_stats(disparr_glac,var,cfg_set,ax_i)
            
        elif var == "CD":
            filenameWV062 = "%sWV_062_disp_sub_%s_%s%s.npy" % (filename_basic,cfg_set["t_start"].strftime("%H%M"),
                                                          cfg_set["t_end"].strftime("%H%M"),cfg_set["file_ext_verif"])
            filenameIR108 = "%sIR_108_disp_sub_%s_%s%s.npy" % (filename_basic,cfg_set["t_start"].strftime("%H%M"),
                                                          cfg_set["t_end"].strftime("%H%M"),cfg_set["file_ext_verif"])
            disparr_subWV062 = np.load(filenameWV062)
            disparr_subIR108 = np.load(filenameIR108)
            disparr_CD = disparr_subWV062-disparr_subIR108
            
            ## Make plot of statistics:
            ax_subp = read_stats(disparr_CD,var,cfg_set,ax_i)
            
        elif var == "UD":
            filenameIR108 = "%sIR_108_disp_sub_%s_%s%s.npy" % (filename_basic,cfg_set["t_start"].strftime("%H%M"),
                                                          cfg_set["t_end"].strftime("%H%M"),cfg_set["file_ext_verif"])
            disparr_subIR108 = np.load(filenameIR108)
            disparr_UD = np.zeros(disparr_subIR108.shape)*np.nan
            disparr_UD[0:-2,:,:] = disparr_subIR108[0:-2,:,:]-disparr_subIR108[1:-1,:,:]
            
            ## Make plot of statistics:
            ax_subp = read_stats(disparr_UD,var,cfg_set,ax_i)
        
    plt.tight_layout()
    axes[-2, -1].axis('off')
    axes[-1, -1].axis('off')
    fig.patch.set_visible(False)
    
    ## Save figure:
    filename_basic = "%scasestudy/stats_plot/%s_" % (cfg_set["root_path"], date_file.strftime("%Y%m%d%H%M"))
    filename = "%sstat_sub_%s_%s%s_wide.png" % (filename_basic,cfg_set["t_start"].strftime("%H%M"),
                                              cfg_set["t_end"].strftime("%H%M"),cfg_set["file_ext_verif"])
    print("    Saved plot to %s" % filename)
    plt.savefig(filename, transparent=True)
    
## Plot variable-specific statistics
def plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,perc_val_pix=None,ax_i=None):
    """Plot statistics of specific variable.
    
    stats : numpy array
        Array of time-dependent statistics
    legend_entries : list
        List of legend entries (same number as 2nds axis of stats)
    var : str
        Variable whose statistics should be plotted
    cfg_set : dict
        Dictionary with the basic variables used throughout the code
    ax_i : axis
        Figure axis from bigger plot (default: None -> Create single plot)
    """
    import matplotlib.dates as md
    
    ## Get time-steps for x-axis
    timestamps = np.arange(cfg_set["t_start"], cfg_set["t_end"],
                           datetime.timedelta(minutes=cfg_set["timestep"])).astype(datetime.datetime)
    ## Set time format and the interval of ticks (every 15 minutes)
    xformatter = md.DateFormatter('%H:%M')
    xlocator = md.MinuteLocator(byminute=[0,15,30,45], interval = 1)
    
    
    ## Make plot
    if ax_i is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plt.grid(True)
        lineobjects = plt.plot(timestamps,stats,marker='o',linestyle='solid')
        plt.title("Variable: %s (%s)" % (title_var,var))
        plt.xlabel("Time")
        plt.ylabel(ylab)
        plt.legend(lineobjects,legend_entries)
        ax.xaxis.set_major_locator(xlocator)
        
        ## Plot second y-axis with percentage of non-NaN pixels:
        if perc_val_pix is not None:
            ax_i2 = ax_i.twinx()
            line_perc = ax_i2.plot(timestamps,perc_val_pix,linestyle='dotted',color='k')
            ax_i2.set_ylim(0,100)
            ax_i2.set_ylabel("Percentage of pixels [%]")
        
        #plt.show()
        #return(ax)
    else:
        ## Plot grid:
        ax_i.patch.set_visible(False)
        ax_i.grid(True)
        ## Plot second y-axis with percentage of non-NaN pixels:
        if perc_val_pix is not None:
            ax_i2 = ax_i.twinx()
            #ls = ["--", ":"] if perc_val_pix.ndim==2 else ":"
            if perc_val_pix.ndim==1:
                line_perc = ax_i2.plot(timestamps,perc_val_pix,color='k',linestyle="--",linewidth=2.)
            else:
                ax_i2.plot(timestamps,perc_val_pix[:,0],color='k',linestyle="--",linewidth=2.) # line_perc = 
                ax_i2.plot(timestamps,perc_val_pix[:,1],color='k',linestyle=":", linewidth=2.)
            if var!="THX":
                ax_i2.set_ylim(0,50)
                ax_i2.set_ylabel("Percentage of pixels [%]")
            else:
                ax_i2.set_ylim(bottom = 0)
                ax_i2.set_ylabel("Number of Lightnings")
            if perc_val_pix.ndim==2: ax_i2.legend(["15dBZ","45dBZ"],fontsize='small',loc='upper left') #line_perc,fontsize='small'
        ## Plot lines:
        lineobjects = ax_i.plot(timestamps,stats,marker='o',linestyle='solid')
        ## Plot title and labels:
        ax_i.set_title("Variable: %s" % title_var) # (%s)" % (title_var,cfg_set["abbrev_dict"][var]))
        #ax_i.set_xlabel("Time")
        ax_i.set_ylabel(ylab)
        ## Plot legend:
        if var in ["CD","Glac","UD"]: leg_pos = "lower right"
        else: leg_pos = "upper right"
        if var in ["EZC"]: ax_i.set_ylim(0,30)
        ax_i.legend(lineobjects,legend_entries,fontsize='medium',loc=leg_pos)
        ## Adjust x-axis:
        #ax_i.set_xlim(timestamps[0] -3*datetime.timedelta(minutes=cfg_set["timestep"]),
        #              timestamps[-1]+datetime.timedelta(minutes=cfg_set["timestep"]))
        ax_i.set_xlim(timestamps[0] -  datetime.timedelta(minutes=cfg_set["timestep"]),
                      timestamps[-1]+4*datetime.timedelta(minutes=cfg_set["timestep"]))
        ax_i.xaxis.set_major_locator(xlocator)
        ax_i.xaxis.set_major_formatter(xformatter)
        for tick in ax_i.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
            tick.label.set_rotation(45)
        ## Adjust y-axis:
        if var in ["EZC","BZC","MZC"]: ax_i.set_ylim(bottom = 0)

## Calculate percentage of non-NaN pixels per timestep
def perc_nonnan(disparr_sub_rev):
    """Calculate percentage of non-NaN pixels per timestep.
    
    disparr_sub : numpy array
        Array of data subset
    """
    
    perc_val_pix = np.zeros(disparr_sub_rev.shape[0])*np.nan
    n_pix_tot = disparr_sub_rev.shape[1]*disparr_sub_rev.shape[2]
    for t_step in range(disparr_sub_rev.shape[0]):
        perc_val_pix[t_step] = np.sum(~np.isnan(disparr_sub_rev[t_step,:,:]))/n_pix_tot*100
    return(perc_val_pix)
        
        
## Calculate variable-specific statistics
def read_stats(disparr_sub,var,cfg_set,ax_i=None):
    """Calculate statistics of specific variable.
    
    disparr_sub : numpy array
        Array of data subset
    var : str
        Variable whose statistics should be plotted
    cfg_set : dict
        Dictionary with the basic variables used throughout the code
    ax_i : axis
        Figure axis from bigger plot (default: None -> Create single plot)
    """
    
    ## 
    if var=="RZC":
        ## Set to nan and reverse order (so far, newest obs comes first):
        disparr_sub_rev = disparr_sub[::-1,:,:]
        disparr_sub_rev[disparr_sub_rev<0.1] = np.nan
        
        ## Calculate statistics:
        stats = np.stack([np.nanpercentile(disparr_sub_rev, 95, axis=(1,2)), #np.nanmax(disparr_sub_rev, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 90, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 75, axis=(1,2))], axis=1)
        
        ## Read in percentage of non-NaN pixels:
        perc_val_pix = perc_nonnan(disparr_sub_rev)
        
        ## Plotting annotation:
        legend_entries = ["RR$_{95\%}$", "RR$_{90\%}$", "RR$_{75\%}$"] # "RZC$_{max}$",
        ylab = "Rain Rate [mm h$^{-1}$]"
        title_var = "Rain Rate (RR)"
        
        ## Make plot:
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,perc_val_pix,ax_i)
        
    elif var=="BZC":
        disparr_sub_rev = disparr_sub[::-1,:,:]
        disparr_sub_rev[disparr_sub_rev<0.0001] = np.nan
        stats = np.stack([np.nanmax(disparr_sub_rev, axis=(1,2)),
                          #np.nanpercentile(disparr_sub_rev, 90, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 50, axis=(1,2))], axis=1)
        perc_val_pix = perc_nonnan(disparr_sub_rev)
        legend_entries = ["POH$_{max}$", "POH$_{med}$"] # , "POH$_{75\%}$"
        ylab = "POH [%]" 
        title_var = "Probability of Hail (POH)"
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,perc_val_pix,ax_i)
    
    elif var=="LZC":
        disparr_sub_rev = disparr_sub[::-1,:,:]
        disparr_sub_rev[disparr_sub_rev<5] = np.nan
        stats = np.stack([np.nanmax(disparr_sub_rev, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 90, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 75, axis=(1,2))], axis=1)
        perc_val_pix = perc_nonnan(disparr_sub_rev)
        legend_entries = ["VIL$_{max}$", "VIL$_{90\%}$", "VIL$_{75\%}$"]
        ylab = "VIL [kg m$^{-2}$]"
        title_var = "Vertically Integrated Liquid (VIL)"
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,perc_val_pix,ax_i)
        
    elif var=="MZC":
        disparr_sub_rev = disparr_sub[::-1,:,:]
        disparr_sub_rev[disparr_sub_rev<2] = np.nan
        stats = np.stack([np.nanmax(disparr_sub_rev, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 50, axis=(1,2))], axis=1)
        perc_val_pix = perc_nonnan(disparr_sub_rev)
        legend_entries = ["MESHS$_{max}$", "MESHS$_{med}$"] # "MESHS$_{90\%}$", "MESHS$_{75\%}$"]
        ylab = "MESHS [cm]"
        title_var = "Maximum Expected Severe Hail Size (MESHS)"
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,perc_val_pix,ax_i)
        
    elif var=="EZC":
        disparr_sub_rev = disparr_sub[:,::-1,:,:]
        disparr_sub_rev[disparr_sub_rev<0.0001] = np.nan
        stats = np.stack([np.nanmax(disparr_sub_rev[0,:,:,:], axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev[0,:,:,:], 50, axis=(1,2)),
                          np.nanmax(disparr_sub_rev[1,:,:,:], axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev[1,:,:,:], 50, axis=(1,2))], axis=1)
        perc_val_pix = np.stack([perc_nonnan(disparr_sub_rev[0,:,:,:]),perc_nonnan(disparr_sub_rev[1,:,:,:])], axis=1)
        legend_entries = ["ET15$_{max}$", "ET15$_{med}$","ET45$_{max}$", "ET45$_{med}$"]
        ylab = "Altitude a.s.l. [km]"
        title_var = "Echo Top (ET)"
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,perc_val_pix,ax_i)
        
    elif var=="THX":
        disparr_sub_rev = disparr_sub[::-1,:,:]
        disparr_sub_rev[disparr_sub_rev<0.0001] = np.nan
        #stats = np.stack([np.nanmax(disparr_sub_rev, axis=(1,2)),
        #                  np.nanpercentile(disparr_sub_rev, 90, axis=(1,2)),
        #                  np.nanpercentile(disparr_sub_rev, 75, axis=(1,2))], axis=1)
        stats = np.stack([np.nanmax(disparr_sub_rev, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 90, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 75, axis=(1,2))], axis=1)
        perc_val_pix = np.nansum(disparr_sub_rev, axis=(1,2))
        legend_entries = ["THX$_{max}$", "THX$_{90\%}$", "THX$_{75\%}$"]
        ylab = "Lightning density [km$^{-2}$]"
        title_var = "Lightning Density (THX)"
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,perc_val_pix,ax_i)
        
    elif var=="IR_108":
        disparr_sub_rev = disparr_sub[::-1,:,:]
        stats = np.stack([np.nanmin(disparr_sub_rev, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 10, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 25, axis=(1,2))], axis=1)
        legend_entries = ["IR$_{min}$", "IR$_{10\%}$", "IR$_{25\%}$"]
        ylab = "IR 10.8$\mu$m [K]"
        title_var = "Brightness Temperature $T_{B}$"
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,None,ax_i)
        
    elif var=="Glac":
        disparr_sub_rev = disparr_sub[::-1,:,:]
        stats = np.stack([np.nanpercentile(disparr_sub_rev, 99, axis=(1,2)), #np.nanmax(disparr_sub_rev, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 90, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 75, axis=(1,2))], axis=1)
        legend_entries = ["Glac$_{99\%}$", "Glac$_{90\%}$", "Glac$_{75\%}$"]
        ylab = "IR 12.0$\mu$m - IR 10.8$\mu$m [K]"
        title_var = "Glaciation indicator (GI)"
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,None,ax_i)
        
    elif var=="CD":
        disparr_sub_rev = disparr_sub[::-1,:,:]
        stats = np.stack([np.nanmax(disparr_sub_rev, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 90, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 75, axis=(1,2))], axis=1)
        legend_entries = ["COD$_{max}$", "COD$_{90\%}$", "COD$_{75\%}$"]
        ylab = "WV 6.1$\mu$m - IR 10.8$\mu$m [K]"
        title_var = "Cloud optical depth indicator (COD)"
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,None,ax_i)
        
    elif var=="UD":
        disparr_sub_rev = disparr_sub[::-1,:,:]
        stats = np.stack([np.nanpercentile(disparr_sub_rev, 10, axis=(1,2)), #np.nanmax(disparr_sub_rev, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 55, axis=(1,2)),
                          np.nanpercentile(disparr_sub_rev, 90, axis=(1,2))], axis=1)
        legend_entries = ["w$_{T,10\%}$", "w$_{T,med}$", "w$_{T,90\%}$"]
        ylab = "IR 10.8$\mu$m (t$_0$) - IR 10.8$\mu$m (t$_{-5}$) [K]"
        title_var = "Updraft strength indicator ($w_{T}$)"
        ax_subp = plot_stats(stats,legend_entries,ylab,title_var,var,cfg_set,None,ax_i)
    else: return None
    
    return ax_subp
        
## Read in subset and calculate variable-specific statistics
def subset_stats_NoIdeaWhyICreatedThisFunction(cfg_set):
    """Read in data subsets and calculate time-dependent and
    variable specific statistics.
    
    cfg_set : dict
        Dictionary with the basic variables used throughout the code
    """
    date_file = cfg_set["t_end"] if cfg_set["t_end_alt"] is None else cfg_set["t_end_alt"]
    
    print("Get subset data and calculate variable-specific, time-dependent statistics ...")
    plot_var_list = ["IR_108","Glac","CD","UD","RZC","LZC","EZC","BZC","MZC","THX"]
    
    ## Prepare big plot (as for TRT cells)
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c'])
    
    fig, axes = plt.subplots(3,4, figsize=(32, 10))

    ## Read in subsets:
    axes_arr = axes.reshape(-1)
    axes_arr_plot = np.delete(axes_arr,[7,11],0)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
