""" [COALITION3] Function to calculate skill scores regarding the displacement error.."""

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pysteps as st
#from pysteps.vf import scores_det_cat_fcst

from coalition3.inout.paths import path_creator_vararr, path_creator_UV_disparr, path_creator
from coalition3.inout.iotmp import save_file, load_file

## =============================================================================
## FUNCTIONS:
    
# Calculate statistics for current variable:
def calc_statistics(cfg_set,var,vararr_disp):
    """Calculate skill scores for current displaced variable var.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable for which skill score is calculated
        
    vararr : numpy.array
        Array with the current observation
    """
    
    ## Import numpy array with preceding statistics results:
    filename_verif_stat = "%stmp/%s_%s_stat_verif.npy" % (cfg_set["root_path"],
                           cfg_set["verif_param"],
                           str(cfg_set[cfg_set["verif_param"]]))
    stat_array = np.load(filename_verif_stat)
    
    ## Create temporary statistics array:
    stat_array_temp = np.zeros((1,len(cfg_set["var_list"]),int(cfg_set["n_stat"])))*np.nan
    
    ## Get threshold for pixels with values
    value_threshold = 0. #cfg_set["R_threshold"] if var=="RZC" else 0.0
    
    ## Assign statistics to respective column:
    var_list = np.array(cfg_set["var_list"])
    ind_var = np.where(var_list==var)[0][0]
    
    ## Find pixels which are not NaN or zero
    bool_nonnan_nonzero = np.logical_and(np.isfinite(vararr_disp[0,:,:].flatten()),
                                      vararr_disp[0,:,:].flatten()>value_threshold)
    
    stat_array_temp[0,ind_var,0] = np.nanmin(vararr_disp[0,:,:])
    stat_array_temp[0,ind_var,1] = np.nanmax(vararr_disp[0,:,:])
    stat_array_temp[0,ind_var,2] = np.nanmean(vararr_disp[0,:,:].flatten()[np.where(bool_nonnan_nonzero)])
    stat_array_temp[0,ind_var,3] = np.nanpercentile(vararr_disp[0,:,:].flatten()[np.where(bool_nonnan_nonzero)],90)
    stat_array_temp[0,ind_var,4] = np.sum(bool_nonnan_nonzero)
          
    ## Save stats results again:
    if ind_var!=0:
        stat_array[-1,ind_var,:] = stat_array_temp[0,ind_var,:]
        print("      "+var+" statistics are saved")
    else:
        if stat_array.shape[0]==1 and np.all(stat_array < -9998): #np.all(np.isnan(stat_array)):
            stat_array = stat_array_temp
            print("      New stats array is created (with %s stats in first column)" % var)
        else:
            stat_array = np.concatenate((stat_array,stat_array_temp), axis=0)
            #if np.all(np.isnan(stat_array)): print("Still all nan"); print(stat_array)

    np.save(filename_verif_stat,stat_array)
    
    
# Calculate skill scores for current variable:
def calc_skill_scores(cfg_set,var,vararr_disp):
    """Calculate skill scores for current displaced variable var.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable for which skill score is calculated
        
    vararr_disp : numpy.array
        Array where the current observation is at the level 0 of the first dimension
    """
    
    ## Import numpy array with preceding verification results:
    filename_verif = "%stmp/%s_%s_%s_verif.npy" % (cfg_set["root_path"],
                      cfg_set["verif_param"], str(cfg_set[cfg_set["verif_param"]]),var)
    verif_array = np.load(filename_verif)
    ## Copy new layer (if its not the first verification, then just overwrite the NaNs)
    #if (np.isnan(verif_array)).all:
    #    verif_array_temp = verif_array
    #else:
    verif_array_temp = np.zeros((1,len(cfg_set["scores_list"]),
                                cfg_set["n_integ"]-1))*np.nan
    
    ## Define threshold for categorical forecast:
    if cfg_set["R_thresh_meth"] == "fix":
        R_tresh = cfg_set["R_threshold"]
    elif cfg_set["R_thresh_meth"] == "perc":
        R_tresh = 0.1
    #    R_tresh = np.min([np.nanpercentile(oflow_source_data[0,:,:],cfg_set["R_threshold"]),
    #                      np.nanpercentile(oflow_source_data[1,:,:],cfg_set["R_threshold"])])
    threshold = R_tresh if var=="RZC" else 0.0

    ## Loop through integration steps:
    #t1 = datetime.datetime.now()
    for t_step in range(1,cfg_set["n_integ"]):
        verif_array_temp[0,:,t_step-1] = st.vf.scores_det_cat_fcst(vararr_disp[t_step,:,:],
                                                             vararr_disp[0,:,:],
                                                             threshold,
                                                             cfg_set["scores_list"])
                                                                 
    #t2 = datetime.datetime.now()
    #test_arr = np.apply_over_axes(st.vf.scores_det_cat_fcst,[1,2],vararr_disp[1:,:,:],
    #                               obs=vararr_disp[0,:,:],thr=threshold,scores=cfg_set["scores_list"])
    #print("         Elapsed time for for-loop calculating scores: "+str(t2-t1))
    
    #if np.array_equal(test_arr[1:,:,:],verif_array_temp[:,:,:]):
    #    print("np.array_equal successfull")
    #else: print("np.array_equal NOT successfull"); print(test_arr); print(verif_array_temp[-1,:,:])
    #print(verif_array_temp)
    
    ## Save verification results again:
    if verif_array.shape[0]==1 and np.all(verif_array < -9998): #np.all(np.isnan(verif_array)):
        verif_array = verif_array_temp
        print("      New "+var+" skill score array is created")
    else:
        verif_array = np.concatenate((verif_array,verif_array_temp), axis=0)
    #if np.all(np.isnan(verif_array)): print("Still all nan"); print(verif_array)
    
    np.save(filename_verif,verif_array)
    print("      "+var+" skill score array is saved")
    
    # Do procedure only in case some interesting variable is actually detected? But then, what else could be skipped?
    # Have array where time information is collected, when displacement was done, and skill scores were calculated.

## Analyse skill scores:
def analyse_skillscores(cfg_set,var):
    """Analyse skill scores.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable whose skill scores should be analysed
    """
    
    ## Read in skill scores of the respective variable:
    filename_verif = "%stmp/%s_%s_%s_verif.npy" % (cfg_set["root_path"],
                      cfg_set["verif_param"], str(cfg_set[cfg_set["verif_param"]]),var)
    verif_array = np.load(filename_verif)
    
    ## Read in statistics:
    filename_verif_stat = "%stmp/%s_%s_stat_verif.npy" % (cfg_set["root_path"],
                           cfg_set["verif_param"],
                           str(cfg_set[cfg_set["verif_param"]]))
    stat_array = np.load(filename_verif_stat)
    
    ## verif_array & stat_array:
    ## Dim 1 -> Zeit
    
    ## verif_array:
    ## Dim 2 -> Skill score (csi,hk,sedi)
    ## Dim 3 -> Lead time (5 - 45min)
    
    ## stat_array:
    ## Dim 2 -> Variable (RZC,BZC,LZC,MZC,EZC,THX)
    ## Dim 3 -> Statistic:
    ## - Min
    ## - Max
    ## - Mean
    ## - 90% quantile
    ## - Number of Pixels
    
    ## Make plot of decrease in skill with increasing lead-time (HK score):
    plt.clf()
    fig, ax = plt.subplots()
    plt.grid(True)
    bp = plt.boxplot(verif_array[:,1,:],notch=True,patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgrey')
   
    plt.title("HK score as function of Lead time\nVariable: %s" % var)
    plt.xlabel("Lead time [min]")
    plt.ylabel("Hanssen-Kuipers Discriminant")
    ax.set_xticklabels(np.arange(5,45,5))
    #plt.show()
    #plt.figure(figsize=(2,2))
    filename = "%splot_verif/HKvsLeadT_%s_%s_%s.pdf" % (cfg_set["output_path"],cfg_set["verif_param"],
                                                          str(cfg_set[cfg_set["verif_param"]]),var)
    plt.savefig(filename)
    
    ## Make plot of skill at lead time = 5min as function of 
    ## average rain-rate (used for optical flow):
    plt.clf()
    fig, ax = plt.subplots()
    plt.plot(stat_array[:,0,2], verif_array[:,1,0], 'bo')
    plt.title("HK score (lead time = 5min) as function of mean rain rate\nVariable: %s" % var)
    plt.xlabel("Rain rate [mm/h]")
    plt.ylabel("Hanssen-Kuipers Discriminant")
    plt.grid(True)
    #plt.show()
    #plt.figure(figsize=(2,2))
    filename = "%splot_verif/HKvsRR_%s_%s_%s.pdf" % (cfg_set["output_path"],cfg_set["verif_param"],
                                                          str(cfg_set[cfg_set["verif_param"]]),var)
    plt.savefig(filename)
    
## Compare skill scores:
def compare_skillscores_help(cfg_set,var,verif_param_ls):
    """Analyse skill scores.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable whose skill scores should be analysed
        
    verif_param_ls : list
        List with additional verification parameter.
    """
    
    ## Read in statistics:
    filename_verif_stat = "%stmp/%s_%s_stat_verif.npy" % (cfg_set["root_path"],
                           cfg_set["verif_param"],
                           str(cfg_set[cfg_set["verif_param"]]))
    stat_array = np.load(filename_verif_stat)
    
    ## Read in skill scores of the respective variable:
    verif_array_ls = []
    time_dim_ls = []
    for verif_param in verif_param_ls:
        #filename = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/threshold_verif/180527_12_20/%s_%s_%s_verif.npy" % (cfg_set["verif_param"],verif_param, var)
        filename = "%stmp/%s_%s_%s_verif.npy" % (cfg_set["root_path"],cfg_set["verif_param"],verif_param,var)
        verif_array_ls.append(np.load(filename))
        time_dim_ls.append(verif_array_ls[-1].shape[0])
        
    ## Check for time dimension (if not equal, set to minimum length):
    min_time_dim = np.nanmin(time_dim_ls)
    if any(x.shape[0]!=min_time_dim for x in verif_array_ls):
        print("   *** Warning: Verification arrays are not of the same length! ***")
        for i in range(len(verif_array_ls)):
            verif_array_ls[i] = verif_array_ls[i][:min_time_dim,:,:]
    
    ## Concatenate to one big file (along new, first dimension):
    verif_array_con = np.stack(verif_array_ls)
    
    """
    filename_verif_01 = "%stmp/%s_0.1_%s_verif.npy" % (cfg_set["root_path"],
                         cfg_set["verif_param"], var)
    filename_verif_05 = "%stmp/%s_0.5_%s_verif.npy" % (cfg_set["root_path"],
                         cfg_set["verif_param"], var)
    filename_verif_10 = "%stmp/%s_1.0_%s_verif.npy" % (cfg_set["root_path"],
                         cfg_set["verif_param"], var)
    verif_array_01 = np.load(filename_verif_01)
    verif_array_05 = np.load(filename_verif_05)
    verif_array_10 = np.load(filename_verif_10)
    
    ## Concatenate to minimum time covered by all verification datasets:
    min_time_dim = np.nanmin([verif_array_01.shape[0],verif_array_05.shape[0],verif_array_10.shape[0]])
    verif_array_con = np.stack([verif_array_01[:min_time_dim,:,:],
                                verif_array_05[:min_time_dim,:,:],
                                verif_array_10[:min_time_dim,:,:]])
    """
    
    ## Plot skill score comparison:
    compare_skillscores(cfg_set,var,verif_array_con,stat_array,verif_param_ls)
    
## Compare skill scores:
def compare_skillscores(cfg_set,var,verif_array_con,stat_array,verif_param_ls): #,verif_param_ls):
    """Analyse skill scores.

    Parameters
    ----------

    cfg_set : list
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    var : str
        Variable whose skill scores should be analysed
        
    verif_param_ls : list
        List with additional verification parameter.
    """

    from cycler import cycler
    reduced_ss_set = False #True
    if reduced_ss_set:
        verif_array_con = verif_array_con[:,:,[0,2],:]
        list_ss_names = ["Critical Success\nIndex CSI","Symmetric Extremal\nDependence Index SEDI"]
    
    legend_title = "<Insert title here>"
    if cfg_set["verif_param"]=="R_threshold":
        legend_unit = "mm/h" if verif_param_ls[i_verif_samp] < 50. else "%"
        legend_title = "%s Threshold =" % cfg_set["abbrev_dict"][cfg_set["oflow_source"]] #"RZC Threshold ="
    elif cfg_set["verif_param"]=="resid_method":
        legend_unit = ""
        legend_title = "Method to reduce\nresidual movement:"
    else: 
        legend_unit = "<Insert Unit>"
        legend_title = "<Insert title>"
    
    ## Define some basic variables:
    n_verif_samp  = verif_array_con.shape[0]
    n_skillscores = verif_array_con.shape[2]
    n_leadtimes   = verif_array_con.shape[3]
    
    x_val = np.arange(cfg_set["timestep"],
                      cfg_set["timestep"]*(verif_array_con.shape[3]+1),
                      cfg_set["timestep"])
        
    ## Initialise plot:
    plt.clf()
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'm', 'k'])))
    fig, axs = plt.subplots(n_skillscores, sharex=True, figsize=(8,7))
    #axs[-1].set_xticklabels(np.arange(5,45,5))
    
    plot_array = np.zeros([verif_array_con.shape[0],verif_array_con.shape[0]])
       
    for i_skillscore in np.arange(n_skillscores):
        ## Initialise array for median and IQR (3rd dim -> Median and IQR)
        plot_array = np.zeros([n_verif_samp,n_leadtimes,3])*np.nan
        
        ## Calculate median and IQR:
        for i_verif_samp in np.arange(n_verif_samp):
            for i_leadtimes in np.arange(n_leadtimes):
                plot_array[i_verif_samp,i_leadtimes,0] = np.nanmedian(verif_array_con[i_verif_samp,
                                                                                      :,
                                                                                      i_skillscore,
                                                                                      i_leadtimes])
                                                                                   
                plot_array[i_verif_samp,i_leadtimes,1] = np.nanpercentile(verif_array_con[i_verif_samp,
                                                                                          :,
                                                                                          i_skillscore,
                                                                                          i_leadtimes], 
                                                                          25)
        
                plot_array[i_verif_samp,i_leadtimes,2] = np.nanpercentile(verif_array_con[i_verif_samp,
                                                                                          :,
                                                                                          i_skillscore,
                                                                                          i_leadtimes], 
                                                                          75)
                
        col_list = ["#17becf","#1f77b4","#bcbd22","#ff7f0e","#d62728"]
        
        ## Plot IQR:
        for i_verif_samp in np.arange(n_verif_samp):
            axs[i_skillscore].fill_between(x_val,
                                           plot_array[i_verif_samp,:,1],
                                           plot_array[i_verif_samp,:,2],
                                           alpha=0.1, facecolor=col_list[i_verif_samp]) # facecolor='green',
        
        ## Plot Median:
        ylab = cfg_set["abbrev_dict"][cfg_set["scores_list"][i_skillscore]] if not reduced_ss_set else list_ss_names[i_skillscore]
        
        for i_verif_samp in np.arange(n_verif_samp):
            legend_entry = "%s %s" % (verif_param_ls[i_verif_samp], legend_unit)
            axs[i_skillscore].plot(x_val, plot_array[i_verif_samp,:,0],marker='o',linestyle='solid', label=legend_entry, color=col_list[i_verif_samp]) # facecolor='green',
            axs[i_skillscore].set(ylabel=ylab)
            axs[i_skillscore].grid(True)
            axs[i_skillscore].set_xlim((x_val[0]-1,x_val[-1]+1))
            axs[i_skillscore].set_ylim((0,1))
            
        ## Insert title and legend:
        if i_skillscore==0:
            axs[i_skillscore].set(title="Skill scores for variable %s" % cfg_set["abbrev_dict"][var]) #var)            
            axs[i_skillscore].legend(loc='upper right',title=legend_title,fontsize="small",
                                     ncol=3) #loc='lower center', bbox_to_anchor=(0, -0.5))
        
        ## Insert y-label:
        if i_skillscore==np.arange(n_skillscores)[-1]:
            axs[i_skillscore].set(xlabel="Lead time [min]")
            
    plt.tight_layout()
    
    ## Save file:
    filename = "%splot_verif/SkillScores_%s_%s.pdf" % (cfg_set["output_path"],cfg_set["verif_param"],var)
    plt.savefig(filename)
    #fig.delaxes(axs[1])
    #plt.draw()
    #plt.show()
    
    #plt.clf()
    #fig2, axs_2 = plt.subplots(2, sharex=True, figsize=(7,8))
    #axs_2[0] = axs[0]
    #axs_2[1] = axs[2]
    #fig2.show()
           
        