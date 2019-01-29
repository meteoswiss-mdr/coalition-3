""" [COALITION3] Functions to derive the flow field and to displace variables
    or TRT cell centres along (against) it."""

from __future__ import division
from __future__ import print_function

import os
import datetime
import numpy as np
import pysteps as st

from joblib import Parallel, delayed
import multiprocessing

import coalition3.inout.paths as pth
import coalition3.inout.iotmp as iotmp
import coalition3.verification.skillscores as sksc

## =============================================================================
## FUNCTIONS:
    
## Check whether precalculated displacement array for training is already existent, otherwise create it.
def check_create_precalc_disparray(cfg_set):
    """Check whether precalculated displacement array for training is already existent, otherwise create it.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py         
    """
    ## Change values in cfg_set to fit the requirements for the precalculation
    ## 288 integration steps for 24h coverage
    cfg_set_precalc = cfg_set.copy()
    cfg_set_precalc["n_integ"] = int(24*60/cfg_set["timestep"])
    if not cfg_set["future_disp_reverse"]:
        cfg_set_precalc["t0"] = datetime.datetime(cfg_set["t0"].year,cfg_set["t0"].month,cfg_set["t0"].day,0,0)+datetime.timedelta(days=1)
    else:
        cfg_set_precalc["t0"] = datetime.datetime(cfg_set["t0"].year,cfg_set["t0"].month,cfg_set["t0"].day,0,0)
        
    ## Check path for precalculated displacement array, and its dimensions:
    path_name = pth.path_creator_UV_disparr("standard",cfg_set,
                                        path=cfg_set["UV_precalc_path"],
                                        t0=cfg_set["t0"].strftime("%Y%m%d"))

    ## If not existent, create new one, else check dimensions,
    ## if correct, update, else, create new one.
    if os.path.isfile(path_name):
        #disparr = np.load(path_name)
        disparr = iotmp.load_file(path_name,"Vx")
        if disparr.ndim!=3 or disparr.shape[0]!=(cfg_set_precalc["n_integ"]):
            print("   Dimensions of existent precalculated displacement array do not match current settings.\n   Ndim: "+
                  str(disparr.ndim)+"\n   Shape: "+str(disparr.shape[0])+" instead of "+str(cfg_set_precalc["n_integ"])+"\n   A new one is created")
            create_new_disparray(cfg_set_precalc,precalc=True)
        else:
            print("Found precalculated disparray file")
        #if cfg_set["precalc_UV_disparr"] and cfg_set["generating_train_ds"]:
        #    get_the_disp_array for specific t0
    else:
        print("Found no precalculated disparray file, create such.")
        create_new_disparray(cfg_set_precalc,precalc=True)
                
## Get array of datetime objects of integration steps:
def datetime_integ_steps(cfg_set):
    """Get datetime objects of integration steps"""
    t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=cfg_set["timestep"])
    t_integ = cfg_set["t0"] - cfg_set["time_change_factor"]*t_delta
    return t_integ

## Check whether Displacement array is already existent, otherwise create it.
## If existent, replace with newest displacement:
def check_create_disparray(cfg_set):
    """Check whether Displacement array is already existent, otherwise create it.
    If existent, replace with newest displacement.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py         
    """
    
    ## Check path for displacement array, and its dimensions:
    path_name = pth.path_creator_UV_disparr("standard",cfg_set)                                     

    ## In case more than one day is within integration period, check whether all precalculated files are available:
    all_precalc_files_existent = True
    t_integ_ls = datetime_integ_steps(cfg_set)
    if len(np.unique([t_integ.day for t_integ in t_integ_ls]))>1:
        for t_integ in t_integ_ls:
            path_name_precalc = pth.path_creator_UV_disparr("standard",cfg_set,path=cfg_set["UV_precalc_path"],
                                                        t0=t_integ.strftime("%Y%m%d"))
            if not os.path.isfile(path_name_precalc): all_precalc_files_existent = False
        
    ## If not existent, create new one, else check dimensions,
    ## if correct, update, else, create new one.
    if os.path.isfile(path_name):
        #disparr = np.load(path_name)
        disparr = iotmp.load_file(path_name,"Vx")
        ## Check that existing dimensions match the required dimensions:
        if disparr.ndim!=3 or disparr.shape[0]!=(cfg_set["n_integ"]):
            print("   Dimensions of existent displacement array do not match current settings.\n   Ndim: "+
                  str(disparr.ndim)+"\n   Shape: "+str(disparr.shape[0])+"\n   A new one is created")
            if cfg_set["precalc_UV_disparr"] and all_precalc_files_existent:
                read_disparray_from_precalc(cfg_set)
            else:
                create_new_disparray(cfg_set)
    else:
        if cfg_set["precalc_UV_disparr"] and all_precalc_files_existent:
            read_disparray_from_precalc(cfg_set)
        else:
            create_new_disparray(cfg_set)

## Read UV and Displacement arrays from precalculated array:
def read_disparray_from_precalc(cfg_set):
    """Read UV and Displacement arrays from precalculated array. This function
    is only executed, if the respective precalculated UV- and displacement array
    is available, and t0 is not too close to midnight.      
    """

    t_integ_ls = datetime_integ_steps(cfg_set)
    ## Case that all integration time steps are within the same day:
    if len(np.unique([t_integ.day for t_integ in t_integ_ls]))<=1:    
        path_name = pth.path_creator_UV_disparr("standard",cfg_set,path=cfg_set["UV_precalc_path"],
                                            t0=cfg_set["t0"].strftime("%Y%m%d"))
        #path_name = "%s/%s_%s_disparr_UV%s.%s" % (cfg_set["UV_precalc_path"], cfg_set["t0"].strftime("%Y%m%d"), #(cfg_set["t0"]-datetime.timedelta(days=1)).strftime("%Y%m%d"),
        #                                          cfg_set["oflow_source"],cfg_set["file_ext_verif"],
        #                                          cfg_set["save_type"])
        print("   Read from precalculated UV-disparray...\n      %s" % path_name)

        ## Get indices to timesteps of interest:
        if not cfg_set["future_disp_reverse"]:
            t2_ind = int((24*60-cfg_set["t0"].hour*60-cfg_set["t0"].minute+cfg_set["n_integ"]*cfg_set["timestep"])/cfg_set["timestep"])
            t1_ind = int((24*60-cfg_set["t0"].hour*60-cfg_set["t0"].minute)/cfg_set["timestep"])
        else:
            t2_ind = int((cfg_set["t0"].hour*60+cfg_set["t0"].minute+cfg_set["n_integ"]*cfg_set["timestep"])/cfg_set["timestep"])
            t1_ind = int((cfg_set["t0"].hour*60+cfg_set["t0"].minute)/cfg_set["timestep"])
        t_ind = range(t1_ind,t2_ind)
    
        ## Get subset of precalculated UV displacement array:
        UVdisparr = iotmp.load_file(path_name)
        Vx = UVdisparr["Vx"][t_ind,:,:]; Vy = UVdisparr["Vy"][t_ind,:,:]
        Dx = UVdisparr["Dx"][t_ind,:,:]; Dy = UVdisparr["Dy"][t_ind,:,:]
    ## Case that integration time steps span over two days (around 00:00):
    else:
        Dx = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
        Dy = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
        Vx = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
        Vy = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
        path_name_file_t0 = pth.path_creator_UV_disparr("standard",cfg_set,path=cfg_set["UV_precalc_path"],
                                                    t0=t_integ_ls[0].strftime("%Y%m%d"))
        UVdisparr = iotmp.load_file(path_name_file_t0)
        print("   Read from precalculated UV-disparray...\n      %s" % path_name_file_t0)
        t_integ_0_day = t_integ_ls[0].day
        DV_index = 0        

        ## Go through all time steps to read in respective UV disparr field
        for t_integ in t_integ_ls:
            ## In case of non-reverse case, subtract five minutes (as 00:05 is first (!) element in precalc array)
            if not cfg_set["future_disp_reverse"]:
                t_integ_corr = t_integ-datetime.timedelta(minutes=cfg_set["timestep"])
                #t_integ_corr = t_integ-cfg_set["time_change_factor"]*datetime.timedelta(minutes=cfg_set["timestep"])
            else:
                t_integ_corr = t_integ
            
            ## If day changes, read in new daily array
            if t_integ_corr.day!=t_integ_0_day:
                path_name_file_tinteg = pth.path_creator_UV_disparr("standard",cfg_set,path=cfg_set["UV_precalc_path"],
                                                                t0=t_integ_corr.strftime("%Y%m%d"))
                UVdisparr = iotmp.load_file(path_name_file_tinteg)
                print("   Read from precalculated UV-disparray...\n      %s" % path_name_file_tinteg)
                t_integ_0_day = t_integ_corr.day
                
            ## Calculate index to be read:
            if not cfg_set["future_disp_reverse"]:
                UVdisparr_index = int((24*60-t_integ_corr.hour*60-t_integ_corr.minute)/cfg_set["timestep"])-1
            else:
                UVdisparr_index = int((t_integ_corr.hour*60+t_integ_corr.minute)/cfg_set["timestep"])
            
            ## Read in the data:
            Vx[DV_index,:,:] = UVdisparr["Vx"][UVdisparr_index,:,:]; Vy[DV_index,:,:] = UVdisparr["Vy"][UVdisparr_index,:,:]
            Dx[DV_index,:,:] = UVdisparr["Dx"][UVdisparr_index,:,:]; Dy[DV_index,:,:] = UVdisparr["Dy"][UVdisparr_index,:,:] 
            DV_index += 1            
    
    filename = pth.path_creator_UV_disparr("standard",cfg_set)
    #filename = "%stmp/%s_%s_disparr_UV%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
    #                                            cfg_set["oflow_source"],cfg_set["file_ext_verif"],
    #                                            cfg_set["save_type"])
    print("   ... saving UV-disparray subset in\n      %s" % filename)
    iotmp.save_file(filename, data_arr=[Dx,Dy,Vx,Vy],var_name=["Dx","Dy","Vx","Vy"],
              cfg_set=cfg_set)

## Read input to calculate optical flow displacement array for specific time-step
def calc_disparr(t_current, cfg_set, resid=False):
    """Get 2-dim displacement array for flow between timesteps t_current and t_current - n_past_frames*timestep.
   
    Parameters
    ----------
    
    t_current : datetime object
        Current time for which to calculate displacement array.
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
        
    UV_inter : bool
        If UV_inter is true, the calculated UV and sparsened UV vectors are returned as well.
    
    See function check_create_disparray(t0, timestep, n_integ, root_path, t0_str, oflow_source, oflow_source_path)
    """
    
    if resid:
        ## Read in current displaced oflow_source file:
        filename = pth.path_creator_vararr("disp",cfg_set["oflow_source"],cfg_set)
        #filename = "%stmp/%s_%s_disp%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                       cfg_set["oflow_source"], cfg_set["file_ext_verif"], cfg_set["save_type"])
        t_diff = cfg_set["t0"] - t_current
        t_diff_ind = int((t_diff.seconds/60)/cfg_set["timestep"])
        #oflow_source_data = np.load(filename)[t_diff_ind:t_diff_ind+cfg_set["n_past_frames"]+1,:,:]
        #oflow_source_data = np.load(filename)[t_diff_ind+cfg_set["n_past_frames_resid"]::-1,:,:][:cfg_set["n_past_frames"]+1]
        oflow_source_data = iotmp.load_file(filename,cfg_set["oflow_source"])[t_diff_ind+cfg_set["n_past_frames_resid"]::-1,:,:][:cfg_set["n_past_frames"]+1]
        if oflow_source_data.shape[0]==1:
            UV = R = np.zeros((2,oflow_source_data.shape[1],oflow_source_data.shape[2]))
            if not cfg_set["UV_inter"]: return UV, R
            else: return UV, R, np.zeros(4)*np.nan, np.zeros(4)*np.nan
        if np.all(np.array_equal(oflow_source_data[0,:,:],oflow_source_data[1,:,:])):
            raise ValueError("Input data equal")
    else:
        ## Read in current oflow_source file:
        filenames, timestamps = pth.path_creator(t_current, cfg_set["oflow_source"], cfg_set["source_dict"][cfg_set["oflow_source"]], cfg_set)
        ret = metranet.read_file(filenames[0], physic_value=True)
        oflow_source_data = np.atleast_3d(ret.data)
        for filename in filenames[1:]:
            ret_d_t = metranet.read_file(filename, physic_value=True)
            oflow_source_data_d_t = np.atleast_3d(ret_d_t.data)
            oflow_source_data = np.append(oflow_source_data,oflow_source_data_d_t, axis=2)
        
        oflow_source_data = np.moveaxis(oflow_source_data,2,0)
        #oflow_source_data_masked = np.ma.masked_invalid(oflow_source_data)
        #oflow_source_data_masked = np.ma.masked_where(oflow_source_data_masked==0,oflow_source_data_masked)    
        

    ## Check whether there are non-nan entries:
    if np.any(np.isnan(oflow_source_data).all(axis=(1,2))):
        print("   *** Warning: Input oflow source field is all NAN!\n                Returning NAN fields.***")
        nan_arr = oflow_source_data[0,:,:]*np.nan
        D  = np.array([nan_arr,nan_arr])
        UV = np.array([nan_arr,nan_arr])
        UV_vec  = []; UV_vec_sp = []
        if not cfg_set["UV_inter"]:
            return D, UV
        else: return D, UV, UV_vec, UV_vec_sp
        
    ## Convert linear rainrates to logarithimc dBR units
    if not cfg_set["oflow_source"]=="RZC":
        raise NotImplementedError("So far displacement array retrieval only implemented for RZC")
    else:
        ## Get threshold method:
        if not resid:
            R_thresh_meth = cfg_set["R_thresh_meth"]
            R_threshold = cfg_set["R_threshold"]
        else:
            R_thresh_meth = cfg_set["R_thresh_meth_resid"]
            R_threshold = cfg_set["R_threshold_resid"]
        
        ## Get threshold value:
        if R_thresh_meth == "fix":
            R_thresh = R_threshold
        elif R_thresh_meth == "perc":
            R_thresh = np.min([np.nanpercentile(oflow_source_data[0,:,:],R_threshold),
                              np.nanpercentile(oflow_source_data[1,:,:],R_threshold)])
        else: raise ValueError("R_thresh_meth must either be set to 'fix' or 'perc'")
                
        ## Convert to dBR
        dBR, dBRmin = st.utils.mmhr2dBR(oflow_source_data, R_thresh)
        dBR[~np.isfinite(dBR)] = dBRmin
        #R_thresh = cfg_set["R_threshold"]
        
        ## In case threshold is not exceeded, lower R_threshold by 20%
        while (dBR==dBRmin).all():
            if cfg_set["verbose"]: print("   *** Warning: Threshold not exceeded, "+
                                         "lower R_threshold by 20% to "+str(R_thresh*0.8)+" ***")
            R_thresh = R_thresh*0.8
            dBR, dBRmin = st.utils.mmhr2dBR(oflow_source_data, R_thresh)
            dBR[~np.isfinite(dBR)] = dBRmin
        
        ## For correction of residuals original mm/h values are used:
        if resid:
            oflow_source_data_min = oflow_source_data
            oflow_source_data_min[oflow_source_data_min<=R_thresh] = R_thresh
            oflow_source_data_min[~np.isfinite(oflow_source_data_min)] = R_thresh
  
    ## Calculate UV field
    oflow_method = st.optflow.get_method(cfg_set["oflow_method_name"])
    if not resid:
        UV, UV_vec, UV_vec_sp = oflow_method(dBR,return_single_vec=True,return_declust_vec=True)
    else:
        UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,winsize_LK5=(120,20),quality_level_ST=0.05,
                                             max_speed=20,nr_IQR_outlier=5,k=30,
                                             decl_grid=cfg_set["decl_grid_resid"],function=cfg_set["inter_fun_resid"],
                                             epsilon=cfg_set["epsilon_resid"],#factor_median=.2,
                                             return_single_vec=True,return_declust_vec=True,
                                             zero_interpol=cfg_set["zero_interpol"])
        #UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,block_size_ST=15,winsize_LK5=(120,20),quality_level_ST=0.05,
        #                                                  max_speed=20,nr_IQR_outlier=5,decl_grid=20,function="inverse",k=20,factor_median=0.05,
        #                                                  return_single_vec=True,return_declust_vec=True,zero_interpol=True)
        #UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,block_size_ST=15,winsize_LK5=(120,20),quality_level_ST=0.05,
        #                                                  max_speed=20,nr_IQR_outlier=5,decl_grid=20,function="nearest",k=20,factor_median=.2,
        #                                                  return_single_vec=True,return_declust_vec=True)
       
    ## In case no motion vectors were detected, lower R_threshold by 20%
    if np.any(~np.isfinite(UV)):
        dBR_orig = dBR
    n_rep = 0
    while np.any(~np.isfinite(UV)):
        if cfg_set["verbose"]:
            print("   *** Warning: No motion vectors detected, lower R_threshold by 30% to "+str(R_thresh*0.7)+" ***")
        R_thresh = R_thresh*0.7
        dBR, dBRmin = st.utils.mmhr2dBR(oflow_source_data, R_thresh)
        dBR[~np.isfinite(dBR)] = dBRmin
        
        if resid:
            oflow_source_data_min = oflow_source_data
            oflow_source_data[oflow_source_data<=R_thresh] = R_thresh
            oflow_source_data[~np.isfinite(oflow_source_data)] = R_thresh
        if not resid:
            UV, UV_vec, UV_vec_sp = oflow_method(dBR,return_single_vec=True,return_declust_vec=True)
        else:
            UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,winsize_LK5=(120,20),quality_level_ST=0.05,
                                                 max_speed=20,nr_IQR_outlier=5,k=30,
                                                 decl_grid=cfg_set["decl_grid_resid"],function=cfg_set["inter_fun_resid"],
                                                 epsilon=cfg_set["epsilon_resid"],#factor_median=.2,
                                                 return_single_vec=True,return_declust_vec=True,
                                                 zero_interpol=cfg_set["zero_interpol"])
            #UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,block_size_ST=15,winsize_LK5=(120,20),quality_level_ST=0.05,
            #                                                  max_speed=20,nr_IQR_outlier=5,decl_grid=20,function="inverse",k=20,epsilon=10,#factor_median=0.05,
            #                                                  return_single_vec=True,return_declust_vec=True,zero_interpol=True)
            #UV, UV_vec, UV_vec_sp = oflow_method(oflow_source_data_min,min_distance_ST=2,block_size_ST=15,winsize_LK5=(120,20),quality_level_ST=0.05,
            #                                                  max_speed=20,nr_IQR_outlier=5,decl_grid=20,function="nearest",k=20,factor_median=.2,
            #                                                  return_single_vec=True,return_declust_vec=True)
        n_rep += 1
        if n_rep > 2:
            UV = np.zeros((2,dBR.shape[1],dBR.shape[2]))
            if cfg_set["verbose"]: print("   *** Warning: Return zero UV-array! ***")
            break
    
    ## Invert direction of intermediate motion vectors
    #if cfg_set["UV_inter"]:
    #    UV_vec[2:3,:,:] = -UV_vec[2:3,:,:]
    #    UV_vec_sp[2:3,:,:] = -UV_vec_sp[2:3,:,:]
    
    """
    ## Advect disp_test to get the advected test_array and the displacement array
    adv_method = st.advection.get_method(cfg_set["adv_method"])
    dBR_adv, D = adv_method(dBR[-1,:,:], UV, 1, return_displacement=True) 
    
    ## convert the forecasted dBR to mmhr
    if cfg_set["oflow_source"]=="RZC":
        if cfg_set["R_thresh_meth"] == "fix":
            R_tresh = cfg_set["R_threshold"]
        elif cfg_set["R_thresh_meth"] == "perc":
            R_tresh = np.min([np.nanpercentile(oflow_source_data[0,:,:],cfg_set["R_threshold"]),
                              np.nanpercentile(oflow_source_data[1,:,:],cfg_set["R_threshold"])])
        else: raise NotImplementedError("R_thresh_meth must either be set to 'fix' or 'perc'")
        oflow_source_data_forecast = st.utils.dBR2mmhr(dBR_adv, R_tresh)
    
    ## Print results:
    if False:
        calc_disparr_ctrl_plot(D,timestamps,oflow_source_data,oflow_source_data_forecast,cfg_set)
    """
    #plt.imshow(D[0,:,:])
    #plt.show()
    #fig, axes = plt.subplots(nrows=1, ncols=2)
    #fig1=axes[0].imshow(UV[0,:,:])
    #fig.colorbar(fig1,ax=axes[0])#,orientation='horizontal')
    #fig2=axes[1].imshow(UV[1,:,:])
    #fig.colorbar(fig2,ax=axes[1])#,orientation='horizontal')
    #fig.tight_layout()
    #plt.show()
    #sys.exit()
    

    if np.all(UV==0): #np.any(~np.isfinite(UV)):
        if cfg_set["instant_resid_corr"] and not resid:
            print("   *** Warning: No residual movement correction performed ***")
        D = UV.copy()
    else:
        adv_method = st.advection.get_method(cfg_set["adv_method"])
        dBR_disp, D = adv_method(dBR[-2,:,:],UV,1,return_displacement=True,return_XYW=False)
        
        if cfg_set["instant_resid_corr"] and not resid:    
            if cfg_set["verbose"]:
                print("   Make instantaneous residual movement correction")
            ## Advect second last observation to t0:
            dBR_disp[~np.isfinite(dBR_disp)] = dBRmin
            
            ## Convert dBR values of t0 and second last time step to mm/h:
            RZC_resid_fields = np.stack([st.utils.dBR2mmhr(dBR_disp[0,:,:], R_thresh),
                                         st.utils.dBR2mmhr(dBR[-1,:,:], R_thresh)])
            #plt.imshow(RZC_resid_fields[0,:,:]); plt.title("RZC_resid_fields[0,:,:]"); plt.show()
            #plt.imshow(RZC_resid_fields[1,:,:]); plt.title("RZC_resid_fields[1,:,:]"); plt.show()

            ## Get residual displacement field
            UV_resid = oflow_method(RZC_resid_fields,min_distance_ST=2,
                                    winsize_LK5=(120,20),quality_level_ST=0.05,
                                    max_speed=20,nr_IQR_outlier=5,k=30,
                                    decl_grid=cfg_set["decl_grid_resid"],function=cfg_set["inter_fun_resid"],
                                    epsilon=cfg_set["epsilon_resid"],#factor_median=.2,
                                    zero_interpol=cfg_set["zero_interpol"])
            
            ## Add UV_resid to original UV array
            n_rep = 0
            while np.any(~np.isfinite(UV_resid)):
                print("       No UV_resid field found")
                R_thresh *= 0.7
                RZC_resid_fields = np.stack([st.utils.dBR2mmhr(dBR_disp[0,:,:], R_thresh),
                                             st.utils.dBR2mmhr(dBR[-1,:,:], R_thresh)])
                
                UV_resid = oflow_method(RZC_resid_fields,min_distance_ST=2,
                                        winsize_LK5=(120,20),quality_level_ST=0.05,
                                        max_speed=20,nr_IQR_outlier=5,k=30,
                                        decl_grid=cfg_set["decl_grid_resid"],function=cfg_set["inter_fun_resid"],
                                        epsilon=cfg_set["epsilon_resid"],#factor_median=.2,
                                        zero_interpol=cfg_set["zero_interpol"])
                n_rep += 1
                if n_rep > 2:
                    UV_resid = np.zeros((2,dBR.shape[1],dBR.shape[2]))
                    #if cfg_set["verbose"]: 
                    print("   *** Warning: Return zero UV_resid array! ***")
                    break
            UV += UV_resid
            
            ## Displace with UV_resid field to get D_resid and add to D array:
            dBR_disp_disp, D = adv_method(RZC_resid_fields[0,:,:],UV,1,
                                          return_displacement=True,return_XYW=False)  
            #D += D_resid
    
    if not cfg_set["UV_inter"]:
        return D, UV
    else: return D, UV, UV_vec, UV_vec_sp
   
## In case displacement array does not exist yet, it is created with this function:
def create_new_disparray(cfg_set,extra_verbose=False,resid=False,precalc=False):
    """Create Displacement array.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
        
    precalc : bool
        Create precalculated displacement array?
        Default: False.
    """
    
    ## Calculate n_integ time deltas:
    resid_print = " for residual movement correction" if resid else ""
    print("Create new displacement array%s..." % resid_print)
    Dx = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
    Dy = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
    Vx = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
    Vy = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])
    if cfg_set["UV_inter"]: UV_vec = []; UV_vec_sp = []

    ## Calculate flow fields and displacement fields for each integration timestep n_integ:
    i = 0   
    t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=cfg_set["timestep"])
    for t_d in t_delta:
        t_current = cfg_set["t0"] - cfg_set["time_change_factor"]*t_d
        if len(t_delta) > 10 and t_current.minute == 0: print("  ... for %s" % t_current.strftime("%Y-%m-%d %H:%M"))
        if cfg_set["UV_inter"]:
            D, UV, UV_vec_temp, UV_vec_sp_temp = calc_disparr(t_current, cfg_set, resid)
            UV_vec.append(UV_vec_temp) #if UV_vec_temp is not None: 
            UV_vec_sp.append(UV_vec_sp_temp) #if UV_vec_sp_temp is not None: UV_vec_sp.append(UV_vec_sp_temp)
        else:
            D, UV = calc_disparr(t_current, cfg_set, resid)
            
        Vx[i,:,:] = UV[0,:,:]; Vy[i,:,:] = UV[1,:,:]
        Dx[i,:,:] =  D[0,:,:]; Dy[i,:,:] =  D[1,:,:]
        if extra_verbose: print("   Displacement calculated between "+str(t_current-datetime.timedelta(minutes=cfg_set["timestep"]))+
                          " and "+str(t_current))
        i += 1
        if False:
            figname = "%s%s_%s_disparr_UV.png" % (cfg_set["output_path"], t_current.strftime("%Y%m%d%H%M"),
                                                  cfg_set["oflow_source"])
            plt.subplot(1, 2, 1)
            st.plt.quiver(D,None,25)
            plt.title('Displacement field\n%s' % t_current.strftime("%Y-%m-%d %H:%M"))
            plt.subplot(1, 2, 2,)
            st.plt.quiver(UV,None,25)
            plt.title('UV field\n%s' % t_current.strftime("%Y-%m-%d %H:%M"))
            plt.savefig(figname)

    ## Give different filename if the motion field is precalculated for training dataset:
    type = "resid" if resid else "standard"
    if precalc:
        if not cfg_set["future_disp_reverse"]:
            t0_str = (cfg_set["t0"]-datetime.timedelta(days=1)).strftime("%Y%m%d")
        else: 
            t0_str = cfg_set["t0"].strftime("%Y%m%d")
        filename = pth.path_creator_UV_disparr(type,cfg_set,
                                           path=cfg_set["UV_precalc_path"],
                                           t0=t0_str)
        #filename1 = "%s/%s_%s_disparr_UV%s%s.%s" % (cfg_set["UV_precalc_path"], (cfg_set["t0"]-datetime.timedelta(days=1)).strftime("%Y%m%d"),
        #                                           cfg_set["oflow_source"],append_str_resid,cfg_set["file_ext_verif"],
        #                                           cfg_set["save_type"])
    else:
        filename = pth.path_creator_UV_disparr(type,cfg_set)
        #filename = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                              cfg_set["oflow_source"],append_str_resid,cfg_set["file_ext_verif"],
        #                                              cfg_set["save_type"])
                                                  
    #np.savez(filename, Dx=Dx, Dy=Dy, Vx=Vx, Vy=Vy)
    iotmp.save_file(filename, data_arr=[Dx,Dy,Vx,Vy],var_name=["Dx","Dy","Vx","Vy"],
              cfg_set=cfg_set)
    print("  ... new displacement array saved in:\n       %s" % filename)
        
    ## Save combined displacement array (initial displacment + residual displacment):
    if cfg_set["resid_disp_onestep"] and resid:
        ## Load initial displacement field:
        filename_ini = pth.path_creator_UV_disparr("standard",cfg_set)
        #filename_ini = "%stmp/%s_%s_disparr_UV%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                    cfg_set["oflow_source"],cfg_set["file_ext_verif"],
        #                                                    cfg_set["save_type"])
        UVdisparr_ini = iotmp.load_file(filename_ini)       
        
        ## Save summation of initial and residual displacment field
        filename_combi = pth.path_creator_UV_disparr("resid_combi",cfg_set)
        #filename_combi = "%stmp/%s_%s_disparr_UV_resid_combi%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                              cfg_set["oflow_source"],cfg_set["file_ext_verif"],
        #                                                              cfg_set["save_type"])
        #np.savez(filename_combi, Dx=Dx+UVdisparr_ini["Dx"], Dy=Dy+UVdisparr_ini["Dy"],
        #                         Vx=Vx+UVdisparr_ini["Vx"], Vy=Vy+UVdisparr_ini["Vy"])
        iotmp.save_file(filename_combi, data_arr=[Dx+UVdisparr_ini["Dx"][:,:,:],Dy+UVdisparr_ini["Dy"][:,:,:],
                                            Vx+UVdisparr_ini["Vx"][:,:,:],Vy+UVdisparr_ini["Vy"][:,:,:]],
                  var_name=["Dx","Dy","Vx","Vy"],cfg_set=cfg_set)
        print("  ... combined displacement array saved in:\n       %s" % filename_combi)
        
    ## Also save intermediate UV motion vectors:
    if cfg_set["UV_inter"]:
        type_vec = "vec_resid" if resid else "vec"
        filename = pth.path_creator_UV_disparr(type_vec,cfg_set,save_type="npz")
        #filename = "%stmp/%s_%s_disparr_UV_vec%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                   cfg_set["oflow_source"],append_str_resid,cfg_set["file_ext_verif"],
        #                                                   cfg_set["save_type"])
                                                           
        
        ## Give different filename if the motion field is precalculated for training dataset:
        #if cfg_set["precalc_UV_disparr"]:
        #    filename = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], (cfg_set["t0"]-datetime.timedelta(days=1)).strftime("%Y%m%d"),
        #                                                  cfg_set["oflow_source"],append_str_resid,cfg_set["file_ext_verif"],
        #                                                  cfg_set["save_type"])
        
        ## Set last value to np.inf to otbain right numpy array of numpy arrays:
        UV_vec[-1]=np.inf
        UV_vec_sp[-1]=np.inf
        
        ## Save intermediate UV motion vectors:
        #np.savez(filename, UV_vec=UV_vec, UV_vec_sp=UV_vec_sp)
        iotmp.save_file(filename, data_arr=[UV_vec,UV_vec_sp],
                  var_name=["UV_vec","UV_vec_sp"],cfg_set=cfg_set,filetype="npy")
        print("  ... new UV vector lists saved in:\n      %s" % filename)

        
## Displace fields with current displacement arrays:
def displace_fields(cfg_set, resid=False):
    """Displace fields with current displacement arrays.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
    """
    
    ## Loop over variables for displacement:
    #print("Displace variables to time "+cfg_set["t0"].strftime("%d.%m.%Y %H:%M")+"...")
    
    ## Change suffixes of files to read and write in case residual movements are corrected:
    if not resid:
        input_suffix  = "_orig"
        output_suffix = "_disp"
        input_UV_suffix = "standard"
        append_str = ""
    else:
        input_suffix  = "_disp" if not cfg_set["resid_disp_onestep"] else "_orig"
        output_suffix = "_disp_resid" if not cfg_set["resid_disp_onestep"] else "_disp_resid_combi"
        input_UV_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        append_str    = " for residual movement" if not cfg_set["resid_disp_onestep"] else " for residual movement (combi)"
    print_options = [input_suffix,output_suffix,input_UV_suffix,append_str]
    
    ## Get current UV-field
    filename = pth.path_creator_UV_disparr(input_UV_suffix,cfg_set)
    #filename = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
    #                                              cfg_set["oflow_source"], input_UV_suffix, cfg_set["file_ext_verif"],
    #                                              cfg_set["save_type"])
    #UVdisparr = np.load(filename)
    UVdisparr = iotmp.load_file(filename)
    Vx = UVdisparr["Vx"][:,:,:]; Vy = UVdisparr["Vy"][:,:,:]
    #if var=="TRT":
    #    UV_t0 = np.moveaxis(np.dstack((-Vx[-1,:,:],-Vy[-1,:,:])),2,0)
    #else:
    UV_t0 = np.moveaxis(np.dstack((Vx[0,:,:],Vy[0,:,:])),2,0)
    
    ## Load displacement array and add up (don't include most recent displacement array,
    ## as the last displacement to t0 is done by the UV array loaded above
    #if var=="TRT":
    #    Dx = (UVdisparr["Dx"][:-1,:,:])[::-1,:,:]; Dy = (UVdisparr["Dy"][:-1,:,:])[::-1,:,:]
    #else:
    Dx = (UVdisparr["Dx"][:,:,:])[1:,:,:]; Dy = (UVdisparr["Dy"][:,:,:])[1:,:,:]
    Dx_sum = np.cumsum(Dx,axis=0); Dy_sum = np.cumsum(Dy,axis=0)
    if cfg_set["displ_TRT_cellcent"]:
        Dx_sum_neg = np.cumsum(-Dx,axis=0)
        Dy_sum_neg = np.cumsum(-Dy,axis=0)
        
    ## Initiate XYW array of semilagrangian function to avoid going through for-loop
    ## for all variables.
    #XYW_prev_0 = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])*np.nan
    #XYW_prev_1 = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])*np.nan
    #precalc_XYW = False
    
    ## Calculate displacement per variable:
    ## Loop over variables for displacement:
    if len(cfg_set["var_list"])>0 and not cfg_set["displ_TRT_cellcent"]:
        if cfg_set["use_precalc_XYW"]:
            XYW_prev_0, XYW_prev_1 = displace_specific_variable(cfg_set["var_list"][0],cfg_set,print_options,
                                                                UV_t0,Dx_sum,Dy_sum)
            
            num_cores = np.max([multiprocessing.cpu_count()-2,1])
            print("  Parallelising displacement with %s cores" % num_cores)
            Parallel(n_jobs=num_cores)(delayed(displace_specific_variable)(var,cfg_set,print_options,UV_t0,Dx_sum,Dy_sum,
                                               XYW_prev_0=XYW_prev_0,XYW_prev_1=XYW_prev_1) for var in cfg_set["var_list"][1:])

        else:
            for var in cfg_set["var_list"]: displace_specific_variable(var,cfg_set,print_options,
                                                                       UV_t0,Dx_sum,Dy_sum)
            #if cfg_set["displ_TRT_cellcent"]:
            #    raise NotImplementedError("So far backdisplacement of TRT cells only implemented if usage ")
    if cfg_set["displ_TRT_cellcent"]: displace_specific_variable("TRT",cfg_set,print_options,
                                                                 UV_t0=None,Dx_sum=Dx_sum_neg,Dy_sum=Dy_sum_neg,
                                                                 Vx=Vx,Vy=Vy)

     
    # for var in cfg_set["var_list"]:
        # if cfg_set["use_precalc_XYW"]:
            # XYW_prev_0, XYW_prev_1 = displace_specific_variable(var,cfg_set,print_options,
                                                                # UV_t0,Dx_sum,Dy_sum)
            # displace_specific_variable(var,cfg_set,print_options,UV_t0,Dx_sum,Dy_sum,
                                       # XYW_prev_0=XYW_prev_0,XYW_prev_1=XYW_prev_1)
        # else:
            # displace_specific_variable(var,cfg_set,print_options,UV_t0,Dx_sum,Dy_sum,
                                       # XYW_prev_0=XYW_prev_0,XYW_prev_1=XYW_prev_1)
    
    
def displace_specific_variable(var,cfg_set,print_options,UV_t0,Dx_sum,Dy_sum,
                               XYW_prev_0=None,XYW_prev_1=None,Vx=None,Vy=None):
    """Core function of displace_fields (variable specific displacement)"""
    ## Check that Vx and Vy are available if UV_t0 is None:
    if UV_t0 is None and (Vx is None or Vy is None):
        raise ValueError("If UV_t0 is None, the complete Vx and Vy arrays have to be provided.")
    
    ## Set config file entry for cfg_set["use_precalc_XYW"] to False if variable is TRT.
    if var=="TRT": cfg_set["use_precalc_XYW"] = False
    
    ## Unpack printing and naming variables:
    input_suffix    = print_options[0]
    output_suffix   = print_options[1]
    input_UV_suffix = print_options[2]
    append_str      = print_options[3]
    
    ## See whether precalculated XYW arrays are available:
    precalc_XYW = False if (XYW_prev_0 is None or XYW_prev_1 is None) else True
    
    if cfg_set["verbose"]: print("  ... "+var+" is displaced%s" % append_str)
    filename = pth.path_creator_vararr(input_suffix,var,cfg_set)
    #filename = "%stmp/%s_%s%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
    #                                   input_suffix, cfg_set["file_ext_verif"], cfg_set["save_type"])
    vararr = iotmp.load_file(filename,var_name=var)
        
    ## Loop over time steps to create displaced array:
    vararr_disp = np.copy(vararr)
        
    ## Displace every time-step of the variable
    ## Current time step: No displacement
    ## Last time step: Only displacement with current UV-field UV_t0
    adv_method = st.advection.get_method(cfg_set["adv_method"])
    if not precalc_XYW:
        #if var=="TRT": UV_t0 = np.moveaxis(np.dstack((-Vx[0,:,:],-Vy[0,:,:])),2,0)
        #vararr_disp[1,:,:], XYW_prev_ls = adv_method(vararr[1,:,:],UV_t0,1,
        #                                             return_XYW=True)
        if var=="TRT":
            UV_t0 = np.moveaxis(np.dstack((Vx[0,:,:],Vy[0,:,:])),2,0)
            vararr_disp[1,:,:] = adv_method(vararr[1,:,:],UV_t0,1,return_XYW=False,foward_mapping=True)
        else:
            vararr_disp[1,:,:], XYW_prev_ls = adv_method(vararr[1,:,:],UV_t0,1,
                                                         return_XYW=True)
        
        if cfg_set["use_precalc_XYW"]:
            XYW_prev_0 = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])*np.nan
            XYW_prev_1 = np.zeros((cfg_set["n_integ"],)+cfg_set["xy_ext"])*np.nan
            XYW_prev_0[0,:,:] = XYW_prev_ls[0]
            XYW_prev_1[0,:,:] = XYW_prev_ls[1]
          
    else: vararr_disp[1,:,:], = adv_method(vararr[1,:,:],UV_t0,1,
                                           XYW_prev=[XYW_prev_0[0,:,:],XYW_prev_1[0,:,:]])
        
    ## For preceding time steps, also provide cumulatively summed displacement field
    for i in range(2,cfg_set["n_integ"]):
        if var=="TRT": UV_t0 = np.moveaxis(np.dstack((Vx[i-1,:,:],Vy[i-1,:,:])),2,0)
        D_prev_arr = np.moveaxis(np.dstack((Dx_sum[i-2,:,:],Dy_sum[i-2,:,:])),2,0)
        if not precalc_XYW:
            if var=="TRT":
                vararr_disp[i,:,:] = adv_method(vararr_disp[i-1,:,:],UV_t0,1,foward_mapping=True)
            else:
                vararr_disp[i,:,:], XYW_prev_ls = adv_method(vararr[i,:,:],UV_t0,1,D_prev=D_prev_arr,
                                                             return_XYW=True)
            if cfg_set["use_precalc_XYW"]:
                XYW_prev_0[i,:,:] = XYW_prev_ls[0]
                XYW_prev_1[i,:,:] = XYW_prev_ls[1]
        else: vararr_disp[i,:,:] = adv_method(vararr[i,:,:],UV_t0,1,D_prev=D_prev_arr,
                                              XYW_prev=[XYW_prev_0[i,:,:],XYW_prev_1[i,:,:]])
                
    ## If first var has been processed, set precalc_XYW from False to True
    #if cfg_set["use_precalc_XYW"]: precalc_XYW = True
                
    ## Save displaced variable:
    filename = pth.path_creator_vararr(output_suffix,var,cfg_set)
    #filename = "%stmp/%s_%s%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
    #                                    output_suffix, cfg_set["file_ext_verif"], cfg_set["save_type"])
    #np.save(filename, vararr_disp)
    iotmp.save_file(filename, data_arr=vararr_disp,var_name=var,cfg_set=cfg_set)
    if cfg_set["verbose"]: print("      "+var+" displaced array is saved")
            
    ## In case verification of displacements should be performed:
    if cfg_set["verify_disp"] and (cfg_set["resid_disp"]==resid):
        ## Do not calculate skill scores if residual movement is corrected
        ## but currently, the initial displacement was just calculated.
        #if not (cfg_set["resid_disp"] and resid): break
        sksc.calc_skill_scores(cfg_set,var,vararr_disp)
        sksc.calc_statistics(cfg_set,var,vararr_disp)
    
    ## Retrun precalculated arrays:
    if not precalc_XYW: return XYW_prev_0, XYW_prev_1
            
## Do further displacement to correct for residual movement:
def residual_disp(cfg_set):
    """Do further displacement to correct for residual movement.

    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
    """
    
    ## Create new displacement array based on discplaced RZC values:
    create_new_disparray(cfg_set,extra_verbose=False,resid=True)
    displace_fields(cfg_set, resid=True)     
        