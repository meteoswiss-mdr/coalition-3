from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
import configparser
import numpy as np

import pysteps as st

import coalition3.casestudy.casestudy as ccs
import coalition3.inout.readconfig as cfg
import coalition3.inout.paths as pth

sys.path.insert(0, '/data/COALITION2/database/radar/ccs4/python')
import metranet
import matplotlib.pyplot as plt
import subprocess


## ===============================================================================
## Make static setting

## Define path to config file
## Input basics
t0_str = "201805272300"
n_updates = 0
t_start   = datetime.datetime(2015, 7, 7, 11, 45)
t_end     = datetime.datetime(2015, 7, 7, 15, 45)

t_end_alt = datetime.datetime(2015, 7, 7, 18, 30)

#x_range   = (300,355)#(300,370)
#y_range   = (160,220)#(160,230)
x_range   = (350,450)#(300,370)
y_range   = (200,300)#(160,230)

t_end_str = datetime.datetime.strftime(t_end, "%Y%m%d%H%M")

cfg_set, cfg_var, cfg_var_combi = cfg.get_config_info_op()
cfg_set = cfg.cfg_set_append_t0(cfg_set,t_end_str)
cfg_set = ccs.update_cfg_set(cfg_set,t_start,t_end,x_range,y_range,t_end_alt)

#============================================================================================
#============================================================================================

## Produce smaller skill-score plot:
#Nip.compare_skillscores_help(cfg_set,"BZC",[1.0,8.0,95.0,99.0,99.9])

#============================================================================================

#HRV = np.load("/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/tmp/201507071830_HRV_disp_sub_1145_1545.npy")
#plt.imshow(HRV[8,:,:])
#plt.show()

def plot_vararr_disp(vararr_disp,t0):
    fig, ax = plt.subplots(figsize=(7.1,6.4), dpi=100)
    fig.subplots_adjust(0,0,1,1)
    ax.clear()
    ax.axis('off')

    ax.contour(vararr_disp[::-1,:],linewidths=3,colors='orangered') #"1f77b4" "#ff7f0e")
    #plt.show()
    filename = '/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/casestudy/im_series/frame/disp_%s.png' % t0.strftime("%Y%m%d%H%M")
    fig.savefig(filename, transparent=True)
    
def plot_precip(precip_arr,UV_arr,t0):
    geodata = title = colorbar = None
    colorscale="MeteoSwiss"
    units="mm/h"
    fig, ax_pre = plt.subplots(figsize=(7.1,6.4), dpi=100)
    fig.subplots_adjust(0,0,1,1)
    ax_pre.clear()
    ax_pre.axis('off')
    ax_pre = st.plt.plot_precip_field(precip_arr, False, geodata,
                                   units=units, colorscale=colorscale,
                                   title=title,
                                   colorbar=colorbar,ax_pre=ax_pre)
    st.plt.quiver(UV_arr, geodata, step = 50, color = "orange")
    filename = '/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/casestudy/im_series/RZC/RZC_quiv_%s.png' % t0.strftime("%Y%m%d%H%M")
    fig.savefig(filename, transparent=True)
    #plt.show()

def plot_box(cfg_set):
    ## Import datasets:
    UVdisparr = np.load("/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/tmp/201507071830_RZC_disparr_UV.npz")
    RZC = np.load("/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/tmp/201507071830_RZC_orig.npy")
    Vx = UVdisparr["Vx"][0:,:,:]; Vy = UVdisparr["Vy"][0:,:,:]
    UV_t0 = np.moveaxis(np.dstack((Vx[0,:,:],Vy[0,:,:])),2,0)
    Dx = UVdisparr["Dx"]; Dy = UVdisparr["Dy"]
    
    ## Add up displacement:
    #Dx_sum = np.cumsum(Dx[33:,:,:],axis=0); Dy_sum = np.cumsum(Dy[33:,:,:],axis=0)
    Dx_sum = np.cumsum(Dx[0:,:,:],axis=0); Dy_sum = np.cumsum(Dy[0:,:,:],axis=0)
    
    ## Set t0:
    t0 = t_end_alt
    #Vx = UVdisparr["Vx"][33:,:,:]; Vy = UVdisparr["Vy"][33:,:,:]
    #t0 = t_end
    
    ## Produce and plot rectangular window for t0
    arr_frame = np.zeros((640,710))
    arr_frame[190:290,380:480] = 1
    plot_vararr_disp(arr_frame,t0)

    ## Plot precip at t0    
    plot_precip(RZC[0,:,:],UV_t0,t0)
    
    ## Advect precip to t-5min    
    adv_method = st.advection.get_method("semilagrangian") 
    arr_frame_disp = adv_method(arr_frame,-UV_t0,1)
    
    ## Update t0 to t-5min and plot precip
    t0 = t0 - datetime.timedelta(minutes=5)    
    plot_vararr_disp(arr_frame_disp[0,:,:],t0)
    
    ## Loop over the dataset:
    i_Dsum = -1
    for _ in range(96):
        
        ## Update t0:
        t0 = t0 - datetime.timedelta(minutes=5)
        print(t0.strftime("%Y%m%d%H%M"))
        i_Dsum += 1
        
        ## Plot HRoverview image
        #plot_HRoverview(t0)
        #continue
        
        ## Advect and plot displacement array
        UV_t0 = np.moveaxis(np.dstack((Vx[i_Dsum+1,:,:],Vy[i_Dsum+1,:,:])),2,0)
        D_prev_arr = np.moveaxis(np.dstack((Dx_sum[i_Dsum,:,:],Dy_sum[i_Dsum,:,:])),2,0)
        arr_frame_disp = adv_method(arr_frame,-UV_t0,1,D_prev=-D_prev_arr)
        plot_vararr_disp(arr_frame_disp[0,:,:],t0)
        
        ## Plot precipitation field
        plot_precip(RZC[i_Dsum+1,:,:],UV_t0,t0)
        
        ## Make composite of these images
        filename_RZC    = '/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/casestudy/im_series/RZC/RZC_quiv_%s.png' % t0.strftime("%Y%m%d%H%M")
        filename_frame    = '/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/casestudy/im_series/frame/disp_%s.png' % t0.strftime("%Y%m%d%H%M")
        filename_HRV      = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/casestudy/im_series/HRoverview/%s" % t0.strftime('MSG_ccs4_%y%m%d%H%M.png')
        filename_HRVframe = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/casestudy/im_series/Composit/%s" % t0.strftime('MSG_ccs4_%y%m%d%H%M_comp.png')
        #if t0.minute%15==0:
        subprocess.call('composite %s %s %s' % (filename_RZC,filename_HRV,filename_HRVframe), shell=True)
        subprocess.call('composite %s %s %s' % (filename_frame,filename_HRVframe,filename_HRVframe), shell=True)

def plot_HRoverview(t0):
    path = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/casestudy/im_series/HRoverview/"
    subprocess.call('python demo_msg_HRoverview.py %s %s ' % (t0.strftime("%Y %m %d %H %M"),path), shell=True)
            
# plot_box(cfg_set)

#============================================================================================

def plot_oflow_derivation(cfg_set,t0,dt):
    ## Import datasets:
    #RZC = np.load("/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/tmp/201507071830_RZC_orig.npy")
    #UVdisparr = np.load("/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/tmp/201507071830_RZC_disparr_UV.npz")
    #Vx = UVdisparr["Vx"][0:,:,:]; Vy = UVdisparr["Vy"][0:,:,:]
    #UV_t0 = np.moveaxis(np.dstack((Vx[0,:,:],Vy[0,:,:])),2,0)
    #Dx = UVdisparr["Dx"]; Dy = UVdisparr["Dy"]
    
    ## Get index of respective RZC fields at t0
    ind  = (cfg_set["t_end_alt"]-t0).seconds/60/cfg_set["timestep"]
    ind2 = ind+(dt/cfg_set["timestep"])
        
    ## Read in current oflow_source file:
    t_current = t0
    cfg_set["timestep"] = dt
    filenames, timestamps = pth.path_creator(t_current, cfg_set["oflow_source"], "RADAR", cfg_set)
    ret = metranet.read_file(filenames[0], physic_value=True)
    oflow_source_data = np.atleast_3d(ret.data)
    
    for filename in filenames[1:]:
        ret_d_t = metranet.read_file(filename, physic_value=True)
        oflow_source_data_d_t = np.atleast_3d(ret_d_t.data)
        oflow_source_data = np.append(oflow_source_data,oflow_source_data_d_t, axis=2)
    
    oflow_source_data = np.moveaxis(oflow_source_data,2,0)
    #oflow_source_data_masked = np.ma.masked_invalid(oflow_source_data)
    #oflow_source_data_masked = np.ma.masked_where(oflow_source_data_masked==0,oflow_source_data_masked)    
    
    ## convert linear rainrates to logarithimc dBR units
    if cfg_set["oflow_source"]=="RZC":
        if cfg_set["R_thresh_meth"] == "fix":
            R_tresh = cfg_set["R_threshold"]
        elif cfg_set["R_thresh_meth"] == "perc":
            R_tresh = np.min([np.nanpercentile(oflow_source_data[0,:,:],cfg_set["R_threshold"]),
                              np.nanpercentile(oflow_source_data[1,:,:],cfg_set["R_threshold"])])
        else: raise ValueError("R_thresh_meth must either be set to 'fix' or 'perc'")
        
        dBR, dBRmin = st.utils.mmhr2dBR(oflow_source_data, R_tresh)
        dBR[~np.isfinite(dBR)] = dBRmin
        #R_thresh = cfg_set["R_threshold"]
        
        ## In case threshold is not exceeded, lower R_threshold by 20%
        while (dBR==dBRmin).all():
            print("   *** Warning: Threshold not exceeded, lower R_threshold by 20% to "+str(R_thresh*0.8)+" ***")
            R_thresh = R_thresh*0.8
            dBR, dBRmin = st.utils.mmhr2dBR(oflow_source_data, R_thresh)
            dBR[~np.isfinite(dBR)] = dBRmin
    else:
        raise ValueError("So far displacement array retrieval only implemented for RZC")
        
    ## Calculate UV field
    oflow_method = st.optflow.get_method(cfg_set["oflow_method_name"])
    UV = oflow_method(dBR,return_single_vec=True)[1]
    UV_decl  = oflow_method(dBR,return_declust_vec=True)[0]
    UV_final = oflow_method(dBR)
    #plt.imshow(oflow_source_data[0,:,:])
    UV_final = np.stack([-UV_final[0,:,:],UV_final[1,:,:]])
    print(UV.shape)
    print(UV_decl.shape)
    
    #fig, axs = plt.subplots(1,2, figsize=(10,6.5))
    fig = plt.figure(figsize=(14,9.5)) 
    
    from matplotlib import gridspec
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1, 1],
                           wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.05, right=0.95) 

    if t0==datetime.datetime(2015,07,07,15,00):
        xlimit = (270,340)
        ylimit = (360,380)
    elif t0==datetime.datetime(2015,07,07,14,00):
        xlimit = (140,210)
        ylimit = (310,405)
    else:
        xlimit = None
        ylimit = None
        
    cmap, norm, clevs, clevsStr = st.plt.get_colormap("mm/h", "MeteoSwiss")
    for i in range(2):
        #axs[i].imshow(dBR[i,:,:],interpolation='nearest', cmap=cmap, norm=norm, alpha=0.3)
        ax = plt.subplot(gs[i])
        ax.imshow(dBR[i,:,:],interpolation='nearest', cmap=cmap, norm=norm, alpha=0.3) #s[i]
        st.plt.quiver(-UV_final, step = 25, color = "orange")
        ax.quiver(UV_decl[0,:]+0.3, UV_decl[1,:]+0.3, UV_decl[2,:], UV_decl[3,:], angles='xy', scale_units='xy', scale=1, color="#1f77b4") #,width=1.2)
        ax.quiver(UV[1,:,0], UV[0,:,0], UV[2,:,0], UV[3,:,0], angles='xy', scale_units='xy', scale=1, color="#d62728") #,width=1)
        ax.set_xlim(xlimit)
        ax.set_ylim(ylimit)
        ax.axis('off')
        ax.patch.set_visible(False)
        #axs[i].margins(x=0.01,y=0.01)
    
    #fig.subplots_adjust(0,0,1,1)
    #plt.tight_layout()
    #plt.show()
    #fig.patch.set_visible(False)
    path_to_figures = os.path.join(cfg_set["fig_output_path"],"poster_plots/")
    if not os.path.exists(path_to_figures):
        os.makedirs(path_to_figures)
    filename = os.path.join(path_to_figures,t0.strftime('oflow_%y%m%d%H%M.pdf'))
    fig.savefig(filename, transparent=True, bbox_inches='tight')
    

plot_oflow_derivation(cfg_set, t0=datetime.datetime(2015,07,07,14,00), dt=5)





















