""" [COALITION3] Creating control plots of displacement (original and displaced fields
    side by side)"""

from __future__ import division
from __future__ import print_function

import os
import datetime
import numpy as np
import pysteps as st
import matplotlib.pylab as plt
from coalition3.inout.paths import path_creator_vararr, path_creator_UV_disparr
from coalition3.inout.iotmp import load_file

## =============================================================================
## FUNCTIONS:
               
## Plot displaced fields next to original ones:
def plot_displaced_fields(var,cfg_set,future=False,animation=False,TRT_form=False):
    """Plot displaced fields next to original ones.

    Parameters
    ----------
    
    var : str
        Variable which should be plotted
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
    """
    
    """
    if not resid:
        resid_str  = ""
        resid_suffix = "standard"
        resid_suffix_plot = ""
        resid_suffix_vec = "vec"
        resid_dir = ""
    else:
        resid_str  = " residual" if not cfg_set["resid_disp_onestep"] else " residual (combi)"
        resid_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        resid_suffix_plot = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        resid_suffix_vec = "vec_resid" if not cfg_set["resid_disp_onestep"] else "vec"
        resid_dir = "/_resid" if not cfg_set["resid_disp_onestep"] else "/_resid_combi"    
    """
    #resid_str = "" if not resid else " residual"
    #resid_suffix = "" if not resid else "_resid"
    #resid_dir = "" if not resid else "/_resid"
    print("Plot animation of %s..." % (var))
    
    
    ## Load files
    filename_orig = path_creator_vararr("orig",var,cfg_set,disp_reverse="")
    print("   File of original variable:         %s" % filename_orig)
    vararr = load_file(filename_orig,var_name=var)
    if future:
        filename_orig_future = path_creator_vararr("orig",var,cfg_set,disp_reverse="_rev")
        print("   File of future original variable:  %s" % filename_orig_future)
        vararr_future = load_file(filename_orig_future,var_name=var)
        vararr = np.concatenate([vararr_future[1:,:,:][::-1,:,:],vararr],axis=0)
    vararr = vararr[::-1,:,:]
    
    filename_UV = path_creator_UV_disparr("standard",cfg_set,disp_reverse="")
    print("   File of UV field:                  %s" % filename_UV)
    UVdisparr = load_file(filename_UV)
    Vx = UVdisparr["Vx"][:,:,:]; Vy = UVdisparr["Vy"][:,:,:]
    if future:
        filename_UV_future = path_creator_UV_disparr("standard",cfg_set,disp_reverse="_rev")
        print("   File of future UV field:           %s" % filename_UV_future)
        UVdisparr_future = load_file(filename_UV)
        Vx_future = UVdisparr_future["Vx"][1:,:,:][::-1,:,:]
        Vy_future = UVdisparr_future["Vy"][1:,:,:][::-1,:,:]
        Vx = np.concatenate([Vx_future,Vx],axis=0)
        Vy = np.concatenate([Vy_future,Vy],axis=0)    
    Vx = Vx[::-1,:,:]; Vy = Vy[::-1,:,:]
    
    cmap, norm, clevs, clevsStr = st.plt.get_colormap("mm/h", "MeteoSwiss")

    ## Set values to nan
    #plt.hist(vararr[~np.isnan(vararr)].flatten()); plt.show(); return
    if var=="RZC":
        vararr[vararr < 0.1] = np.nan
    elif var=="BZC":
        vararr[vararr <= 0] = np.nan
    elif var=="LZC":
        vararr[vararr < 5] = np.nan
    elif var=="MZC":
        vararr[vararr <= 2] = np.nan
    elif var in ["EZC","EZC45","EZC15"]:
        vararr[vararr < 1] = np.nan
    #elif "THX" in var:
    #    vararr[vararr < 0.001] = np.nan
    
    
    ## Get forms of TRT cells:
    if TRT_form:
        filename_TRT_disp = path_creator_vararr("disp","TRT",cfg_set,disp_reverse="")
        filename_TRT_disp = filename_TRT_disp[:-3]+"_domain.nc"
        print("   File of TRT domain:                %s" % filename_TRT_disp)
        vararr_TRT = load_file(filename_TRT_disp,var_name="TRT")
    
        if future:
            filename_TRT_disp_future = filename_TRT_disp[:-3]+"_rev.nc"
            vararr_TRT_future = load_file(filename_TRT_disp_future,var_name="TRT")
            vararr_TRT = np.concatenate([vararr_TRT_future[1:,:,:][::-1,:,:],vararr_TRT],axis=0)
        vararr_TRT = vararr_TRT[::-1,:,:]
 
    t_delta = np.array(range(cfg_set["n_integ"]))*-cfg_set["timestep"]
    if future:
        t_delta = np.concatenate([np.arange(1,cfg_set["n_integ"])[::-1] * \
                                  cfg_set["timestep"],t_delta])
    t_delta = t_delta[::-1]
    #t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=-cfg_set["timestep"])
    #if future:
    #    t_delta = np.concatenate([np.arange(1,cfg_set["n_integ"])[::-1] * \
    #                              datetime.timedelta(minutes=cfg_set["timestep"]),t_delta])

    ## Setup the plot:
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(10,10)
    plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
    plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
    
    ## Make plots
    for i in range(vararr.shape[0]):
        #if animation: plt.cla()
        
        ## Prepare UV field for quiver plot
        UV_field = np.moveaxis(np.dstack((Vx[i,:,:],Vy[i,:,:])),2,0)
        step = 40; X,Y = np.meshgrid(np.arange(UV_field.shape[2]),np.arange(UV_field.shape[1]))
        UV_ = UV_field[:, 0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]
        X_ = X[0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]; Y_ = Y[0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]
        
        
        ## Make plot:
        if not animation:
            fig, axes = plt.subplots(nrows=1, ncols=1)
            fig.set_size_inches(10,10)
            plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
            plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        axes.grid(which='major', color='orange', linestyle='-', linewidth=0.5)
        
        ## Generate title:
        t_current = cfg_set["t0"] + datetime.timedelta(minutes=t_delta[i])
        title_str_basic = "%s fields at %s" % (var, t_current.strftime("%d.%m.%Y - %H:%M"))
        if t_current > cfg_set["t0"]: title_str_addon = "(Future t0 + %02dmin)" % t_delta[i]
        elif t_current < cfg_set["t0"]: title_str_addon = "(Past t0 - %02dmin)" % -t_delta[i]
        elif t_current == cfg_set["t0"]: title_str_addon = "(Present t0 + 00min)"
        title_str = "%s\n%s" % (title_str_basic,title_str_addon)

        #axes.cla()
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        axes.grid(which='major', color='orange', linestyle='-', linewidth=0.5)
        axes.set_title(title_str)
        axes.imshow(vararr[i,:,:], aspect='equal', cmap=cmap)
        axes.quiver(X_, Y_, UV_[0,:,:], -UV_[1,:,:], pivot='tip', color='grey')
        #if cfg_set["UV_inter"] and type(UV_vec[i]) is not float and len(UV_vec[i].shape)>1:
        #    axes.quiver(UV_vec[i][1,:,0], UV_vec[i][0,:,0], UV_vec[i][2,:,0], -UV_vec[i][3,:,0], pivot='tip', color='red')
        #    axes.quiver(UV_vec_sp[i][1,:], UV_vec_sp[i][0,:], UV_vec_sp[i][2,:], -UV_vec_sp[i][3,:], pivot='tip', color='blue')
        #axes[1].set_title('Displaced')
        #axes[1].imshow(vararr_disp[i,:,:], aspect='equal', cmap=cmap)
        if TRT_form:
            #col_ls = ["r","b","g","m","k"]
            #for circ in range(len(iCH_ls)):
            #    col = col_ls[circ%len(col_ls)]
            #    axes[1].contour(circle_array[:,:,circ],linewidths=0.3,alpha=0.7) #,levels=diameters)
            axes.contour(vararr_TRT[i,:,:],linewidths=1,alpha=1,color="red") #,levels=diameters)
        #axes[1].grid(which='major', color='red', linestyle='-', linewidth=1)

        #fig.tight_layout()    
        
        if animation:
            #plt.show()
            plt.pause(1)
            axes.clear()
        else:
            path = "%sTRT_cell_disp/%s/%s/" % (cfg_set["output_path"],cfg_set["t0_str"],var)
            new_dir = ""
            if not os.path.exists(path):
                os.makedirs(path)
                new_dir = "(new) "
            figname = "%s%s_%s_TRT_domains.png" % (path, t_current.strftime("%Y%m%d%H%M"), var)
            fig.savefig(figname, dpi=100)
            if i == vararr.shape[0]-1:
                print("   Plots saved in %sdirectory: %s" % (new_dir, path))
    plt.close()  

    
## Displace fields with current displacement arrays:
def plot_displaced_fields_old(var,cfg_set,resid=False,animation=False,TRT_form=False):
    """Plot displaced fields next to original ones.

    Parameters
    ----------
    
    var : str
        Variable which should be plotted
        
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
    """
    if not resid:
        resid_str  = ""
        resid_suffix = "standard"
        resid_suffix_plot = ""
        resid_suffix_vec = "vec"
        resid_dir = ""
    else:
        resid_str  = " residual" if not cfg_set["resid_disp_onestep"] else " residual (combi)"
        resid_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        resid_suffix_plot = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        resid_suffix_vec = "vec_resid" if not cfg_set["resid_disp_onestep"] else "vec"
        resid_dir = "/_resid" if not cfg_set["resid_disp_onestep"] else "/_resid_combi"    
    
    #resid_str = "" if not resid else " residual"
    #resid_suffix = "" if not resid else "_resid"
    #resid_dir = "" if not resid else "/_resid"
    print("Plot comparison of moving and displaced%s %s..." % (resid_str,var))
    
    
    ## Load files
    if not resid or cfg_set["resid_disp_onestep"]:
        filename_orig = path_creator_vararr("orig",var,cfg_set)
        #filename_orig = "%stmp/%s_%s_orig%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
        #                                           cfg_set["file_ext_verif"], cfg_set["save_type"])
    else:
        filename_orig = path_creator_vararr("disp",var,cfg_set)
        #filename_orig = "%stmp/%s_%s_disp%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
        #                                           cfg_set["file_ext_verif"], cfg_set["save_type"])
    print("   File of original variable:  %s" % filename_orig)
    vararr = load_file(filename_orig,var_name=var)
    resid_suffix_temp = "" if resid_suffix=="standard" else resid_suffix
    filename_disp = path_creator_vararr("disp"+resid_suffix_temp,var,cfg_set)
    #filename_disp = "%stmp/%s_%s_disp%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
    #                                             resid_suffix, cfg_set["file_ext_verif"], cfg_set["save_type"])
    print("   File of displaced variable: %s" % filename_disp)
    vararr_disp = load_file(filename_disp,var_name=var)
    filename_UV = path_creator_UV_disparr(resid_suffix,cfg_set)
    #filename_UV = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
    #                                                  cfg_set["oflow_source"], resid_suffix, cfg_set["file_ext_verif"],
    #                                                  cfg_set["save_type"])
    print("   File of UV field:           %s" % filename_UV)
    if cfg_set["UV_inter"]:
        filename_UV_vec = path_creator_UV_disparr(resid_suffix_vec,cfg_set,save_type="npz")
        #filename_UV_vec = "%stmp/%s_%s_disparr_UV_%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                      cfg_set["oflow_source"], resid_suffix_vec, cfg_set["file_ext_verif"], 
        #                                                      cfg_set["save_type"])
        print("   File of UV vectors: %s" % filename_UV_vec)
    cmap, norm, clevs, clevsStr = st.plt.get_colormap("mm/h", "MeteoSwiss")

    ## Set values to nan
    #plt.hist(vararr[~np.isnan(vararr)].flatten()); plt.show(); return
    if var=="RZC":
        vararr[vararr < 0.1] = np.nan
        vararr_disp[vararr_disp < 0.1] = np.nan
    elif var=="BZC":
        vararr[vararr <= 0] = np.nan
        vararr_disp[vararr_disp <= 0] = np.nan
    elif var=="LZC":
        vararr[vararr < 5] = np.nan
        vararr_disp[vararr_disp < 5] = np.nan
    elif var=="MZC":
        vararr[vararr <= 2] = np.nan
        vararr_disp[vararr_disp <= 2] = np.nan
    elif var in ["EZC","EZC45","EZC15"]:
        vararr[vararr < 1] = np.nan
        vararr_disp[vararr_disp < 1] = np.nan
    elif "THX" in var:
        vararr[vararr < 0.001] = np.nan
        vararr_disp[vararr_disp < 0.001] = np.nan
    
    ## Prepare UV field for quiver plot
    #UVdisparr = np.load(filename_UV)
    UVdisparr = load_file(filename_UV)
    Vx = UVdisparr["Vx"][:,:,:]; Vy = UVdisparr["Vy"][:,:,:]
    UV_t0 = np.moveaxis(np.dstack((Vx[0,:,:],Vy[0,:,:])),2,0)
    
    ## Get forms of TRT cells:
    if TRT_form:
        print("   *** Warning: This part of the code (plotting the circles) is completely hard-coded! ***")
        TRT_info_file = pd.read_pickle("/opt/users/jmz/0_training_NOSTRADAMUS_ANN/TRT_sampling_df_testset_enhanced.pkl")
        jCH_ls = TRT_info_file["jCH"].loc[(TRT_info_file["date"]==cfg_set["t0"]) & (TRT_info_file["RANKr"]>=10)].tolist()
        iCH_ls = TRT_info_file["iCH"].loc[(TRT_info_file["date"]==cfg_set["t0"]) & (TRT_info_file["RANKr"]>=10)].tolist()
                
        diameters = [16,24,32]
        circle_array = np.zeros((640,710,len(iCH_ls)))
        for ind in range(len(jCH_ls)):
            X, Y = np.meshgrid(np.arange(0, 710), np.arange(0, 640))
            for diameter in diameters:
                interior = ((X-jCH_ls[ind])**2 + (Y-iCH_ls[ind])**2) < (diameter/2)**2
                circle_array[interior,ind] += 1
    
    ## Get UV vectors if these were created:
    if cfg_set["UV_inter"]:
        #UV_vec_arr = np.load(filename_UV_vec)
        UV_vec_arr = load_file(filename_UV_vec)
        UV_vec = UV_vec_arr["UV_vec"]; UV_vec_sp = UV_vec_arr["UV_vec_sp"]
        #print(UV_vec)
        #print(len(UV_vec))
        #print(UV_vec_sp)
        #print(len(UV_vec_sp))
    #pdb.set_trace()
    ## Get time array
    t_delta = np.array(range(cfg_set["n_integ"]))*datetime.timedelta(minutes=cfg_set["timestep"])
    
    ## Setup the plot:
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12.5, 6.5)
    plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
    plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
    
    ## Make plots
    #if animation: plt.clf()
    for i in range(vararr.shape[0]):
    
        ## Prepare UV field for quiver plot
        UV_field = np.moveaxis(np.dstack((Vx[i,:,:],Vy[i,:,:])),2,0)
        step = 40; X,Y = np.meshgrid(np.arange(UV_field.shape[2]),np.arange(UV_field.shape[1]))
        UV_ = UV_field[:, 0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]
        X_ = X[0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]; Y_ = Y[0:UV_field.shape[1]:step, 0:UV_field.shape[2]:step]
        
        ## Generate title:
        t_current = cfg_set["t0"] - t_delta[i]
        title_str = "%s fields at %s" % (var, t_current.strftime("%d.%m.%Y - %H:%M"))
        plt.suptitle(title_str); 
        
        """## Make plot:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12.5, 6.5)
        plt.suptitle(title_str); 
        plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
        plt.setp(axes, xticks=np.arange(50, 1000, 50), yticks=np.arange(50, 1000, 50))
        for ax in axes:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(which='major', color='orange', linestyle='-', linewidth=0.5)
        """

        for ax in axes:
            ax.cla()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(which='major', color='orange', linestyle='-', linewidth=0.5)
        axes[0].set_title('Original')
        axes[0].imshow(vararr[i,:,:], aspect='equal', cmap=cmap)
        axes[0].quiver(X_, Y_, UV_[0,:,:], -UV_[1,:,:], pivot='tip', color='grey')
        if cfg_set["UV_inter"] and type(UV_vec[i]) is not float and len(UV_vec[i].shape)>1:
            axes[0].quiver(UV_vec[i][1,:,0], UV_vec[i][0,:,0], UV_vec[i][2,:,0], -UV_vec[i][3,:,0], pivot='tip', color='red')
            axes[0].quiver(UV_vec_sp[i][1,:], UV_vec_sp[i][0,:], UV_vec_sp[i][2,:], -UV_vec_sp[i][3,:], pivot='tip', color='blue')
        axes[1].set_title('Displaced')
        axes[1].imshow(vararr_disp[i,:,:], aspect='equal', cmap=cmap)
        if TRT_form:
            col_ls = ["r","b","g","m","k"]
            for circ in range(len(iCH_ls)):
                col = col_ls[circ%len(col_ls)]
                axes[1].contour(circle_array[:,:,circ],linewidths=0.3,alpha=0.7) #,levels=diameters)
        #axes[1].grid(which='major', color='red', linestyle='-', linewidth=1)

        fig.tight_layout()    
        
        if animation:
            #plt.show()
            plt.pause(.2)
            #plt.clf()
        else:        
            figname = "%scomparison_move_disp/%s%s/%s_%s%s_displacement.png" % (cfg_set["output_path"], var, resid_dir,
                                                                                t_current.strftime("%Y%m%d%H%M"), var, resid_suffix_plot)
            fig.savefig(figname, dpi=100)
    plt.close()
    
  
## Calculate optical flow displacement array for specific time-step
def calc_disparr_ctrl_plot(D,timestamps,oflow_source_data,oflow_source_data_forecast,cfg_set):
    """Plot and save two frames as input of LucasKanade and the advected field, as well as plot the displacement field.
   
    Parameters
    ----------
    
    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
    
    timestamps : datetime object
        Timestamps of optical flow input fields.
    
    D : array-like
        Displacement field.
    
    oflow_source_data : array-like
        Optical flow input fields.
        
    oflow_source_data_forecast : array-like
        Advected optical flow input field.
    """
    
    plt.subplot(1, 2, 1)
    plt.imshow(D[0,:,:])
    plt.title('X component of displacement field')

    plt.subplot(1, 2, 2)
    plt.imshow(D[1,:,:])
    plt.title('Y component of displacement field')
    #plt.show()

    # Plot two frames for UV-derivation and advected frame
    for i in range(oflow_source_data.shape[0] + 1):
        plt.clf()
        if True:
            if i < oflow_source_data.shape[0]:
                # Plot last observed rainfields
                st.plt.plot_precip_field(oflow_source_data[i,:,:], None, units="mmhr",
                              colorscale=cfg_set["colorscale"], 
                              title=timestamps[i].strftime("%Y-%m-%d %H:%M"), 
                              colorbar=True)
                
                figname = "%s%s_%s_simple_advection_%02d_obs.png" % (cfg_set["output_path"], timestamps[i].strftime("%Y%m%d%H%M"),
                                                                      cfg_set["oflow_source"], i)
                plt.savefig(figname)
                print(figname, 'saved.')
            else:
                # Plot nowcast
                st.plt.plot_precip_field(oflow_source_data_forecast[i - oflow_source_data.shape[0],:,:], 
                              None, units="mmhr", 
                              title="%s +%02d min" % 
                              (timestamps[-1].strftime("%Y-%m-%d %H:%M"),
                              (1 + i - oflow_source_data.shape[0])*cfg_set["timestep"]),
                              colorscale=cfg_set["colorscale"], colorbar=True)
                figname = "%s%s_%s_simple_advection_%02d_nwc.png" % (cfg_set["output_path"], timestamps[-1].strftime("%Y%m%d%H%M"),
                                                                      cfg_set["oflow_source"], i)
                plt.savefig(figname)
                print(figname, "saved.")

