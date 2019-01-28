""" [COALITION3] Updating the displacement and variable arrays in an operational context
    (the respective NetCDF or .npy files are opened, but only the most recent observations
    are added, whereas the older observations are kept)."""

from __future__ import division
from __future__ import print_function

import os
import datetime
import numpy as np

from coalition3.inout.paths import path_creator_vararr, path_creator_UV_disparr
from coalition3.inout.iotmp import load_file, save_file
from coalition3.inout.readccs4 import get_vararr_t
from coalition3.operational.lagrangian import displace_fields

## =============================================================================
## FUNCTIONS:
    
    
## Update fields for next time step:
def update_fields(cfg_set,verbose_time=False):
    """Update fields for next time step."""
    
    print("\nUpdate fields for time %s..." % cfg_set["t0"].strftime("%d.%m.%Y %H:%M"))
    t0_old = cfg_set["t0"] - cfg_set["time_change_factor"] * datetime.timedelta(minutes=cfg_set["timestep"])
    
    if verbose_time: t1 = datetime.datetime.now()
    
    ## Load files of different variables
    for var in cfg_set["var_list"]:
        filename_new = path_creator_vararr("orig",var,cfg_set)
        #filename_new = "%stmp/%s_%s_orig%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"), var,
        #                                          cfg_set["file_ext_verif"], cfg_set["save_type"])
        filename_old = path_creator_vararr("orig",var,cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
        #filename_old = "%stmp/%s_%s_orig%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"), var,
        #                                          cfg_set["file_ext_verif"], cfg_set["save_type"])
        
        bool_new_hour = cfg_set["t0"].hour==t0_old.hour
        #if var in ["Wind","Conv"] and bool_new_hour:
        #    vararr = get_vararr_t(cfg_set["t0"], var, cfg_set)
        #    #np.save(filename_new, vararr)
        #    save_file(filename_new, data_arr=vararr,
        #              var_name=var,cfg_set=cfg_set)
        #else:
        ## Load old array, move fields back in time and drop oldest field
        vararr = load_file(filename_old,var_name=var)
        vararr[1:,:,:] = np.copy(vararr)[:-1,:,:]
        
        ## Get field for new time step and assign to newest position
        vararr_t = get_vararr_t(cfg_set["t0"], var, cfg_set)
        vararr[0,:,:] = vararr_t[0,:,:]
        #np.save(filename_new, vararr)
        save_file(filename_new, data_arr=vararr,
                  var_name=var,cfg_set=cfg_set)
        if cfg_set["verbose"]: print("  ... "+var+" is updated")
        if cfg_set["delete_prec"]:
            os.remove(filename_old)
            if cfg_set["verbose"]: print("      and old _orig file removed")
    if verbose_time:  print("     Update _orig files: "+str(datetime.datetime.now()-t1)+"\n")

    
    ## Update disparr fields and displace variables with initial flow field:
    if verbose_time: t1 = datetime.datetime.now()
    update_disparr_fields(cfg_set, t0_old)
    if verbose_time:  print("    Update _disparr file: "+str(datetime.datetime.now()-t1)+"\n")

    if verbose_time: t1 = datetime.datetime.now()
    print("  Displace fields to new time step %s..." % cfg_set["t0"].strftime("%d.%m.%Y %H:%M"))
    displace_fields(cfg_set)
    if verbose_time:  print("    Displace fields: "+str(datetime.datetime.now()-t1)+"\n")
    
    ## Update disparr fields for residual movement and displace variables with
    ## residual movement flow field:
    if cfg_set["resid_disp"]:
        if verbose_time: t1 = datetime.datetime.now()
        update_disparr_fields(cfg_set, t0_old, resid=True)
        if verbose_time:  print("    Update _disparr file (resid): "+str(datetime.datetime.now()-t1)+"\n")

        if verbose_time: t1 = datetime.datetime.now()
        print("  Displace fields (resid) to new time step %s..." % cfg_set["t0"].strftime("%d.%m.%Y %H:%M"))
        displace_fields(cfg_set, resid=True)
        if verbose_time:  print("    Displace fields (resid): "+str(datetime.datetime.now()-t1)+"\n")
    
    ## Delete files associated with preceding time step:
    if verbose_time: t1 = datetime.datetime.now()
    if cfg_set["delete_prec"] and cfg_set["t0"]!=cfg_set["t0_orig"]:
        for var in cfg_set["var_list"]:
            print_extension = ""
            filename_old = path_creator_vararr("disp",var,cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
            #filename_old = "%stmp/%s_%s_disp%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"),
            #                                           var, cfg_set["file_ext_verif"], cfg_set["save_type"])
            os.remove(filename_old)
            if cfg_set["resid_disp"]:
                fileext_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
                filename_old_resid = path_creator_vararr("disp"+fileext_suffix,var,cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
                #filename_old_resid = "%stmp/%s_%s_disp%s%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"),
                #                                                   var, fileext_suffix, cfg_set["file_ext_verif"],
                #                                                   cfg_set["save_type"])
                os.remove(filename_old_resid)
                print_extension = " (including %s)" % fileext_suffix
        print("     and old _disp file removed"+print_extension)
    if verbose_time:  print("    Delete old files: "+str(datetime.datetime.now()-t1)+"\n")        
    
## Helper function of update_fields(cfg_set) updating motion fields:
def update_disparr_fields(cfg_set, t0_old, resid=False):
    """Helper function of update_fields(cfg_set) updating motion fields.
    
    Parameters
    ----------
    
    resid : bool
        Do displacement array creation for residual movement correction?
        Default: False.
    """

    ## Change suffixes of files to read and write in case residual movements are corrected:   
    if not resid:
        #UV_suffix = ""
        append_str = ""
    else:
        #UV_suffix = "_resid" if not cfg_set["resid_disp_onestep"] else "_resid_combi"
        append_str    = " for residual movement" if not cfg_set["resid_disp_onestep"] else " for residual movement (combi)"
    
    print("  Calculate new displacement field%s (%s)..." % (append_str,cfg_set["t0"].strftime("%d.%m.%Y %H:%M")))
    resid_suffix = "resid" if resid else "standard"
    filename_old = path_creator_UV_disparr(resid_suffix,cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
    #filename_old = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"),
    #                                                  cfg_set["oflow_source"], resid_suffix, cfg_set["file_ext_verif"],
    #                                                  cfg_set["save_type"])
    
    ## Load old array, move fields back in time and drop oldest field
    #UVdisparr = np.load(filename_old)
    UVdisparr = load_file(filename_old)
    Vx = UVdisparr["Vx"][:,:,:]; Vy = UVdisparr["Vy"][:,:,:]
    Dx = UVdisparr["Dx"][:,:,:]; Dy = UVdisparr["Dy"][:,:,:]
    Vx[1:,:,:] = np.copy(Vx)[:-1,:,:]; Vy[1:,:,:] = np.copy(Vy)[:-1,:,:]
    Dx[1:,:,:] = np.copy(Dx)[:-1,:,:]; Dy[1:,:,:] = np.copy(Dy)[:-1,:,:]

    ## Get flow field for new time step and assign to newest position
    if cfg_set["UV_inter"]:
        D_new, UV_new, UV_vec_temp, UV_vec_sp_temp = calc_disparr(cfg_set["t0"], cfg_set, resid)
    else:
        D_new, UV_new = calc_disparr(cfg_set["t0"], cfg_set, resid)
    Vx[0,:,:] = UV_new[0,:,:]; Vy[0,:,:] = UV_new[1,:,:]
    Dx[0,:,:] =  D_new[0,:,:]; Dy[0,:,:] =  D_new[1,:,:]
    
    ## Save displacement field file
    filename_new = path_creator_UV_disparr(resid_suffix,cfg_set)
    #filename_new = "%stmp/%s_%s_disparr_UV%s%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
    #                                                  cfg_set["oflow_source"], resid_suffix, cfg_set["file_ext_verif"],
    #                                                  cfg_set["save_type"])
    
    #np.savez(filename_new, Dx=Dx, Dy=Dy, Vx=Vx, Vy=Vy)
    save_file(filename_new, data_arr=[Dx,Dy,Vx,Vy],var_name=["Dx","Dy","Vx","Vy"],
              cfg_set=cfg_set)
    print("  ...UV and displacement arrays are updated"+append_str)
    
    ## Save combined displacement array (initial displacment + residual displacment):
    if cfg_set["resid_disp_onestep"] and resid:
        ## Load initial displacement field:
        filename_ini = path_creator_UV_disparr("standard",cfg_set)
        #filename_ini = "%stmp/%s_%s_disparr_UV%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                cfg_set["oflow_source"],cfg_set["file_ext_verif"], cfg_set["save_type"])
        UVdisparr_ini = load_file(filename_ini)
        
        ## Save summation of initial and residual displacment field
        filename_combi = path_creator_UV_disparr("resid_combi",cfg_set)
        #filename_combi = "%stmp/%s_%s_disparr_UV_resid_combi%s.%s" % (cfg_set["root_path"], cfg_set["t0"].strftime("%Y%m%d%H%M"),
        #                                                              cfg_set["oflow_source"],cfg_set["file_ext_verif"], cfg_set["save_type"])
        #np.savez(filename_combi, Dx=Dx+UVdisparr_ini["Dx"], Dy=Dy+UVdisparr_ini["Dy"],
        #                         Vx=Vx+UVdisparr_ini["Vx"], Vy=Vy+UVdisparr_ini["Vy"])
        save_file(filename_combi, data_arr=[Dx+UVdisparr_ini["Dx"][:,:,:],Dy+UVdisparr_ini["Dy"][:,:,:],
                                            Vx+UVdisparr_ini["Vx"][:,:,:],Vy+UVdisparr_ini["Vy"][:,:,:]],
                  var_name=["Dx","Dy","Vx","Vy"],cfg_set=cfg_set)
        print("      & combined UV and displacement array is updated")
        
        ## Remove old disparr_UV_resid_combi file:
        filename_combi_old = path_creator_UV_disparr("resid_combi",cfg_set,t0=t0_old.strftime("%Y%m%d%H%M"))
        #filename_combi_old = "%stmp/%s_%s_disparr_UV_resid_combi%s.%s" % (cfg_set["root_path"], t0_old.strftime("%Y%m%d%H%M"),
        #                                                                  cfg_set["oflow_source"],cfg_set["file_ext_verif"], cfg_set["save_type"])
        
        if cfg_set["delete_prec"]:
            #if ("disparr" in filename_combi_old or "UV_vec" in filename_combi_old) and filename_combi_old[-4:]==".npy":
            #    filename_combi_old = filename_combi_old[:-4]+".npz"
            os.remove(filename_combi_old)
            print("     and old disparr_UV file"+append_str+" removed")    
    
    if cfg_set["delete_prec"]:
        #if ("disparr" in filename_old or "UV_vec" in filename_old) and filename_old[-4:]==".npy":
        #    filename_old = filename_old[:-4]+".npz"
        os.remove(filename_old)
        print("     and old disparr_UV file"+append_str+" removed")
    
        