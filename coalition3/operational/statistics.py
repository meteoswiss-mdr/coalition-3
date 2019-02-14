""" [COALITION3] Reading the statistics within the domains around the TRT cell centres,
    and add auxiliary statistics (solar time, altitude) and derived variables (TRT Ranks)."""

from __future__ import division
from __future__ import print_function

import os
import datetime
import pickle
import ephem
import configparser

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pylab as plt
from scipy import ndimage, signal, interpolate, spatial

from mpop.satin import swissradar

from coalition3.inout.iotmp import save_nc, load_file
from coalition3.inout.paths import path_creator_vararr, path_creator_UV_disparr

## =============================================================================
## FUNCTIONS:

## Function which reads the indices corresponding to the domain of interest around the displaced TRT cell centres:
def read_TRT_area_indices(cfg_set_input,reverse):
    """Function which reads the indices corresponding to the domain
    of interest around the displaced TRT cell centres"""

    ## Change settings related calculating statistics from future or current observations:
    cfg_set = cfg_set_input.copy()
    cfg_set["future_disp_reverse"] = True if reverse else False
    cfg_set["time_change_factor"]  = -1   if reverse else 1
    string_time = "future" if cfg_set["future_disp_reverse"] else "past"
    print("Read %s indices of the domains of interest..." % string_time)

    ## Read TRT info dataframe:
    filename = "%stmp/%s%s" % (cfg_set["root_path"],
                               cfg_set["t0"].strftime("%Y%m%d%H%M"),
                               "_TRT_df.pkl")
    cell_info_df = pd.read_pickle(filename)
    cell_info_df = cell_info_df.loc[cell_info_df["RANKr"] >= cfg_set["min_TRT_rank_op"]*10]
    cell_info_df["Border_cell"] = False
    
    ## Correct date column (repetitions out of nowhere...)
    if len(cell_info_df[["date"]].values[0][0])>12:
        cell_info_df["date"] = np.array([date_i[:12] for date_i in cell_info_df["date"].values],dtype=np.object)
        
    ## Read file with displaced TRT centres:
    orig_disp_TRT = "disp" if cfg_set["displ_TRT_cellcent"] else "orig"
    filename = path_creator_vararr(orig_disp_TRT,"TRT",cfg_set)
    TRTarr = load_file(filename,var_name="TRT")

    ## Create Meshgrid or boolean array
    if cfg_set["stat_sel_form"] == "square":
        X = None
        Y = None
    elif cfg_set["stat_sel_form"] == "circle":
        X, Y = np.meshgrid(np.arange(0,TRTarr.shape[2]),np.arange(0,TRTarr.shape[1]))
    else: raise ValueError("stat_sel_form can only be 'square' or 'circle'")

    ## Create array to save indices in:
    time_delta_coord = np.arange(cfg_set["n_integ"],dtype=np.int16) * cfg_set["timestep"]
    arr_index        = np.zeros((cell_info_df.shape[0],len(time_delta_coord),cfg_set["stat_sel_form_size"]),dtype=np.uint32)
    xarr_index_flat = xr.DataArray(arr_index,
                              coords=[np.array(cell_info_df.index.tolist(),dtype=np.object),
                                      time_delta_coord, #time_dir_coord,
                                      np.arange(cfg_set["stat_sel_form_size"],dtype=np.int32)],
                              dims=['TRT_ID', 'time_delta', 'pixel_indices'], #'time_dir',
                              name="TRT_domain_indices")
    xarr_index_ij  = xr.DataArray(np.zeros((cell_info_df.shape[0],len(time_delta_coord),2),dtype=np.uint16),
                              coords=[np.array(cell_info_df.index.tolist(),dtype=np.object),
                                      time_delta_coord,
                                      np.array(['CHi','CHj'],dtype=np.object)],
                              dims=['TRT_ID', 'time_delta', 'CHi_CHj'], #'time_dir',
                              name="TRT_cellcentre_indices")

    ## Save nc file showing domains around TRT cells for control:
    if cfg_set["save_TRT_domain_map"]:
        TRTarr_plot = TRTarr.copy()
        TRTarr_plot[:,:,:] = int(-1)

    ## Loop over TRT-cells to read in the statistics:
    for cell in cell_info_df.index.tolist():
        ind_triple = zip(*np.where(TRTarr==int(cell[8:])))

        if len(ind_triple) != cfg_set["n_integ"]:
            if len(ind_triple) < cfg_set["n_integ"]: print("   *** Warning: TRT cell centre is moved out of domain ***")
            if len(ind_triple) > cfg_set["n_integ"]: raise ValueError("cell centres occurring more than once in the same time step")
        border_cell = False
        for cell_index in ind_triple:
            ## Append information on cell centres:
            xarr_index_ij.loc[cell,cell_index[0]*cfg_set["timestep"],:] = [cell_index[1],cell_index[2]]

            ## Get indices of domain of specific TRT cell:
            indices = get_indices_of_domain(cell_index[1],cell_index[2],cfg_set,X,Y)[0]

            ## Check whether all pixels are within ccs4 domain:
            if indices.shape == (xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],:].values).shape:
                xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],:] = indices
            else:
                if not border_cell:
                    print("   *** Warning: Domain around TRT cell %s crosses observational domain ***" % cell)
                    cell_info_df.loc[cell_info_df.index==cell,"Border_cell"] = True
                border_cell = True
                xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],range(indices.shape[0])] = indices
            if cfg_set["save_TRT_domain_map"]:
                TRTarr_plot[cell_index[0],:,:].flat[xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],:].values] *= 0
                #TRTarr_plot[cell_index[0],:,:].flat[xarr_index_flat.loc[cell,cell_index[0]*cfg_set["timestep"],:].values] += int(cell)

    ## Create Xarray file containing the TRT information and the domain-of-interest indices:
    xr_ind_ds = xr.Dataset.from_dataframe(cell_info_df)
    xr_ind_ds.rename({"index": "TRT_ID"},inplace=True)
    xr_ind_ds = xr.merge([xr_ind_ds,xarr_index_flat,xarr_index_ij])

    ## Rename ID_TRT to Date_TRT_ID, including date of reading:
    xr_ind_ds.rename({"TRT_ID": "DATE_TRT_ID"},inplace=True)
    xr_ind_ds["DATE_TRT_ID"] = np.array([cfg_set["t0"].strftime("%Y%m%d%H%M")+"_"+TRT_ID for TRT_ID in xr_ind_ds["DATE_TRT_ID"].values],
                                        dtype=np.object)

    ## Save xarray object to temporary location:
    disp_reverse_str = "" if not cfg_set["future_disp_reverse"] else "_rev"
    filename = os.path.join(cfg_set["tmp_output_path"],"%s%s%s%s" % \
                            (cfg_set["t0_str"],"_stat_pixcount",disp_reverse_str,".pkl"))
    with open(filename, "wb") as output_file: pickle.dump(xr_ind_ds, output_file, protocol=-1)

    ## Save nc file with TRT domains to disk:
    if cfg_set["save_TRT_domain_map"]:
        TRTarr_plot += 1
        TRTarr_plot = np.array(TRTarr_plot,dtype=np.float32)
        filename = os.path.join(cfg_set["tmp_output_path"],"%s%s%s%s" % \
                                (cfg_set["t0_str"],"_TRT_disp_domain",disp_reverse_str,".nc"))
        save_nc(filename,TRTarr_plot,"TRT",np.float32,"-","Domain around TRT cells",cfg_set["t0"],"",dt=5)

## Get indices based on i,j coordinate:
def get_indices_of_domain(i,j,cfg_set,X=None,Y=None):
    if cfg_set["stat_sel_form"] == "square":
        dx = int(cfg_set["stat_sel_form_width"]/2)
        bool_arr = np.full(cfg_set["xy_ext"], False, dtype=bool)
        bool_arr[:,:] = False
        bool_arr[i-dx:i+dx,j-dx:j+dx] = True
        indices = np.where(bool_arr.flat)
    elif cfg_set["stat_sel_form"] == "circle":
        interior = ((X-j)**2 + (Y-i)**2) < (cfg_set["stat_sel_form_width"]/2.)**2
        indices = np.where(interior.flat)
    return indices


## Gather statistics and pixel counts and save into xarray structure:
def append_statistics_pixcount(cfg_set_input,cfg_var,cfg_var_combi,reverse=False):
    """ Wrapper function to get:
        - Domain indices
        - Variables to read statistics and pixel counts from
        - Read the actual statistics and pixel counts (NaN and "minimum-value").
    """

    ## Change settings related calculating statistics from future or current observations:
    cfg_set = cfg_set_input.copy()
    print_reverse = "future" if reverse else "past"
    cfg_set["future_disp_reverse"] = True  if reverse else False
    cfg_set["time_change_factor"]  = -1    if reverse else 1
    print("Read statistics of %s observations related to time step: %s" %
          (print_reverse,cfg_set["t0"].strftime("%Y-%m-%d %H:%M")))
    cfg_set["verbose"] = False

    ## Read file with TRT domain indices:
    disp_reverse_str = "" if not cfg_set["future_disp_reverse"] else "_rev"
    filename = os.path.join(cfg_set["tmp_output_path"],"%s%s%s%s" % \
                            (cfg_set["t0_str"],"_stat_pixcount",disp_reverse_str,".pkl"))
    with open(filename, "rb") as output_file: xr_stat_pixcount = pickle.load(output_file)

    ## Check list of statistics (hard-coded):
    if any(cfg_set["stat_list"] != ["SUM","MEAN","STDDEV","MIN","PERC01","PERC05","PERC25",
                                    "PERC50","PERC75","PERC95","PERC99","MAX"]):
        raise NotImplementedError("Different of statistics than implemented.")
    xr_stat_pixcount["statistic"]   = cfg_set["stat_list"]
    xr_stat_pixcount["pixel_count"] = cfg_set["pixcount_list"][:2]

    ## Define file type to be read (depending on residual field correction method):
    if cfg_set["displ_TRT_cellcent"]:
        file_type = "orig"
    else:
        if cfg_set["instant_resid_corr"] or cfg_set["resid_method"]=="None":
            file_type = "disp"
        elif cfg_set["resid_method"]=="Twostep":
            file_type = "disp_resid"
        elif cfg_set["resid_method"]=="Onestep":
            file_type = "disp_resid_combi"

    ## Read in the number of "minimum-value" and NaN pixels and calculate the statistics:
    dtype_pixc, fill_value_pc = fill_value_pixelcount(cfg_set)
    for var in cfg_set["var_list"]+cfg_set["var_combi_list"]:
        calculate_statistics_pixcount(var,cfg_set,cfg_var,cfg_var_combi,file_type,xr_stat_pixcount,dtype_pixc,fill_value_pc)
        if False:
            plt.clf()
            time_fac = -cfg_set["time_change_factor"]
            plt.plot(time_fac*xr_stat_pixcount["time_delta"],np.moveaxis(xr_stat_pixcount[var+"_stat"].values[:,:,1],0,1))
            #plt.plot(-xr_stat_pixcount["time_delta"],np.moveaxis(xr_stat_pixcount[var+"_stat"].values[:,:,-3],0,1))
            plt.title("Mean of variable "+var+" ("+print_reverse+")")
            #plt.title("95% percentile of variable: "+var)
            plt.pause(1)

    ## Add minus sign to time_delta coordinate for observations in the past:
    if not reverse: xr_stat_pixcount["time_delta"] *= -1

    ## Save the respective pickle file to the temporary location on the disk:
    with open(filename, "wb") as output_file: pickle.dump(xr_stat_pixcount, output_file, protocol=-1)

    ## Save control-image if necessary:
    if cfg_set["save_stat_ctrl_imag"]:
        filename_fig = os.path.join(cfg_set["tmp_output_path"],"%s%s%s%s" % \
                                    (cfg_set["t0_str"],"_RZC_stat",disp_reverse_str,".pdf"))
        plot_var = cfg_set["var_list"][0]+"_stat" if "RZC" not in cfg_set["var_list"] else "RZC_stat"
        #""" DELETE +"_nonmin" in row below """
        #xr_stat_pixcount[plot_var+"_nonmin"][:,:,1].plot.line(x="time_delta",add_legend=False)
        xr_stat_pixcount[plot_var][:,:,1].plot.line(x="time_delta",add_legend=False)
        plt.savefig(filename_fig,format="pdf")
        plt.close()

    ## Potential parallelised version (DEPRECATED)
    #num_cores = np.max([multiprocessing.cpu_count()-2,1])
    #print("  Parallelising displacement with %s cores" % num_cores)
    #Parallel(n_jobs=num_cores)(delayed(calculate_statistics_pixcount)(var,cfg_set,
    #         cfg_var,cfg_var_combi,file_type,xr_stat_pixcount,dtype_pixc,fill_value_pc) for var in cfg_set["var_list"]+cfg_set["var_combi_list"])

## Calculate statistics and pixel counts for specific variable array:
def calculate_statistics_pixcount(var,cfg_set,cfg_var,cfg_var_combi,file_type,xr_stat_pixcount,dtype_pixc,fill_value_pc):
    """ Function reading the actual statistics and pixel counts for variable 'var'.

    Parameters
    ----------

    file_type : string
        String specifying which type of non-displaced (orig) or displaced (disp & resid or resid_combi) to be read

    xr_stat_pixcount : xarray object
        Object where into the statistics and pixels counts are written (with the information on the TRT cells
        already written into it)

    dtype_pixc : numpy.dtype object
        Data type of pixel count (if domain is small enough, a lower precision uint dtype can be chosen)

    fill_value_pc : int
        Fill value in case no pixels can be counted (e.g. in case no NaN pixels are within domain)
    """

    if var in cfg_set["var_list"]:
        ## Filter out auxiliary variables which are appended later (due to higher
        ## efficiency when creating the training dataset):
        if cfg_set["source_dict"][var]=="METADATA" and var not in ["U_OFLOW","V_OFLOW"]:
            return
        ## Change setting of file type for U_OFLOW and V_OFLOW variable:
        elif var in ["U_OFLOW","V_OFLOW"]:
            var_name = "Vx" if var=="U_OFLOW" else "Vy"
            if file_type=="orig":
                file_type_UV = "standard"
            else: file_type_UV = "resid" if file_type=="disp_resid" else "resid_combi"
            vararr   = load_file(path_creator_UV_disparr(file_type_UV,cfg_set),var_name)
        else:
            vararr  = load_file(path_creator_vararr(file_type,var,cfg_set),var)
        min_val = cfg_set["minval_dict"][var]
    elif var in cfg_set["var_combi_list"]:
        ## Get variable combination:
        vararr = get_variable_combination(var,cfg_set,cfg_var,cfg_var_combi,file_type)
        min_val = np.nan

    if cfg_set["verbose"]: print("  read statistics for "+var)

    ## Fill nan-values in COSMO_CONV fields:
    if np.any(np.isnan(vararr)) and \
       cfg_var.loc[cfg_var["VARIABLE"]==var,"SOURCE"].values=="COSMO_CONV":
        t1_inter = datetime.datetime.now()
        vararr = interpolate_COSMO_fields(vararr, method="KDTree")
        t2_inter = datetime.datetime.now()
        if var=="RELHUM_85000" and cfg_set["verbose"]:
            print("   Elapsed time for interpolating the data in %s: %s" % (var,str(t2_inter-t1_inter)))

    ## Calculate local standard deviation of specific COSMO_CONV fields:
    if cfg_var.loc[cfg_var["VARIABLE"]==var,"VARIABILITY"].values:
        t1_std = datetime.datetime.now()
        scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                           [-10+0j, 0+ 0j, +10 +0j],
                           [ -3+3j, 0+10j,  +3 +3j]])
        #plt.imshow(vararr[2,:,:]); plt.show()
        for t in range(vararr.shape[0]):
            vararr[t,:,:] = np.absolute(signal.convolve2d(vararr[t,:,:], scharr,
                                        boundary='symm', mode='same'))
        #plt.imshow(vararr[2,:,:]); plt.show()
        t2_std = datetime.datetime.now()
        if var=="POT_VORTIC_70000" and cfg_set["verbose"]:
            print("   Elapsed time for finding the local standard deviation in %s: %s" % (var,str(t2_std-t1_std)))

    ## Smooth (COSMO) fields:
    if cfg_var.loc[cfg_var["VARIABLE"]==var,"SMOOTH"].values:
        t1_smooth = datetime.datetime.now()
        #if var=="RELHUM_85000": plt.imshow(vararr[3,:,:]); plt.title(var); plt.pause(.5)
        for t in range(vararr.shape[0]): vararr[t,:,:] = ndimage.gaussian_filter(vararr[t,:,:],cfg_set["smooth_sig"])
        #if var=="RELHUM_85000": plt.imshow(vararr[3,:,:]); plt.title(var+" smooth"); plt.show() #pause(.5)
        t2_smooth = datetime.datetime.now()
        if var=="RELHUM_85000" and cfg_set["verbose"]:
            print("   Elapsed time for smoothing the fields of %s: %s" % (var,str(t2_smooth-t1_smooth)))

    ## Read in statistics and pixel counts / read in category counts:
    t1_stat = datetime.datetime.now()
    if var not in ["CMA","CT"]:
        ## Read in values at indices:
        vararr_sel = np.stack([vararr[time_point,:,:].flat[xr_stat_pixcount["TRT_domain_indices"].values[:,time_point,:]].astype(np.float32) \
                               for time_point in range(vararr.shape[0])])
        vararr_sel = np.swapaxes(vararr_sel,0,1)
        if np.any(xr_stat_pixcount["TRT_domain_indices"].values==0):
            vararr_sel[xr_stat_pixcount["TRT_domain_indices"].values==0] = np.nan

        ## Get count of nans and minimum values:
        array_pixc = np.stack([np.sum(np.isnan(vararr_sel),axis=2),
                               np.sum(vararr_sel<=min_val,axis=2)],axis=2)
        xr_stat_pixcount[var+"_pixc"] = (('DATE_TRT_ID', 'time_delta', 'pixel_count'), array_pixc.astype(np.uint16,copy=False))

        ## Calculate the actual statistics:
        perc_values = [0,1,5,25,50,75,95,99,100]
        array_stat = np.array([np.sum(vararr_sel,axis=2),  #nansum
                               np.mean(vararr_sel,axis=2), #nanmean
                               np.std(vararr_sel,axis=2)]) #nanstd
        array_stat = np.moveaxis(np.concatenate([array_stat,np.percentile(vararr_sel,perc_values,axis=2)]),0,2) #nanpercentile
        xr_stat_pixcount[var+"_stat"] = (('DATE_TRT_ID', 'time_delta', 'statistic'), array_stat.astype(np.float32,copy=False))


        ## Add specific statistics for Radar variables, only analysing values above minimum value:
        if var not in cfg_set["var_combi_list"] and cfg_set["source_dict"][var]=="RADAR":
            vararr_sel[vararr_sel<=min_val] = np.nan
            array_stat_nonmin = np.array([np.nansum(vararr_sel,axis=2),
                                          np.nanmean(vararr_sel,axis=2),
                                          np.nanstd(vararr_sel,axis=2)])
            array_stat_nonmin = np.moveaxis(np.concatenate([array_stat_nonmin,np.nanpercentile(vararr_sel,perc_values,axis=2)]),0,2)
            xr_stat_pixcount[var+"_stat_nonmin"] = (('DATE_TRT_ID', 'time_delta', 'statistic'), array_stat_nonmin.astype(np.float32,copy=False))

    else:
        ## Read in values at indices:
        vararr_sel = vararr.flat[xr_stat_pixcount["TRT_domain_indices"].values]

        ## Get count different categories:
        raise ImplementationError("Categorical counting not yet implemented")
    t2_stat = datetime.datetime.now()
    if var=="RELHUM_85000" and cfg_set["verbose"]:
        print("   Elapsed time for calculating the statistics of %s: %s" % (var,str(t2_stat-t1_stat)))

    ## Read number of pixels with max-echo value higher than 57dBZ
    if var=="CZC":
        xr_stat_pixcount[var+"_lt57dBZ"] = (('DATE_TRT_ID', 'time_delta'), np.sum(vararr_sel>57.,axis=2).astype(np.uint16,copy=False))
        #print("   Max CZC value: %s" % np.nanmax(vararr_sel))
        #print("   Number of CZC pixels > 57dBZ: %s" % np.sum(vararr_sel>57.,axis=2))

## Get variable combination as vararr array:
def get_variable_combination(var,cfg_set,cfg_var,cfg_var_combi,file_type):
    """ Read ingredients of channel/variable combination and return combination
    as variable array (according to the respective csv config file)
    """

    ## Get channel combination and take simple difference:
    if cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"OPERATION"].values=="diff":
        var_1_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_1"].values[0]
        vararr_1   = load_file(path_creator_vararr(file_type,var_1_name,cfg_set),var_1_name)

        var_2_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_2"].values[0]
        vararr_2   = load_file(path_creator_vararr(file_type,var_2_name,cfg_set),var_2_name)

        vararr_return = vararr_1 - vararr_2
    ## Get channel combination and make mixed summing-difference operation:
    elif cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"OPERATION"].values=="sum_2diff":
        var_1_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_1"].values[0]
        vararr_1   = load_file(path_creator_vararr(file_type,var_1_name,cfg_set),var_1_name)

        var_2_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_2"].values[0]
        vararr_2   = load_file(path_creator_vararr(file_type,var_2_name,cfg_set),var_2_name)

        var_3_name = cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"VARIABLE_3"].values[0]
        vararr_3   = load_file(path_creator_vararr(file_type,var_3_name,cfg_set),var_3_name)

        vararr_return = vararr_1 + vararr_2 - 2*vararr_3
    elif cfg_var_combi.loc[cfg_var_combi["VARIABLE"]==var,"OPERATION"].values=="none":
        raise ValueError("Variable/Channel combination only implemented for operations which are not 'none'")
    return(vararr_return)


## Get the fill value for pixel count variable in xarray:
def fill_value_pixelcount(cfg_set):
    if (cfg_set["stat_sel_form"]=="circle" and cfg_set["stat_sel_form_width"]>18) or \
       (cfg_set["stat_sel_form"]=="square" and cfg_set["stat_sel_form_width"]>15):
        dtype_pixc = np.uint16
        fill_value = 2**16-1
    else:
        dtype_pixc = np.uint8
        fill_value = 2**8-1
    return dtype_pixc, fill_value

## Interpolating COSMO fields:
def interpolate_COSMO_fields(vararr, method="KDTree"):
    """ Interpolating COSMO fields fills NAN holes (due to topography or singularities)

    Parameters
    ----------

    vararr : numpy array
        Array with time_delta as the first dimension of COSMO variables.

    method : string
        Either 'KDTree' (default) or 'interpol_griddata'

    """

    ## Interpolating in 3D -> Less efficient
    """
    plt.imshow(vararr[3,:,:]); plt.title(var+" nan"); plt.pause(1)
    x = np.arange(0, vararr.shape[1])
    y = np.arange(0, vararr.shape[0])
    z = np.arange(0, vararr.shape[2])
    xx, yy, zz = np.meshgrid(x, y, z)
    bool_nanarr = np.isnan(vararr)
    x_nan    = xx[bool_nanarr]
    y_nan    = yy[bool_nanarr]
    z_nan    = zz[bool_nanarr]
    x_nonnan = xx[~bool_nanarr]
    y_nonnan = yy[~bool_nanarr]
    z_nonnan = zz[~bool_nanarr]
    vararr_nonnan = vararr[~bool_nanarr]
    vararr[bool_nanarr] = interpolate.griddata((x_nonnan, y_nonnan, z_nonnan), vararr_nonnan.ravel(),
                                               (x_nan, y_nan, z_nan), method='nearest')
    t2 = datetime.datetime.now()
    print("   Elapsed time using 3D method: %s" % (str(t2-t1)))
    plt.imshow(vararr[3,:,:]); plt.title(var+" interpol"); plt.pause(1)
    """

    ## Define grid on which to interpolate
    x = np.arange(0, vararr[0,:,:].shape[1])
    y = np.arange(0, vararr[0,:,:].shape[0])
    xx, yy = np.meshgrid(x, y)
    vararr2 = vararr.copy()

    if method=="interpol_griddata":
        for t in range(vararr.shape[0]):
            ## Extract array and mask nan-values
            bool_nanarr = np.isnan(vararr[t,:,:])

            ## Get coordinates of nan and non-nan values:
            x_nan    = xx[bool_nanarr]
            y_nan    = yy[bool_nanarr]
            x_nonnan = xx[~bool_nanarr]
            y_nonnan = yy[~bool_nanarr]
            vararr_nonnan = vararr[t,~bool_nanarr]

            ## Interpolate at points with missing values
            vararr[t,bool_nanarr] = interpolate.griddata((x_nonnan, y_nonnan), vararr_nonnan.ravel(),
                                                         (x_nan, y_nan), method='nearest')

    elif method=="KDTree":
        ## Extract array and mask nan-values
        bool_nanarr_2d = np.any(np.isnan(vararr),axis=0)
        x_nan    = xx[bool_nanarr_2d]
        y_nan    = yy[bool_nanarr_2d]
        x_nonnan = xx[~bool_nanarr_2d]
        y_nonnan = yy[~bool_nanarr_2d]
        tree = spatial.cKDTree(zip(x_nonnan.ravel(), y_nonnan.ravel()), leafsize=2)
        _, inds  = tree.query(zip(x_nan.ravel(),y_nan.ravel()), k=1)

        ## Assign nearest values to nan-pixels:
        for t in range(vararr.shape[0]):
            vararr[t,bool_nanarr_2d] = vararr[t,~bool_nanarr_2d].flatten()[inds]

    else: raise ImplementationError("No other interpolation method implemented")
    return vararr

## Add auxiliary and derived variables to operational stats & pixcount dataset:
def add_auxiliary_derived_variables(cfg_set):
    """This wrapper calls those functions adding the auxiliary/static and
    derived (TRT Rank) variables to the stats & pixcount dataset"""
    print("Adding auxiliary and derived variables")
    
    ## Get xarray dataset from tmp/ directory:
    disp_reverse_str = "" if not cfg_set["future_disp_reverse"] else "_rev"
    filename = os.path.join(cfg_set["tmp_output_path"],"%s%s%s%s" % \
                            (cfg_set["t0_str"],"_stat_pixcount",disp_reverse_str,".pkl"))
    with open(filename, "rb") as output_file: ds = pickle.load(output_file)

    ## Add auxiliary and derived variables:
    ds = add_aux_static_variables(ds, cfg_set)
    ds = add_derived_variables(ds)

    ## Save Pickle:
    with open(filename, "wb") as output_file: pickle.dump(ds, output_file, protocol=-1)
    print("  Saved pickle file with added auxiliary variables")
    
## Add auxiliary variables (solar time, topography, radar frequency)
def add_aux_static_variables(ds, cfg_set):
    """This function adds static auxilary variables like solar time (sin/cos), topography information and
    the quality information based on the frequency map"""

    ## Get paths to the auxiliary datasets:
    config = configparser.RawConfigParser()
    config.read(os.path.join(cfg_set["CONFIG_PATH"],u"input_data.cfg"))
    config_aux = config["aux_data_read"]

    ## Add statistics on altitude, slope, and the alignment of the aspect vector with the flow vector:
    ## Define percentiles:
    if any(cfg_set["stat_list"] != ["SUM","MEAN","STDDEV","MIN","PERC01","PERC05","PERC25",
                                    "PERC50","PERC75","PERC95","PERC99","MAX"]):
        raise NotImplementedError("Different of statistics than implemented.")
    perc_values = [0,1,5,25,50,75,95,99,100]

    ## Check that 'TRT_domain_indices' are integer, otherwise convert to uint:
    if not np.issubdtype(ds["TRT_domain_indices"].dtype,np.integer):
        if np.max(ds.TRT_domain_indices.values) < 65535.:
            ds["TRT_domain_indices"] = ds["TRT_domain_indices"].astype(np.uint16,copy=False)
        else: ds["TRT_domain_indices"] = ds["TRT_domain_indices"].astype(np.uint32,copy=False)

    ## Check whether topography information should be added:
    alt_var_ls = {"TOPO_ALTITUDE":"Altitude","TOPO_SLOPE":"Slope","TOPO_ASPECT":"Aspect"}
    if set(alt_var_ls).issubset(cfg_set["var_list"]):
        ## Add topography information:
        ds_alt = xr.open_dataset(config_aux["path_altitude_map"])

        for alt_var in list(set(alt_var_ls).intersection(cfg_set["var_list"])):
            if cfg_set["verbose"]: print("  Get statistics of topography variable '%s'" % alt_var_ls[alt_var])
            DEM_vals = ds_alt[alt_var_ls[alt_var]].values.flat[ds.TRT_domain_indices.values]
            if alt_var == "TOPO_ASPECT":
                # Get x- and y-component of 2d direction of the aspect-vector:
                x_asp   = np.cos(DEM_vals)
                y_asp   = np.sin(DEM_vals)
                DEM_shape = DEM_vals.shape
                del(DEM_vals)

                ## Get u- and v-component of optical flow and extent to the same extent as x_asp/y_asp (to the number of
                ## pixels in domain of interest = DEM_vals.shape[2]):
                #u_oflow = np.repeat(ds.U_OFLOW_stat.sel(statistic="PERC50").values[:,:,np.newaxis],x_asp.shape[2],axis=2)
                u_oflow = ds.U_OFLOW_stat.sel(statistic="PERC50")
                #v_oflow = np.repeat(ds.U_OFLOW_stat.sel(statistic="PERC50").values[:,:,np.newaxis],x_asp.shape[2],axis=2)
                v_oflow = ds.V_OFLOW_stat.sel(statistic="PERC50")
                denominator_2 = np.sqrt(u_oflow**2+v_oflow**2)

                ## Calculate aspect-flow-alignment factor:
                DEM_vals      = np.zeros(DEM_shape)
                if cfg_set["verbose"]: print("   Looping through %s pixel of TRT domain:" % DEM_vals.shape[2])
                for pix_ind in np.arange(DEM_vals.shape[2]):
                    if pix_ind%50==0 and cfg_set["verbose"]: print("\r     Working on pixel index %s" % pix_ind)
                    numerator     = u_oflow*x_asp[:,:,pix_ind] + v_oflow*y_asp[:,:,pix_ind];   #print("     Calculated the numerator")
                    denominator_1 = np.sqrt(x_asp[:,:,pix_ind]**2+y_asp[:,:,pix_ind]**2)
                    #del(x_asp); del(y_asp)
                    #denominator_2 = np.sqrt(u_oflow**2+v_oflow**2)
                    #del(u_oflow); del(v_oflow)
                    denominator   = denominator_1*denominator_2;     #print("     Calculated the denominator")
                    #del(denominator_1)#; del(denominator_2)
                    DEM_vals[:,:,pix_ind]      = -numerator/denominator;          #print("     Calculated the Alignment")
                    #del(numerator); del(denominator)
                del(denominator, numerator, denominator_1, denominator_2, u_oflow, v_oflow, x_asp, y_asp)

            ## Calcualte the statistics:
            array_stat = np.array([np.sum(DEM_vals,axis=2),  #nansum
                                   np.mean(DEM_vals,axis=2), #nanmean
                                   np.std(DEM_vals,axis=2)]) #nanstd
            if cfg_set["verbose"]: print("   Calculated sum / mean / standard deviation")
            array_stat = np.moveaxis(np.concatenate([array_stat,np.percentile(DEM_vals,perc_values,axis=2)]),0,2)
            if cfg_set["verbose"]: print("   Calculated quantiles")

            ## Add variable to dataset:
            ds[alt_var+"_stat"] = (('DATE_TRT_ID', 'time_delta', 'statistic'), array_stat)

    ## Check whether topography information should be added:
    if "RADAR_FREQ_QUAL" in cfg_set["var_list"]:
        if cfg_set["verbose"]: print("  Get radar frequency qualitiy information")

        ## Import radar frequency map:
        from PIL import Image
        frequency_data = swissradar.convertToValue(Image.open(config_aux["path_frequency_image"]),
                                                   config_aux["path_frequency_scale"])
        frequency_data[np.logical_or((frequency_data >= 9999.0),(frequency_data <= 0.5))] = np.nan

        ## Get values in TRT domains:
        qual_vals = frequency_data.flat[ds.TRT_domain_indices.values]

        ## Calcualte the statistics:
        array_stat = np.array([np.sum(qual_vals,axis=2),  #nansum
                               np.mean(qual_vals,axis=2), #nanmean
                               np.std(qual_vals,axis=2)]) #nanstd
        if cfg_set["verbose"]: print("   Calculated sum / mean / standard deviation")
        array_stat = np.moveaxis(np.concatenate([array_stat,np.percentile(qual_vals,perc_values,axis=2)]),0,2)
        if cfg_set["verbose"]: print("   Calculated quantiles")

        ## Add variable to dataset:
        ds["RADAR_FREQ_QUAL_stat"] = (('DATE_TRT_ID', 'time_delta', 'statistic'), array_stat)

    ## Check whether solar time information should be added:
    solar_time_ls = ["SOLAR_TIME_SIN","SOLAR_TIME_COS"]
    if set(solar_time_ls).issubset(cfg_set["var_list"]):
        if cfg_set["verbose"]: print("  Get local solar time (sin & cos component)")

        ## Sin and Cos element of local solar time:
        time_points = [datetime.datetime.strptime(DATE_TRT_ID_date, "%Y%m%d%H%M") for DATE_TRT_ID_date in ds.date.values]
        lon_loc_rad = np.deg2rad(ds.lon_1_stat.sel(statistic="MEAN",time_delta=0).values)
        solar_time_sincos = [solartime(time_points_i,lon_loc_rad_i) for time_points_i,lon_loc_rad_i in zip(time_points,lon_loc_rad)]
        ds["SOLAR_TIME_SIN"] = (('DATE_TRT_ID'), np.array(solar_time_sincos)[:,0])
        ds["SOLAR_TIME_COS"] = (('DATE_TRT_ID'), np.array(solar_time_sincos)[:,1])

    return(ds)

    """
    ## Remove 'time_delta' coordinate in TRT variables (which are only available for t0):
    ds_keys  = np.array(ds.keys())
    keys_TRT = ds_keys[np.where(["stat" not in key_ele and "pixc" not in key_ele for key_ele in ds_keys])[0]]
    keys_TRT_timedelta = ["TRT_domain_indices","TRT_cellcentre_indices","CZC_lt57dBZ","pixel_indices","CHi_CHj","pixel_count"]
    for key_TRT in keys_TRT:
        if key_TRT in keys_TRT_timedelta: continue
        print(key_TRT)
        ds[key_TRT] = ds[key_TRT].sel(time_delta=0).drop("time_delta")
    """

## Calculate solar time (sin/cos) as fun of lon and UTC time
def solartime(time, lon_loc_rad, sun=ephem.Sun()):
    """Return sine and cosine value of solar hour angle depending on longitude and time of TRT cell at t0"""
    obs = ephem.Observer()
    obs.date = time; obs.lon = lon_loc_rad
    sun.compute(obs)
    ## sidereal time == ra (right ascension) is the highest point (noon)
    angle_rad  = ephem.hours(obs.sidereal_time() - sun.ra + ephem.hours('12:00')).norm
    return np.sin(angle_rad), np.cos(angle_rad)

## Read xarray files from disk, depending on file ending (as .pkl or .nc file):
def xarray_file_loader(path_str):
    import psutil
    if path_str[-3:]==".nc":
        expected_memory_need = float(os.path.getsize(path_str))/psutil.virtual_memory().available*100
        if expected_memory_need > 35:
            print("  *** Warning: File %i is opened as dask dataset (expected memory use: %02d%%) ***" %\
                  (path_number, expected_memory_need))
            xr_n = xr.open_mfdataset(path_str,chunks={"DATE_TRT_ID":1000})
        else: xr_n = xr.open_dataset(path_str)
    elif path_str[-4:]==".pkl":
        with open(path_str, "rb") as path: xr_n = pickle.load(path)
    return xr_n

## Add derived information (e.g. TRT-Rank):
def add_derived_variables(ds):
    print("Adding derived information:")

    ## Add TRT-Rank
    print("  Adding TRT Rank:")
    ds = calc_TRT_Rank(ds,ET_option="cond_median")
    
    return(ds)

## Calculate TRT Rank:
def calc_TRT_Rank(xr_stat,ET_option="cond_median"):
    """Calculate TRT Rank for square/circle with CCS4 radar data.
    The option "ET_option" states whether from the EchoTop 45dBZ values
    the conditional median (of all pixels where ET45 is non-zero),
    the median over all pixels, or the max of all pixels should be used."""

    if ET_option not in ["cond_median","all_median","all_max"]:
        raise ValueError("variable 'ET_option' has to be either 'cond_median', 'all_median', or 'all_max'")

    ## Read the variables:
    VIL_scal  = xr_stat.LZC_stat.sel(statistic=b"MAX")              ## Vertical integrated liquid (MAX)
    ME_scal   = xr_stat.CZC_stat.sel(statistic=b"MAX")              ## MaxEcho (Max)
    A55_scal  = xr_stat.CZC_lt57dBZ                                ## N pixels >57dBZ (#)

    if ET_option == "cond_median":
        ET45_scal = xr_stat.EZC45_stat_nonmin.sel(statistic=b"PERC50")  ## EchoTop 45dBZ (cond. Median)
    elif ET_option == "all_median":
        ET45_scal = xr_stat.EZC45_stat.sel(statistic=b"PERC50")         ## EchoTop 45dBZ (Median)
    elif ET_option == "all_max":
        ET45_scal = xr_stat.EZC45_stat.sel(statistic=b"MAX")            ## EchoTop 45dBZ (MAX)

    ## Scale variables to values between min and max according to Powerpoint Slide
    ## M:\lom-prod\mdr-prod\oper\adula\Innovation\6224_COALITION2\06-Presentations\2018-10-17_TRT_Workshop-DACH_MWO_hea.pptx:
    VIL_scal.values[VIL_scal.values>65.]   = 65. ## Max  VIL:       56 kg m-2
    ME_scal.values[ME_scal.values<45.]     = 45. ## Min MaxEcho:    45 dBZ
    ME_scal.values[ME_scal.values>57.]     = 57. ## Max MaxEcho:    57 dBZ
    ET45_scal.values[ET45_scal.values>10.] = 10. ## Max EchoTop:    10 km
    A55_scal.values[A55_scal.values>40.]   = 40. ## Max pix >57dBZ: 40

    ## Scale variables to values between 0 and 4:
    VIL_scal  = VIL_scal/65.*4
    ET45_scal = ET45_scal/10.*4
    ME_scal   = (ME_scal-45.)/12.*4
    A55_scal  = A55_scal/40.*4

    ## Calculate TRT rank:
    TRT_Rank = (2.*VIL_scal+2*ET45_scal+ME_scal+2.*A55_scal)/7.
    TRT_Rank = TRT_Rank.drop("statistic")
    xr_stat["TRT_Rank"] = (('DATE_TRT_ID', 'time_delta'), TRT_Rank)

    ## Calculate TRT rank difference to t0:
    TRT_Rank_diff = TRT_Rank - TRT_Rank.sel(time_delta=0)
    xr_stat["TRT_Rank_diff"] = (('DATE_TRT_ID', 'time_delta'), TRT_Rank_diff)

    return(xr_stat)
