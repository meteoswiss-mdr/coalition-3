""" [COALITION3] Functions collecting the processed statistics and
    pixcount data (conversion from pickle to NetCDF, concatenating
    all time points covering past, and covering all time points
    covering future time steps separately, concatenating these files
    with past and future time steps, add auxiliary variable)."""

from __future__ import division
from __future__ import print_function

import os
import sys
import psutil
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import matplotlib.pylab as plt
from glob import glob

import coalition3.inout.readxr as rxr
import coalition3.inout.readconfig as cfg
import coalition3.operational.statistics as stat

## =============================================================================
## FUNCTIONS:
  
## Concatenating stat/pixcount files of different time points (either all past or all future)
def concat_stat_files(pkl_path, reverse, show_ctrl_plots=False):
    ## Read list of files which should be concatenated:
    fileending = "pixcount.pkl" if not reverse else "pixcount_rev.pkl"
    fileending_out = "pixcount_past.pkl" if not reverse else "pixcount_future.pkl"
    files = sorted(glob(os.path.join(pkl_path,"[0-9]*"+fileending)))
    print("\nStart concatenating the following files:\n   %s, ..." % \
          ', '.join([os.path.basename(file_x) for file_x in files[:3]]))
    sys.stdout.flush()
    
    ## Initate array amongst which the dimensions are saved to check that no entries go missin/appear double:
    #datasets = [read_pickle_file(path) for path in files]
    #datasets = [read_nc_file(path) for path in files]
    #ls_datasets = []
    ls_DATE_TRT_ID = np.array([])

    ## Choose dimension along which the files should be concatenated:
    combined_ds = read_pickle_file(files[0])
    print("These are the dimensions of the xarray object:\n  %s" % combined_ds.dims)
    #print(combined_ds.info())
    dim = raw_input("Along which dimension should the objects be concatenated (e.g. 'DATE_TRT_ID'): ")
    while dim not in combined_ds.dims: dim = raw_input("Please choose out of the list above: ")
    files_KeyError = []
    
    ## Loop over all single files to be concatenated:
    print("Start reading in the data")
    for n, path in enumerate(files[1:],2):
        perc_complete = 100*int(n)/float(len(files))
        str_print = "  Completed %02d%% (working on file: %s)" % (perc_complete,os.path.basename(path))
        print('\r',str_print,end='') #, end='\r'
        sys.stdout.flush()
        #ls_datasets.append(read_nc_file(path))
        #new_ds = read_nc_file(path)
        new_ds = read_pickle_file(path)
        try:
            combined_ds = xr.concat([combined_ds,new_ds], dim)
        except KeyError:
            files_KeyError.append(new_ds.DATE_TRT_ID.values[0][:12])
            print("\n   *** Warning: File %s could not be appended due to KeyError\n" % os.path.basename(path))
        #ls_DATE_TRT_ID = np.append(ls_DATE_TRT_ID,ls_datasets[-1]["DATE_TRT_ID"].values)
        ls_DATE_TRT_ID = np.append(ls_DATE_TRT_ID,new_ds[dim].values)
        if len(np.unique(ls_DATE_TRT_ID))!=len(ls_DATE_TRT_ID):
            print("\n   *** Warning: Dimension lengths start to diverge ***\n")
            pdb.set_trace()
        if n%100==0:
            memoryUse = psutil.virtual_memory()
            memoryUse_MB = memoryUse.used >> 20; memoryUse_perc = memoryUse.percent
            print('\n   Current memory use: %sMB (%s%%)' % (memoryUse_MB,memoryUse_perc))
        if show_ctrl_plots and n%100==0:
            print(combined_ds)
            variable_rand = np.random.choice(np.array(["RZC_stat","IR_108_stat","THX_densCG_stat","CD2_stat","POT_VORTIC_30000_stat"]))
            statistic_rand = np.random.choice(np.array(["MEAN","PERC05","MAX","SUM","PERC99"]))
            #ls_datasets[-1][variable_rand].sel(statistic=statistic_rand).plot.line(x="time_delta", hue="DATE_TRT_ID", add_legend=False, alpha=0.6)
            new_ds[variable_rand].sel(statistic=statistic_rand).plot.line(x="time_delta", hue="DATE_TRT_ID", add_legend=False, alpha=0.6)
            plt.pause(1)
            plt.clf()
    plt.close(); print("\n")
    #print("\nStart concatenating the data along the dimension '%s'" % dim)
    #combined_ds = xr.concat(ls_datasets, dim)
    
    ## Check files which could not be appended due to KeyError:
    if len(files_KeyError)>0:
        print("Files which could not be appended:\n  %s" % files_KeyError)
        pdb.set_trace()
    
    ## Save Pickle:
    pkl_pathfile = os.path.join(pkl_path,"Combined_stat_"+fileending_out)
    with open(pkl_pathfile, "wb") as output_file: pickle.dump(combined_ds, output_file, protocol=-1)
    if os.path.exists(pkl_pathfile): print("Pickle file has been created in %s" % pkl_pathfile)
    else: raise FileNotFoundError("Pickle file was not created")

    ## Save NetCDF:
    nc_path = os.path.join(pkl_path,"nc/")
    if not os.path.exists(nc_path): os.makedirs(nc_path)
    
    nc_pathfile = os.path.join(nc_path,"Combined_stat_"+fileending_out[:-4]+".nc")
    combined_ds.to_netcdf(nc_pathfile)
    if os.path.exists(nc_pathfile): print("NetCDF file has been created in %s" % nc_pathfile)
    else: raise FileNotFoundError("NetCDF file was not created")

## Convert pickle files containing statistics and pixel counts to NetCDF files:
def convert_stat_files(path):
    bool_convert = raw_input("\nShould pickle files be converted to NetCDF? [y/n] ")
    while (bool_convert!="n" and bool_convert!="y"): bool_convert = raw_input("Should pickle files be converted to NetCDF? [y/n] ")
    
    if bool_convert=="n":
        print("  Files are not converted\n")
        return
    else:
        files = sorted(glob(os.path.join(path,"*.pkl")))
        print("  Files to be converted: \n    %s, ..." % ', '.join([os.path.basename(file_x) for file_x in files[:3]]))
        print("  Number of files to convert: %s" % len(files))
        nc_path = os.path.join(path,"nc/")
        if not os.path.exists(nc_path): os.makedirs(nc_path)
        
        for n, pkl_file in enumerate(files,1):
            perc_complete = 100*int(n)/float(len(files))
            str_print = "  Completed %02d%% (working on file: %s)" % (perc_complete,os.path.basename(pkl_file))
            print('\r',str_print,end='') #, end='\r'
            sys.stdout.flush()
            pickle2nc(pkl_file,nc_path)
        print("\nFinished converting NetCDF files to\n  %s\n" % nc_path)

## Concatenate the concatenations of past and future statistics into one big file
def concat_future_past_concat_stat_files(pkl_path):
    print("\nConcatenate the concatenations of past and future statistics into one big file")
    import psutil
    
    ## Read file with future (t0 + n min) statistics:
    file_future = os.path.join(pkl_path,"Combined_stat_pixcount_future.pkl")
    expected_memory_need = float(os.path.getsize(file_future))/psutil.virtual_memory().available*100
    if expected_memory_need > 20:
        print("  *** Warning: File 'Combined_stat_pixcount_future.pkl' is expected ***\n"+\
              "      to use %02d%% of the memory, thus open NetCDF version as mfdataset" % \
              (expected_memory_need))
        file_future = os.path.join(pkl_path,"nc/Combined_stat_pixcount_future.nc")
        xr_future = xr.open_mfdataset(file_future,chunks={"DATE_TRT_ID":1000})
    else:
        with open(file_future, "rb") as path: xr_future = pickle.load(path)
    
    ## Read file with past (t0 - n min) statistics:
    if expected_memory_need > 20:
        file_past = os.path.join(pkl_path,"nc/Combined_stat_pixcount_past.nc")
        xr_past = xr.open_mfdataset(file_past,chunks={"DATE_TRT_ID":1000})
    else:
        file_past = os.path.join(pkl_path,"Combined_stat_pixcount_past.pkl")
        with open(file_past, "rb") as path: xr_past = pickle.load(path)
    xr_past = xr_past.where(xr_past["time_delta"]<0,drop=True)

    ## Drop DATE_TRT_ID which do not agreee:
    DTI_unique = np.intersect1d(xr_past["DATE_TRT_ID"].values,
                                xr_future["DATE_TRT_ID"].values,
                                assume_unique=True)
    xr_past = xr_past.sel(DATE_TRT_ID=DTI_unique)
    xr_future = xr_future.sel(DATE_TRT_ID=DTI_unique)

    ## Concatenate to one file (first bringing 'time_delta' in xr_past in ascending order):
    xr_new = xr.concat([xr_past.sortby("time_delta"),xr_future],"time_delta")
    del(xr_past); del(xr_future)

    ## Remove 'time_delta' coordinate in TRT variables (which are only available for t0):
    ds_keys  = np.array(xr_new.keys())
    keys_TRT = ds_keys[np.where(["stat" not in key_ele and "pixc" not in key_ele for key_ele in ds_keys])[0]]
    keys_TRT_timedelta = xr_new.coords.keys()+["TRT_domain_indices","TRT_cellcentre_indices","CZC_lt57dBZ"]
    keys_TRT = list(set(keys_TRT) - set(keys_TRT_timedelta))
    for key_TRT in keys_TRT:
        if key_TRT in keys_TRT_timedelta: continue
        xr_new[key_TRT] = xr_new[key_TRT].sel(time_delta=0).drop("time_delta")
    
    ## Change all stat-variables to float32:
    keys_stat = ds_keys[np.where(["_stat" in key_ele for key_ele in ds_keys])[0]]
    for key_stat in keys_stat: xr_new[key_stat] = xr_new[key_stat].astype(np.float32)
    
    ## Change all pixc-variables to uint16:
    dtype_pixc = np.uint16 if xr_new.TRT_domain_indices.shape[2]<2**16-1 else np.uint32
    keys_pixc = ds_keys[np.where(["_pixc" in key_ele for key_ele in ds_keys])[0]]
    for key_pixc in keys_pixc: xr_new[key_pixc] = xr_new[key_pixc].astype(dtype_pixc)
    xr_new["CZC_lt57dBZ"] = xr_new["CZC_lt57dBZ"].astype(dtype_pixc)
    xr_new["TRT_cellcentre_indices"] = xr_new["TRT_cellcentre_indices"].astype(np.uint32)
    
    ## In case the dates have bin concatenated, only keep the twelve initial characters (Date+Time):
    #if len(xr_new["date"].values[0])>12:
    #    xr_new["date"].values = np.array([date_i[:12] for date_i in xr_new["date"].values])
    
    ## Save NetCDF:
    file_new = os.path.join(pkl_path,"nc/Combined_stat_pixcount.nc")
    xr_new.to_netcdf(file_new)
    
    ## Save Pickle:
    file_new = os.path.join(pkl_path,"Combined_stat_pixcount.pkl")
    with open(file_new, "wb") as output_file: pickle.dump(xr_new, output_file, protocol=-1)
    
    del(xr_new)

## Wrapper function for adding additional derived variables to dataset (in training dataset creation environment):    
def wrapper_fun_add_derived_variables(pkl_path):
    file_path = os.path.join(pkl_path,"Combined_stat_pixcount_aux.pkl")
    print(" Adding derived variables to xarray object in file:\n   %s" % file_path)
    xr_stat = rxr.xarray_file_loader(file_path)

    ## Add TRT-Rank
    xr_stat = stat.add_derived_variables(xr_stat)

    ## Save Pickle:
    file_new = os.path.join(pkl_path,"Combined_stat_pixcount.pkl")
    with open(file_new, "wb") as output_file: pickle.dump(xr_stat, output_file, protocol=-1)
    print("  Saved to pickle file.")

    ## Save NetCDF:
    file_new = os.path.join(pkl_path,"nc/Combined_stat_pixcount_auxder.nc")
    xr_stat.to_netcdf(file_new)
    print("  Saved to NetCDF file.")

    
## Wrapper function for adding additional auxiliary static variables and TRT Rank
## to dataset (in training dataset creation environment):    
def wrapper_fun_add_aux_static_variables(pkl_path):
    cfg_set_input, cfg_var, cfg_var_combi = cfg.get_config_info_op()
    file_path = os.path.join(pkl_path,"Combined_stat_pixcount.pkl")
    print(" Adding auxiliary variables and TRT Rank to xarray object in file:\n   %s" % file_path)
    cfg_set_input["verbose"] = True
    
    ## Add auxilirary variables:
    #with open(file_path, "rb") as path: ds = pickle.load(path)
    #ds = xr.open_dataset(file_path)
    ds = rxr.xarray_file_loader(file_path)
    ds = stat.add_aux_static_variables(ds, cfg_set_input)
    ds = stat.add_derived_variables(ds)
    
    ## Save Pickle:
    with open(file_path, "wb") as output_file: pickle.dump(ds, output_file, protocol=-1)
    print(" Saved pickle file with added auxiliary variables")
    
    ## Save NetCDF:
    file_new = os.path.join(pkl_path,"nc/Combined_stat_pixcount_auxder.nc")
    ds.to_netcdf(file_new)
    print(" Saved NetCDF file with added auxiliary variables")
    
def pickle2nc(pkl_pathfile,nc_path):
    nc_filename = os.path.join(nc_path,os.path.basename(pkl_pathfile)[:-3]+"nc")
    if not os.path.exists(nc_filename):
        try:
            with open(pkl_pathfile, "rb") as path: xr_stat_pixcount = pickle.load(path)
        except EOFError:
            print("\n   *** Warning: File could not be converted due to EOF Error - skip this file *** \n        %s" % os.path.basename(pkl_pathfile))
        else:
            xr_stat_pixcount.date.values = np.array([date_str[:12] for date_str in xr_stat_pixcount.date.values],dtype=np.object)
            xr_stat_pixcount.to_netcdf(nc_filename)
            path.close()
    
def read_pickle_file(filename):
    try:
        with open(filename, "rb") as path: xr_stat_pixcount = pickle.load(path)
    except EOFError:
        print("\n   *** Warning: File could not be converted due to EOF Error - skip this file *** \n        %s" % os.path.basename(pkl_pathfile))
    else:
        return xr_stat_pixcount
        
def read_nc_file(filename):
    xr_stat_pixcount = xr.open_dataset(filename,autoclose=True) #,decode_cf=False,autoclose=True,chunks={}
    return xr_stat_pixcount
  
def collection_of_plotting_functions(pkl_path):
    file_path = os.path.join(pkl_path,"Combined_stat_pixcount.pkl")
    with open(file_path, "rb") as path: xr_new = pickle.load(path)
    
    ## RZC Time Series (can be replaced by other statistics/variables):
    xr_new["RZC_stat"].sel(statistic="MEAN").plot.line(x="time_delta", hue="DATE_TRT_ID", add_legend=False, alpha=0.1)
    plt.xlim([-45,45])
    plt.show()
    
    ## Strong Lightning event:
    xr_new["THX_dens_stat"].sel(statistic="SUM").where(xr_new["THX_dens_stat"].sel(statistic="SUM")>500,drop=True)
    THX_max_lat = xr_new["lat"].sel(DATE_TRT_ID="201807201655_2018072016000184").values
    THX_max_lon = xr_new["lon"].sel(DATE_TRT_ID="201807201655_2018072016000184").values
    print("Location of maximum lightning event: %s / %s" % (THX_max_lat[0],THX_max_lon[0]))
    
    ## Plot how well TRT and NOSTRADAMUS information agrees:
    ind_nan = np.isnan(xr_new["LZC_stat"].sel(statistic="PERC95",time_delta=-5)) | np.isnan(xr_new["RZC_stat"].sel(statistic="PERC95",time_delta=-5))
    plt.plot(xr_new["LZC_stat"].sel(statistic="PERC95",time_delta=-5).values[np.where(~ind_nan)],xr_new["RZC_stat"].sel(statistic="PERC95",time_delta=-5).values[np.where(~ind_nan)],'x')
    corr_val = np.corrcoef(xr_new["LZC_stat"].sel(statistic="PERC95",time_delta=-5).values[np.where(~ind_nan)],xr_new["RZC_stat"].sel(statistic="PERC95",time_delta=-5).values[np.where(~ind_nan)])
   
    print("Correlation between VIL and Rain Rate: %s" % corr_val[0,1])
    plt.show()

    ind_nan = np.isnan(xr_new["LZC_stat"].sel(statistic="PERC95")) | np.isnan(xr_new["RZC_stat"].sel(statistic="PERC95"))
    np.corrcoef(xr_new["LZC_stat"].sel(statistic="PERC95").values[np.where(~ind_nan)],xr_new["RZC_stat"].sel(statistic="PERC95").values[np.where(~ind_nan)])

    plt.plot([0,0],[16,16],'k-')
    plt.plot(xr_new["EZC45_stat"].sel(statistic="MAX",time_delta=0),xr_new["ET45"],'x',alpha=0.8)
    plt.xlabel("Circle")
    plt.ylabel("TRT-Cell")
    plt.title("Median EchoTop 45dBZ")
    perc_equal = np.sum((xr_new["EZC45_stat"].sel(statistic="MAX",time_delta=0)==xr_new["ET45"]).values)/len(xr_new["ET45"])*100.
    print(len(xr_new["ET45"]), len(xr_new["EZC45_stat"].sel(statistic="MAX",time_delta=0)))
    plt_text = "%02d%% equal" % perc_equal
    plt.text(3,10,plt_text)
    plt.show()


    plt.plot([0,0],[16,16],'k-')
    plt.plot(xr_new["EZC15_stat"].sel(statistic="MAX",time_delta=0),xr_new["ET15"],'x',alpha=0.8)
    plt.xlabel("Circle")
    plt.ylabel("TRT-Cell")
    plt.title("Median EchoTop 15dBZ")
    plt.xlim([4,20])
    plt.ylim([4,20])
    perc_equal = np.sum((xr_new["EZC15_stat"].sel(statistic="MAX",time_delta=0)==xr_new["ET15"]).values)/len(xr_new["ET15"])*100.
    plt_text = "%02d%% equal" % perc_equal
    plt.text(8,14,plt_text)
    plt.show()

## Exploratory data analysis
def EDA_wrapper(pkl_path, cfg_set):
    file_path = os.path.join(pkl_path,"Combined_stat_pixcount.pkl")
    print("Perform exploratory data analysis on the dataset:\n   %s" % file_path)
    





