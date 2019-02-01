""" [COALITION3] With this script, two datasets can be merged or concatenated,
    where the first means that new variables are added (e.g. Nowcasting-SAF
    products etc.), whereas the latter refers to adding for example new
    observations, but of the same variables. Several safety checks are per-
    formed as well"""

from __future__ import division
from __future__ import print_function

import os
import sys
import psutil
import pickle
import xarray as xr
import numpy as np
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
sys.stdout.flush()

def file_path_reader(path_number):
    path_str = "this_is_an_unrealistic_path_name"
    while not os.path.exists(path_str):
        path_str = raw_input("  Please provide path to file %i: " % path_number)
        if not os.path.exists(path_str): print("  No such path found!")
    return path_str

def xarray_file_loader(path_str,path_number):
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

def calculate_statistics(xr_1, xr_2):
    ind = np.where(np.logical_and(np.isfinite(xr_1_common[key].values.flatten()),
                                  np.isfinite(xr_2_common[key].values.flatten())))
    corr_coef = np.ma.corrcoef(xr_1_common[key].values.flatten()[ind],
                               xr_2_common[key].values.flatten()[ind],allow_masked=True)[0,1].round(4)
    mean_diff = np.mean(xr_1_common[key].values.flatten()[ind] -
                        xr_2_common[key].values.flatten()[ind]).round(4)
    mean_fac = np.nanmean(xr_1_common[key].values.flatten()[ind]/
                          xr_2_common[key].values.flatten()[ind]).round(4)
    return([corr_coef, mean_diff, mean_fac])

def dropping_indices(xr_pre_drop, xr_file_2, DTI_common, concat_dim):
    xr_drop = xr_pre_drop.drop(DTI_common,dim=concat_dim)
    n_matches = len(np.array(list(set(xr_drop[concat_dim].values).intersection(set(xr_file_2[concat_dim].values)))))
    if xr_pre_drop[concat_dim].shape[0]-xr_drop[concat_dim].shape[0]==len(DTI_common) and n_matches==0:
        print("  Length of index %s after dropping:  %i" % (concat_dim,xr_drop[concat_dim].shape[0]))
    elif xr_pre_drop[concat_dim].shape[0]-xr_drop[concat_dim].shape[0]!=len(DTI_common):
        print("   *** Dropping error: difference in dimension length not according to length of matching indices ***")
        sys.exit()
    elif n_matches==0:
        print("   *** Dropping error: there exist still common indices ***")
        sys.exit()
    else:
        print("   *** Dropping error ***")
        sys.exit()
    return xr_drop

def print_title(title_str):
    print("\n------------------------------------------------------------------------------")
    print(title_str+"\n")

print("\n==============================================================================")
print(" Merging / Concatenating two training datasets")
print(" ---------------------------------------------")

## Decide between merging and concatenation:
print_text = """
Do you want to merge or concatenate two datasets?
  Concatenating -> Concatenating two training datasets with the same variables (e.g. sampling summers
                   2018 and 2019) along one dimension (e.g. DATE_TRT_ID).
                   In case there is an overlap in the dimension which is concatenated (e.g.
                   several times the same DATE_TRT_ID), the overlap in this dimension is
                   deleted in one dataset!
  Merging       -> Merging two training datasets with the same dimensions but different
                   variables (e.g. append newly created NWC-SAF statistics).
                   Since dimensions must agree, it is necessary to define the type of join
                   (inner -> suggested, outer, ..) which should be performed. In case of an
                   inner join, only the dimension ranges are kept which occur in both datasets.
"""
print(print_text)
combi_type = None
while (combi_type!="m" and combi_type != "c"): combi_type = raw_input("Merge or Concatenation? [m/c] ")

## 1) Reading the paths to the files:
print_title("Paths to the respective xarray datasets:")
path_str_1 = file_path_reader(1)
path_str_2 = file_path_reader(2)

## 2) Reading the files:
print_title("Loading the xarray datasets:")
xr_1 = xarray_file_loader(path_str_1,1)
xr_2 = xarray_file_loader(path_str_2,2)
print("  Finished loading the datasets")

## 3) Compare dimensions of the datasets:
print_title("Comparing the dimensions of the datasets:")
unequal_dimensions = []
#print("Compare dimensions of the two datasets:")
for item in xr_1.dims:
    if xr_1[item].equals(xr_2[item]):
        print("  Dimension %s equal" % item)
    else:
        unequal_dimensions.append(item)
        print("\n  Warning: Differing dimension: %s" % item)
        print(xr_1[item],"\n")
        print(xr_2[item],"\n")

## 4) Define dimensions along which to concatenate the datasets (most likely 'DATE_TRT_ID'):
ipt = " "
if combi_type=="c":
    print_title("Define dimension along which to concatenate the datasets (most likely 'DATE_TRT_ID'):")
    if len(unequal_dimensions)==1:
        print_str = "  Should the datasets be concatenated along dimension %s (else abort script)? [y/n] " % unequal_dimensions[0]
        while (ipt!="y" and ipt != "n"): ipt = raw_input(print_str)
        if ipt=="n": sys.exit()
        else: concat_dim = unequal_dimensions[0]
    elif len(unequal_dimensions)>1:
        print("  More then one dimension found which do not agree: %s" % ', '.join(map(str, unequal_dimensions)))
        print_str = "  Along which dimension should datasets be concatenated? "
        concat_dim = "probably_no_dim_will_ever_be_called_this_way"
        while concat_dim not in unequal_dimensions: concat_dim = raw_input(print_str)
        print_str = "    Are you really sure what you're doing (it will probably not work)..? [y/n] "
        while (ipt!="y" and ipt != "n"): ipt = raw_input(print_str)
        if ipt=="n": sys.exit()
elif combi_type=="m":
    join_opts = ['inner','outer','left','right','exact']; join_opt = ' '
    if len(unequal_dimensions)==0:
        print_title("\nAll dimensions agree, no need to cut dimensions")
        concat_dim = np.array(xr_1.dims,dtype=np.object)[0]
    else:
        print_title("Several dimensions do not agree: %s" % ', '.join(map(str, unequal_dimensions)))
        print_str = "  How do you want to join the datasets (%s)? " % ', '.join(map(str, join_opts))
        while (join_opt not in join_opts): join_opt = raw_input(print_str)
        concat_dim = np.array(xr_1.dims,dtype=np.object)[:]

## 5) Check for common indices (e.g. same 'DATE_TRT_ID'):
dict_concat_common = {}
if combi_type=="c":
    print_title("Check for common indices along dimension to concatenate (e.g. same 'DATE_TRT_IDs'):")
    DTI_common = np.array(list(set(xr_1[concat_dim].values).intersection(set(xr_2[concat_dim].values))))
    dict_concat_common[concat_dim] = DTI_common
    do_comparison = len(DTI_common)>0
elif combi_type=="m":
    print_title("Check for common indices along dimensions:")
    for dim in concat_dim:
        dict_concat_common[dim] = np.array(sorted(list(set(xr_1[dim].values).intersection(set(xr_2[dim].values)))))
    do_comparison = all([len(value)>0 for value in dict_concat_common.values()])

if do_comparison:
    print("  Found common indices, comparing the overlapping (and hopefully equal) data entries")
    xr_1_common = xr_1.loc[dict_concat_common]
    xr_2_common = xr_2.loc[dict_concat_common]

    ## 5.a) Comparing non-TRT keys:
    keys_1 = xr_1.keys()
    keys_2 = xr_2.keys()
    keys_common = np.array(list(set(keys_1).intersection(set(keys_2))))
    matching_keys = {}
    print("\n  Compare pixc and stat keys (This can take a while):  ")

    for key in keys_common:
        if ("_pixc" in key or "_stat" in key or key in xr_1_common.dims or key=="date"):
            print("   Checking key %s                                 " % key, end='\r'); sleep(0.01)
            match = xr_1_common[key].equals(xr_2_common[key])
            if not match:
                matching_keys[key] = [match]+calculate_statistics(xr_1_common, xr_2_common)
    matching_keys = pd.DataFrame.from_dict(matching_keys, orient='index',
                                           columns=["Match","Correlation","Mean difference","Mean scaling factor"])
    print(matching_keys)

    ## 5.b) Comparing TRT keys (and other weird ones):
    print("\n  Compare TRT keys (This can take a while): ")
    matching_keys_TRT_corr = {}
    for key in keys_common:
        if not ("_pixc" in key or "_stat" in key or key in xr_1_common.dims or key=="date"):
            str_print = "   Checking key %s                                 " % key
            print(str_print, end='\r'); sleep(0.01)
            match = xr_1_common[key].equals(xr_2_common[key])
            if not match:
                matching_keys_TRT_corr[key] = [match]+calculate_statistics(xr_1_common, xr_2_common)
    matching_keys_TRT_corr = pd.DataFrame.from_dict(matching_keys_TRT_corr, orient='index',
                                                    columns=["Match","Corr","Mean diff [file1-file2]","Mean factor [file1/file2]"])
    print(matching_keys_TRT_corr)

    ## 5.c) Make decision whether to proceed (is data overlap similar enough?):
    ipt = "1"
    while (ipt!="y" and ipt != "n"): ipt = raw_input("\nAre the datasets very similar (up to machine precision)? Else abort. [y/n] ")
    if ipt=="n": sys.exit()

    ## 5.d1) Drop indices in one dataset if concatenating:
    if combi_type=="c":
        ipt = "1"; drop_file = "x"
        print_str = "\nShould %i common index values deleted? Else abort. [y/n]: " % len(DTI_common)
        while (ipt!="y" and ipt != "n"): ipt = raw_input(print_str)
        if ipt=="n": sys.exit(print_str)
        else:
            while (drop_file not in ["1","2"]): drop_file = raw_input("  Whose files index values should be dropped? [1/2] ")
            print("  Length of index %s before dropping: %i" % (concat_dim,xr_1[concat_dim].shape[0]))
            if drop_file=="1": xr_1 = dropping_indices(xr_1, xr_2, DTI_common, concat_dim)
            else: xr_2 = dropping_indices(xr_2, xr_1, DTI_common, concat_dim)

    ## 5.d2) Drop variables in one of the datasets when merging:
    if combi_type=="m":
        drop_file = "x"
        while (drop_file not in ["1","2"]): drop_file = raw_input("  Whose files common variables should be dropped? [1/2] ")
        if drop_file=="1": xr_1 = xr_1.drop(list(set(keys_common)-set(xr_1.dims)))
        else: xr_2 = xr_2.drop(list(set(keys_common)-set(xr_2.dims)))
else:
    print("  Found no common indices.")
    if combi_type=="m": sys.exit()

## 6. Concatenate along chosen dimension:
if combi_type=="c":
    print_title("Concatenating along dimension %s (this will take a while):" % concat_dim)
    xr_new = xr.concat([xr_1,xr_2],concat_dim)
    print(xr_new)
elif combi_type=="m":
    print_title("Merging the two datasets")
    xr_new = xr.merge([xr_1,xr_2],join=join_opt)
    print(xr_new)

print_title("Saving the concatenated/merged file:")
path_out_nc  = raw_input("  Please provide path (incl. filename & suffix) where to write the '.nc' file: ")
same_path = raw_input("  Should the same path be used for the pickle file? [y/n] ")=="y"
if same_path:
    path_out_pkl = path_out_nc[:-3]+".pkl"
else: path_out_pkl = raw_input("  Please provide path (incl. filename & suffix) where to write the '.pkl' file: ")
print("   Saving the '.nc' file (this will take a while)", end="")
xr_new.to_netcdf(path_out_nc); print(" -> finished")
print("   Saving the '.pkl' file (this will take a while)", end="")
with open(path_out_pkl, "wb") as output_file: pickle.dump(xr_new, output_file, protocol=-1)
print(" -> finished")
sys.exit()



"""
xr_new = xr.concat([xr_5min,xr_30min], "DATE_TRT_ID")
del(xr_30min,xr_5min)
xr_new = xr_new.sortby("DATE_TRT_ID")

ds_keys  = np.array(xr_new.keys())
keys_stat = ds_keys[np.where(["_stat" in key_ele for key_ele in ds_keys])[0]]
for key_stat in keys_stat: xr_new[key_stat] = xr_new[key_stat].astype(np.float32)

dtype_pixc = np.uint16 if xr_new.TRT_domain_indices.shape[2]<2**16-1 else np.uint32
keys_pixc = ds_keys[np.where(["_pixc" in key_ele for key_ele in ds_keys])[0]]
for key_pixc in keys_pixc: xr_new[key_pixc] = xr_new[key_pixc].astype(dtype_pixc)
xr_new["CZC_lt57dBZ"] = xr_new["CZC_lt57dBZ"].astype(dtype_pixc)
xr_new["TRT_cellcentre_indices"] = xr_new["TRT_cellcentre_indices"].astype(np.uint32)

with open("Combined_stat_pixcount.pkl", "wb") as output_file: pickle.dump(xr_new, output_file, protocol=-1)
xr_new.to_netcdf("Combined_stat_pixcount.nc")
"""
