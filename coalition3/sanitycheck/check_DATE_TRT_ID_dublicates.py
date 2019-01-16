# coding: utf-8
from __future__ import division
from __future__ import print_function
import xarray as xr
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

path_23_5min  = u"/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset_5min_16km_23km_20181211/stat_output/diam_23km/nc/Combined_stat_pixcount.nc"
path_23_30min = u"/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset_30min_16km_23km_20181128/stat_output/diam_23km/nc/Combined_stat_pixcount.nc"
path_16_5min  = u"/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset_5min_16km_23km_20181211/stat_output/diam_16km/nc/Combined_stat_pixcount.nc"
path_16_30min = u"/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset_30min_16km_23km_20181128/stat_output/diam_16km/nc/Combined_stat_pixcount.nc"
#with open(path_30min, "rb") as path: xr_1 = pickle.load(path)
#with open(path_5min, "rb") as path: xr_2 = pickle.load(path)
xr_1 = xr.open_dataset(path_23_30min)
xr_2 = xr.open_dataset(path_23_5min)
#xr_1 = xr.open_dataset(path_16_30min)
#xr_2  = xr.open_dataset(path_16_5min)


if not xr_2.keys()==xr_1.keys():
    ipt = "1"
    while (ipt!="y" and ipt != "n"): ipt = raw_input("Non-matching keys - continue? [y/n] ")
    if ipt=="y": continue
    else: sys.exit()
        
unequal_dimensions = []
print("Compare dimensions of the two datasets:")
for item in xr_1.dims:
    if xr_1[item].equals(xr_2[item]):
        print("  Dimension %s equal" % item)
    else:
        unequal_dimensions.append(item)
        print("\n  Differing dimension: %s" % item)
        print(xr_1[item],"\n")
        print(xr_2[item],"\n")
        
ipt = " " 
if len(unequal_dimensions)==1:
    print_str = "\n  Should the datasets be concatenated along dimension %s (else abort script)? [y/n] " % unequal_dimensions[0]
    while (ipt!="y" and ipt != "n"): ipt = raw_input(print_str)
    if ipt=="n": sys.exit()
    else: concat_dim = unequal_dimensions[0]
elif len(unequal_dimensions)>1:
    print("  More then one dimension found which do not agree: %s" % ', '.join(map(str, unequal_dimensions)))
    print_str = "  Along which dimension should datasets be concatenated? "
    concat_dim = "probably_no_dim_will_ever_be_called_this_way"
    while concat_dim not in unequal_dimensions: concat_dim = raw_input(print_str)
    while (ipt!="y" and ipt != "n"): ipt = raw_input("    Are you really sure what you're doing..? [y/n] ")
    if ipt=="n": sys.exit()
    
    
    
DTI_common = np.array(list(set(xr_1[concat_dim].values).intersection(set(xr_2[concat_dim].values))))
if len(DTI_common)==0:
    print("\nFound common indices, compare the entries: ")
    #ipt = " " 
    #while (ipt!="y" or ipt != "n"): ipt = raw_input("No common DATE_TRT_ID indices - continue? [y/n] ")
    #if ipt=="y": continue
    #else: sys.exit()
    
    DTI_common_rand = DTI_common[np.random.randint(len(DTI_common),size=100)]
    xr_1_common = xr_1.sel(DATE_TRT_ID = DTI_common)
    xr_2_common = xr_2.sel(DATE_TRT_ID = DTI_common)

keys = xr_2.keys()
matching_keys = {}
print("\nCompare pixc and stat keys:  ")
for key in keys:
    if ("_pixc" in key or "_stat" in key or key in xr_1_common.dims or key=="date"):
        print("   Checking key %s                                 " % key, end='\r')
        matching_keys[key] = xr_1_common[key].equals(xr_2_common[key])
print(matching_keys)

print("\nCompare TRT keys:  ")
matching_keys_TRT_corr = {}
for key in keys:
    if not ("_pixc" in key or "_stat" in key or key in xr_1_common.dims or key=="date"):
        str_print = "   Checking key %s                                 " % key
        print(str_print, end='\r')
        ind = np.where(np.logical_and(np.isfinite(xr_1_common[key].values.flatten()),
                                      np.isfinite(xr_2_common[key].values.flatten())))
        corr_coef = np.ma.corrcoef(xr_1_common[key].values.flatten()[ind],
                                   xr_2_common[key].values.flatten()[ind]/10,allow_masked=True)[0,1]
        matching_keys_TRT_corr[key] = [xr_1_common[key].equals(xr_2_common[key]/10), corr_coef]
print(matching_keys_TRT_corr)
       
ipt = "1"
while (ipt!="y" or ipt != "n"): ipt = raw_input("Are the datasets very similar (up to machine precision)? [y/n]: ") 
if ipt=="y": continue
else: sys.exit()



xr_combi = xr.open_mfdataset("nc/Combined_stat_pixcount.nc")
print("%s / %s" % (len(np.unique(xr_combi.DATE_TRT_ID)),len(xr_combi.DATE_TRT_ID)))
with open("Combined_stat_pixcount.pkl", "rb") as file: xr_combi=pickle.load(file)
print("%s / %s" % (len(np.unique(xr_combi.DATE_TRT_ID)),len(xr_combi.DATE_TRT_ID)))


DATE_TRT_ID_count = Counter(xr_combi.DATE_TRT_ID.values)
xr_double = xr_combi.sel(DATE_TRT_ID="201808071820_2018080717150128")

import matplotlib.pyplot as plt
plt.imshow(xr_double.geopotential_height_30000_stat[0,:,:]-xr_double.geopotential_height_30000_stat[1,:,:],interpolation=None)
plt.show()

DATE_TRT_ID_count_lt2 = { k:v for k, v in DATE_TRT_ID_count.items() if v > 1}
max(DATE_TRT_ID_count_lt2, key=DATE_TRT_ID_count_lt2.get)


## Get indices of unique DATE_TRT_ID values:
_, index = np.unique(xr_combi["DATE_TRT_ID"], return_index=True)
xr_combi_nondub = xr_combi.isel(DATE_TRT_ID=index)
xr_combi_nondub.to_netcdf("diam_23km/Combined_stat_pixcount_nondublicates.nc")
with open("diam_23km/Combined_stat_pixcount_nondublicates.pkl", "wb") as output_file: pickle.dump(xr_combi_nondub, output_file, protocol=-1)

