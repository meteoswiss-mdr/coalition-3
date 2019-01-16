""" Testing the time difference between saving/loading .npy and .nc files:"""

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
sys.path.insert(0, '/opt/users/jmz/monti-pytroll/packages/mpop')
#import mpop
from mpop.satin import swisslightning_jmz

#sys.path.insert(0, '/opt/users/jmz/monti-pytroll/packages/mpop/mpop/satin')
#import swisslightning


from NOSTRADAMUS_1_input_prep_fun import save_nc, read_nc
    
## ===============================================================================
## TEST:

path = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/tmp/_long_disp_07_07_15/"
outpath = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/test_saving/"
testfile1 = "201507071830_THX_disp_resid_combi.npy"
testfile2 = "201507071830_RZC_disp_resid_combi.npy"
testfile3 = "201507071830_HRV_disp_resid_combi.npy"
    
tf1 = np.load(path+testfile1)
tf2 = np.load(path+testfile2)
tf3 = np.load(path+testfile3)

date = datetime.datetime(2015,07,07,18,30)
cfg_set = {"timestep": 5}

"""
## Save single files:
print("\nSave single arrays as .npy files")
t_temp = datetime.datetime.now()
np.save(outpath+"1_THX.npy",tf1)
print("Saving time for THX file: %s" % str(datetime.datetime.now()-t_temp))

t_temp = datetime.datetime.now()
np.save(outpath+"2_RZC.npy",tf2)
print("Saving time for RZC file: %s" % str(datetime.datetime.now()-t_temp))
    
t_temp = datetime.datetime.now()
np.save(outpath+"3_HRV.npy",tf3)
print("Saving time for HRV file: %s" % str(datetime.datetime.now()-t_temp))
    
print("Save single arrays as .nc files")
t_temp = datetime.datetime.now()
save_nc(outpath+"1_THX.nc",[tf1],["THX"],[np.int16],[date],
        "THX NetCDF file",cfg_set["timestep"])
print("Saving time for THX file: %s" % str(datetime.datetime.now()-t_temp))

t_temp = datetime.datetime.now()
save_nc(outpath+"2_RZC.nc",[tf2],["RZC"],[np.float32],[date],
        "RZC NetCDF file",cfg_set["timestep"])
print("Saving time for RZC file: %s" % str(datetime.datetime.now()-t_temp))
    
t_temp = datetime.datetime.now()
save_nc(outpath+"3_HRV.nc",[tf3],["HRV"],[np.float32],[date],
        "HRV NetCDF file",cfg_set["timestep"])
print("Saving time for HRV file: %s" % str(datetime.datetime.now()-t_temp))
    
## Save all files:
print("\nSave all arrays as .nc files")
t_temp = datetime.datetime.now()
save_nc(outpath+"4_THX_RZC_THX.nc",[tf1,tf2,tf3],["THX","RZC","HRV"],
        [np.int16,np.float32,np.float32],[date],"Combi NetCDF file",cfg_set["timestep"])
print("Saving time for combined file: %s\n" % str(datetime.datetime.now()-t_temp))
"""

## Load files:
print("\nLoad all arrays")
t_temp = datetime.datetime.now()
tf1_npy = np.load(outpath+"1_THX.npy")
print("Loading time for 1_THX.npy file: %s" % str(datetime.datetime.now()-t_temp))

t_temp = datetime.datetime.now()
tf2_npy = np.load(outpath+"2_RZC.npy")
print("Loading time for 2_RZC.npy file: %s" % str(datetime.datetime.now()-t_temp))

t_temp = datetime.datetime.now()
tf3_npy = np.load(outpath+"3_HRV.npy")
print("Loading time for 3_HRV.npy file: %s" % str(datetime.datetime.now()-t_temp))

t_temp = datetime.datetime.now()
tf1_nc = read_nc(outpath+"1_THX.nc","THX")
print("Loading time for 1_THX.nc file: %s" % str(datetime.datetime.now()-t_temp))

t_temp = datetime.datetime.now()
tf2_nc = read_nc(outpath+"2_RZC.nc","RZC")
print("Loading time for 2_RZC.nc file: %s" % str(datetime.datetime.now()-t_temp))
                     
if not np.array_equal(tf2_npy.flatten()[np.isfinite(tf2_npy.flatten())],
                      tf2_nc.flatten()[np.isfinite(tf2_nc.flatten())]):   #not np.testing.assert_equal(tf1_npy,tf1_nc):
    print("\n\n  But arrays are not equal!\nShape: %s .nc, %s .npy" % (tf1_npy.shape,tf1_nc.shape))
    print("   ",np.sum(np.isnan(tf2_npy)),np.sum(np.isnan(tf2_nc)))
    print("   ",np.nanmax(tf2_npy-tf2_nc),"\n\n")
    #print(np.max(tf1_npy.flatten()[np.isfinite(tf1_npy.flatten())]-
    #             tf1_nc.flatten()[np.isfinite(tf1_nc.flatten())]))

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12.5, 6.5)
    axes[0].imshow(tf2_npy[0,:,:], aspect='equal')
    axes[1].imshow(tf2_nc[0,:,:],  aspect='equal')
    plt.show()
    
t_temp = datetime.datetime.now()
tf3_nc = read_nc(outpath+"3_HRV.nc","HRV")
print("Loading time for 3_HRV.nc file: %s" % str(datetime.datetime.now()-t_temp))

        
    
    
    
    
