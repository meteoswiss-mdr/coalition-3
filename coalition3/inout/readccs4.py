""" [COALITION3] Reading functions for CCS4 data (Radar, SEVIR, COSMO, and THX)"""

from __future__ import division
from __future__ import print_function

import sys
import configparser
import datetime
import numpy as np
from netCDF4 import Dataset

## WARNING: These paths are hard-coded (also include in cfg file?)
#sys.path.insert(0, '/data/COALITION2/database/radar/ccs4/python')
import metranet
#sys.path.insert(0, '/opt/users/jmz/monti-pytroll/packages/mpop')
from mpop.satin import swisslightning_jmz, swisstrt, swissradar

## =============================================================================
## FUNCTIONS:

## Read COSMO NetCDF wind file and crop to ccs4:
def read_wind_nc(filename):
    """Read COSMO NetCDF file and crop to ccs4.

    Parameters
    ----------

    filenames : str
        Filepath to file to be imported.
    """

    ncfile = Dataset(filename,'r')

    ## Read wind variables from COSMO
    U_cosmo = ncfile.variables["U"][0,:,:]
    V_cosmo = ncfile.variables["V"][0,:,:]
    UV_cosmo = np.array([U_cosmo,V_cosmo])

    ## Crop from swissXXL to ccs4 format (see Uli's mail):
    crop_x1 = 40; crop_y1 = 155
    UV_cosmo = UV_cosmo[:,:,crop_x1:crop_x1+640,crop_y1:crop_y1+710]

    UV_cosmo[UV_cosmo == -99999.0] = np.nan
    #UV_cosmo_m = np.ma.masked_array(UV_cosmo)
    #UV_cosmo_m.mask = (UV_cosmo == -99999.0)
    UV_cosmo = UV_cosmo[:,:,::-1,:]

    return UV_cosmo

## Read COSMO NetCDF convection file:
def read_convection_nc(filename,var,cfg_set):
    """Read COSMO NetCDF convection file.

    Parameters
    ----------

    filenames : str
        Filepath to file to be imported.

    var : string
        Name of variable to be returned
    """

    ncfile = Dataset(filename,'r')
    #conv_vararr = np.zeros((1,)+cfg_set["xy_ext"])
    if var in ["lat_1","lon_1"]:
        conv_vararr = np.array(ncfile.variables[var][:,:])
        conv_vararr = conv_vararr[::-1,:]
        conv_vararr = np.moveaxis(np.atleast_3d(conv_vararr),2,0)
    elif "POT_VORTIC" in var or "THETAE" in var or "MCONV" in var or \
         "geopotential_height" in var or "RELHUM" in var:
        var_name = var[:-6]; pressure = var[-5:]
        if pressure == "30000" or pressure == "75000":
            pressure_ind = 0
        elif pressure == "50000" or pressure == "85000":
            pressure_ind = 1
        elif pressure == "70000":
            pressure_ind = 2
        conv_vararr = np.array(ncfile.variables[var_name][:,pressure_ind,:,:])
        conv_vararr = conv_vararr[:,::-1,:]
        if "POT_VORTIC" in var or "MCONV" in var: conv_vararr*=10e10
    else:
        conv_vararr  = np.array(ncfile.variables[var][:,:,:])
        conv_vararr  = conv_vararr[:,::-1,:]
    conv_vararr[conv_vararr == -99999.0] = np.nan
    return conv_vararr


## Read COSMO NetCDF convection file:
def read_convection_nc_old(filename,cfg_set):
    """Read COSMO NetCDF convection file.

    Parameters
    ----------

    filenames : str
        Filepath to file to be imported.
    """

    ncfile = Dataset(filename,'r')

    ## Read convection variables from COSMO
    config = configparser.RawConfigParser()
    config.read("%s/%s" % (cfg_set["CONFIG_PATH"],cfg_set["CONFIG_FILE_set"]))
    config_ds = config["variables"]
    conv_vars = config_ds["conv_vars"].split(',')

    conv_vararr = np.zeros((len(conv_vars),)+cfg_set["xy_ext"])
    i = 0
    for conv_var in conv_vars:
        conv_vararr[i,:,:] = ncfile.variables[conv_var][0,:,:]
        i += 1

    conv_vararr[conv_vararr == -99999.0] = np.nan
    #conv_vararr = np.ma.masked_array(conv_vararr)
    #conv_vararr.mask = (conv_vararr == -99999.0)

    return conv_vararr

## Read MSG3 satellite data file:
def read_sat_nc(filename,cfg_set,var):
    """Read SEVIRI NetCDF file.

    Parameters
    ----------

    filenames : str
        Filepath to file to be imported.

    var : string
        Name of satellite channel to be returned
    """

    ncfile = Dataset(filename,'r')

    ## Read satellite SEVIRI channel
    sat_vararr = ncfile.variables[var][:,:,0]
    sat_vararr[sat_vararr == 999.0] = np.nan

    ncfile.close()
    return sat_vararr

## Read lightning data file:
def read_lightning_data(var,filename,cfg_set,t_current):
    """Read THX lightning data file (Ascii .prd or NetCDF file).

    Parameters
    ----------

    var : str
        THX variable to be read from file.

    filenames : str
        Filepath to file to be imported.
    """
    config = configparser.RawConfigParser()
    config.read(os.path.join(CONFIG_PATH,u"input_data.cfg"))
    config_ds = config["light_read"]

    ## Make sure only one lightning dataset is provided:
    if len(filename)>1:
        print("*** Several filenames provided, can only work with one: ***")
        print(filename)
        print("*** Break process ***"); sys.exit()
    filename = filename[0]

    #print("      Read lightning data from .%s files" % config_ds["fn_ext"])

    ## Case if text files should be read:
    if config_ds["fn_ext"]=="prd":
        ## Assure that algorithm works also at 00:00UTC:
        if t_current.hour==0 and t_current.minute==0:
            t_current = t_current-datetime.timedelta(seconds=0.5)
        ## Read in the data:
        vararr = swisslightning_jmz.readLightning(filename,False,t_current,
                                                  cfg_set["timestep"])
        ## Get correct return:
        if var[4:] == "dens": var_ind = 1
        if var[4:] == "densIC": var_ind = 2
        if var[4:] == "densCG": var_ind = 3
        if var[4:] == "curr_abs": var_ind = 4
        if var[4:] == "curr_neg": var_ind = 5
        if var[4:] == "curr_pos": var_ind = 6
        vararr = vararr[var_ind]

    ## Case if nc files should be read:
    elif config_ds["fn_ext"]=="nc":
        ncfile = Dataset(filename,'r')
        nc_time = ncfile.variables["time"]
        if nc_time[0]!=cfg_set["timestep"]:
            print("*** Lightning data only available at %dmin time-steps ***" %
                  nc_time[0])
            sys.exit()
        if t_current.hour==0 and t_current.minute==0:
            vararr = ncfile.variables[var][-1,:,:]
        else:
            t_diff_min = int((t_current - \
                              t_current.replace(hour=00, minute=00)).seconds/60)
            t_ind = np.where(nc_time[:]==t_diff_min)[0]
            if len(t_ind)>1:
                print("ERROR: Unambigous time information for lightning data")
                sys.exit()
            elif len(t_ind)<1:
                print("ERROR: No matching lighting data found")
                sys.exit()
            else: t_ind = t_ind[0]
            vararr = ncfile.variables[var][t_ind,:,:]
        # Get it into the right shape:
        #vararr = np.moveaxis(vararr,1,0)

    ## Unfold lightning data if necessary:
    if config_ds["unfold"]=="True":
        unfold_form = config_ds["unfold_form"]
        dx_unfold   = int(config_ds["dx_unfold"])
        vararr      = swisslightning_jmz.unfold_lightning(vararr,dx_unfold,
                                                          unfold_form)
    return vararr

## Get array of a certain variable in a certain time step:
def get_vararr_t(t_current, var, cfg_set):
    """Get CCS4 variable array at timestep t_current.

    Parameters
    ----------

    t_current : datetime object
        Current time for which to calculate displacement array.

    var : string
        Name of variable to be returned

    cfg_set : dict
        Basic variables defined in input_NOSTRADAMUS_ANN.py
    """

    source = cfg_set["source_dict"][var]
    ## Implement different reader for different variable:
    if source == "RADAR":
        filenames, timestamps = path_creator(t_current, var, source, cfg_set)
        index_timestep = np.where([timestamp==t_current for timestamp in timestamps])[0][0]
        vararr = metranet.read_file(filenames[index_timestep], physic_value=True)
        #print(t_current,np.nanmax(vararr.data))
        vararr = np.moveaxis(np.atleast_3d(vararr.data),2,0)
        return vararr
    elif source == "THX":
        filenames, timestamps = path_creator(t_current, var, source, cfg_set)
        vararr = read_lightning_data(var, filenames, cfg_set, t_current)
        vararr = np.moveaxis(np.atleast_3d(vararr),2,0)#np.moveaxis(,2,1)
        return vararr
    elif source == "COSMO_WIND":
        filename, timestamps = path_creator(t_current, var, source, cfg_set)
        vararr = read_wind_nc(filename)
        plt.imshow(vararr[0,:,:,:])
        plt.show()
        sys.exit()

        return vararr
    elif source == "COSMO_CONV":
        if t_current.minute==0:
            filename, timestamps = path_creator(t_current, var, source, cfg_set)
            vararr = read_convection_nc(filename,var,cfg_set)
        else:
            filename_h_old, timestamp_h_old = path_creator(t_current, var,
                                                           source, cfg_set)
            vararr_old = read_convection_nc(filename_h_old,var,cfg_set)
            weight_old = 1-t_current.minute/60.

            t_current_plus1h = t_current + datetime.timedelta(hours=1)
            filename_h_new, timestamp_h_new = path_creator(t_current_plus1h,
                                                           var, source, cfg_set)
            vararr_new = read_convection_nc(filename_h_new,var,cfg_set)
            weight_new = 1-weight_old

            vararr = weight_old*vararr_old+weight_new*vararr_new

        ## Smooth fields if requested (DEPRICATED):
        ## COSMO fields are smoothed before reading the statistics
        return vararr
    elif source == "SEVIRI":
        filenames, timestamps = path_creator(t_current, var, source, cfg_set)
        if all(filename is None for filename in filenames):
            vararr = np.zeros((1,cfg_set["xy_ext"][0],
                               cfg_set["xy_ext"][1]))*np.nan
        else:
            vararr = read_sat_nc(filenames[0],cfg_set,var)
            vararr = np.moveaxis(np.atleast_3d(vararr),2,0)
        return vararr
    else:
        raise NotImplementedError("So far path_creator implemented \
            RADAR, SEVIRI, COSMO_Conv, and THX variables only")

## Perform smoothing of convective variables:
def smooth_conv_vararr(vararr,sigma):
    """Perform smoothing of convective variables.

    Parameters
    ----------

    vararr : numpy array
        2D numpy array with convective data which should be smoothed.

    sigma : float
        Kernel width for smoothing.
    """
    from scipy import ndimage
    vararr[0,:,:] = ndimage.gaussian_filter(vararr[0,:,:],sigma)
    return(vararr)
