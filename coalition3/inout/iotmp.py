""" [COALITION3] Writing and reading temporary CCS4 and TRT data to disk before calculating statistics"""

from __future__ import division
from __future__ import print_function

import numpy as np
import datetime
from netCDF4 import Dataset, num2date, date2num
from os.path import dirname, exists
from os import makedirs

## =============================================================================
## FUNCTIONS:
    
## Save variable array (vararr) or displacement array (disparr) as Numpy or NetCDF file:
def save_file(output_file_path_str, data_arr,
              var_name=None, t0_datetime=None, filetype=None,
              cfg_set=None, var_type=None, var_unit=None, longname=None, dt=None):
    """Save variable and displacement arrays into files.

    Parameters
    ----------
        
    output_file_path_str : str
        File path defining location of .nc file to be written.
        
    data_arr : (list of) numpy arrays
        (List of) numpy arrays containing the data to be saved.
        
    var_name : (list of) strings
        (List of) strings stating the names of the datasets.
        
    var_type : (list of) numpy datatypes (e.g. <type 'numpy.int8'>)
        (List of) numpy datatypes stating the datatype of the datasets.
        If only one type is provided, it is assumed to apply for all arrays.
        
    var_unit : (list of) units related to the respective array data.
        If only one type is provided, it is assumed to apply for all arrays.
        
    longname : (list of) describing longnames related to the respective array data.
        If only one type is provided, it is assumed to apply for all arrays.
        
    t0_datetime : (list of) datetime objects
        (List of) datetime objects stating the time steps of the data
        (along the first dimension, starting with the most current observation).
        
    dt : int
        Number of minutes between different layers in first dimension of vararr
    """
    
    if filetype!="npy":
        ## Check arguments needed when potentially creating a NetCDF file.
        if (t0_datetime is None or filetype is None or
            cfg_set is None or var_type is None) and cfg_set is None:
            print("either the non-compulsory arguments are provided or "+
                   "a cfg_set dictionary has to be provided")
        ## Set filetype to the one stated in cfg_set (if not provided)
        if filetype is None: filetype = cfg_set["save_type"]

    ## Save numpy file (npy/npz):
    if filetype == "npy":
        if "disparr" not in output_file_path_str and type(data_arr) is not list:
            np.save(output_file_path_str, data_arr)
        elif "disparr" in output_file_path_str and len(data_arr)==4:
            output_file_path_str = output_file_path_str[:-1]+"z"
            if var_name!=["Dx","Dy","Vx","Vy"]: raise ValueError('Ordering must be "Dx","Dy","Vx","Vy"')
            np.savez(output_file_path_str, Dx=data_arr[0], Dy=data_arr[1],
                                           Vx=data_arr[2], Vy=data_arr[3])
        elif "disparr" in output_file_path_str and len(data_arr)==2:
            output_file_path_str = output_file_path_str[:-1]+"z"
            if var_name!=["UV_vec","UV_vec_sp"]: raise ValueError('Ordering must be "UV_vec","UV_vec_sp"')
            np.savez(output_file_path_str, UV_vec=data_arr[0], UV_vec_sp=data_arr[1])
        else: raise ValueError("saving procedure for list of arrays into npz file not yet implemented")
    
    ## Save NetCDF file (nc)
    elif filetype == "nc":
    
        if t0_datetime is None: t0_datetime = cfg_set["t0"]
        if dt is None:
            dt = cfg_set["time_change_factor"]*cfg_set["timestep"]
        
        ## Read auxilary data from cfg_set file:
        if var_name==["Dx","Dy","Vx","Vy"] or var_name==["UV_vec","UV_vec_sp"]:
            var_unit = ["Pixel "+str(dt)+"min-1","Pixel "+str(dt)+"min-1",\
                        "km "+str(dt)+"min-1","km "+str(dt)+"min-1"]
            var_type = np.float32
            longname = ["Displacement eastward","Displacement northward",\
                        "Optical flow eastward","Optical flow northward"]
        else:
            if var_type is None and type(var_name) is not list:
                var_type = cfg_set["type_dict"][var_name]
            if var_unit is None and type(var_name) is not list:
                var_unit = cfg_set["unit_dict"][var_name]
            if longname is None and type(var_name) is not list:
                longname = cfg_set["abbrev_dict"][var_name]
            
        ## Further checks whether all the necessary data is provided:
        if var_type is None and cfg_set is None:
            raise ValueError("either a variable type (var_type) or "+
                             "a cfg_set dictionary has to be provided")
        if var_type is None and var_name not in cfg_set["var_list"]:
            raise ValueError("variable name (var_name) not found in cfg_set dictionary")
        if var_unit is None and cfg_set is None:
            raise ValueError("either a variable unit (var_unit) or "+
                             "a cfg_set dictionary has to be provided")
        if var_unit is None and var_name not in cfg_set["var_list"]:
            raise ValueError("variable name (var_name) not found in cfg_set dictionary")
        
        ## Make description for different datasets:
        var_descib = var_name
        if "_orig" in output_file_path_str:
            description_nc = "Original observation of "
        elif "_disp_resid_combi" in output_file_path_str:
            description_nc = "Displaced observation (with residual movement correction with one displacement) of "
        elif "_disp_resid" in output_file_path_str:
            description_nc = "Displaced observation (with residual movement correction with one displacement) of "
        elif "_disparr_UV_resid_combi" in output_file_path_str:
            description_nc = "Displacement field (with residual movement)"
            var_descib = ""
        elif "_disparr_UV_resid" in output_file_path_str:
            description_nc = "Residual displacement field"
            var_descib = ""
        elif "_disparr_UV" in output_file_path_str:
            description_nc = "Displacement field"
            var_descib = ""
        elif "_disp" in output_file_path_str:
            description_nc = "Displaced observation of "
        else:
            print("  *** Warning: No description added to NetCDF file ***")
        
        description = description_nc+var_descib
        
        ## Save as NetCDF file:
        save_nc(output_file_path_str,data_arr,var_name,var_type,var_unit,longname,
                t0_datetime,description,dt=dt)
    else: raise ValueError("filetype must either be npy or nc.")    
  
 
## Read variable array (vararr) from Numpy or NetCDF file:
def load_file(input_file_path_str, var_name=None):
    """Read variable and displacement arrays into files.

    Parameters
    ----------
        
    input_file_path_str : str
        File path defining location of .nc file to be read.
        
    var_name : (list of) strings
        (List of) strings stating the names of the datasets.
        
    """

    if not exists(dirname(input_file_path_str)):
        print('*** ERROR, input file path does not exist: ' + dirname(input_file_path_str))
        exit()
    
    ## Adjust for disparr files (which should be understood as .npz files):
    if ("disparr" in input_file_path_str or "UV_vec" in input_file_path_str) and input_file_path_str[-4:]==".npy":
        input_file_path_str = input_file_path_str[:-4]+".npz"
    
    ## Analyse file ending:
    if input_file_path_str[-3:]==".nc":
        #if var_name==["Dx","Dy","Vx","Vy"] or var_name==["UV_vec","UV_vec_sp"]:
        #    data_arr = read_nc(input_file_path_str,var_name)
        #    return data_arr
        #if var_name is None:
        #    raise ValueError("variable name necessary to read NetCDF file.")
        if var_name is None:
            print("   *** Warning: Returning opened NetCDF file without closing ***")
            nc_file = read_nc(input_file_path_str,var_name)
            return nc_file
        elif type(var_name) is not list:
            data_arr = read_nc(input_file_path_str,var_name)
            return data_arr
        else:
            data_arr_ls = []
            for var in var_name:
                data_arr_ls.append(read_nc(input_file_path_str,var))
            return data_arr_ls
    elif input_file_path_str[-4:]==".npy":
        if type(var_name) is not list:
            data_arr = np.load(input_file_path_str)
            return(data_arr)
        else:
            raise ValueError("only one variable saved in .npy file.")
    elif input_file_path_str[-4:]==".npz":
        if var_name is None:
            #raise ValueError("several variable names needed to extract arrays from .npz file.")
            data_arr = np.load(input_file_path_str)
            return data_arr
        elif type(var_name) is list:
            data_arr = np.load(input_file_path_str)
            data_arr_ls = []
            for var in var_name:
                data_arr_ls.append(data_arr[var])
            return data_arr_ls
        else:
            data_arr = np.load(input_file_path_str)[var_name]
            return data_arr

## Save variable array (data_arr) as NetCDF file:
def save_nc(output_file_path_str,data_arr,var_name,var_type,var_unit,longname,
            datetime_object,description,dt=None,verbose=False):
    """Save variable array (data_arr) as NetCDF file.

    Parameters
    ----------
        
    output_file_path_str : str
        File path defining location of .nc file to be written.
        
    data_arr : (list of) numpy arrays
        (List of) numpy arrays containing the data to be saved.
        
    var_name : (list of) strings
        (List of) strings stating the names of the datasets.
        
    var_type : (list of) numpy datatypes (e.g. <type 'numpy.int8'>)
        (List of) numpy datatypes stating the datatype of the datasets.
        If only one type is provided, it is assumed to apply for all arrays.
        
    var_unit : (list of) units related to the respective array data.
        If only one type is provided, it is assumed to apply for all arrays.
        
    longname : (list of) describing longnames related to the respective array data.
        If only one type is provided, it is assumed to apply for all arrays.
        
    datetime_object : (list of) datetime objects
        (List of) datetime objects stating the time steps of the data
        (along the first dimension, starting with the most current observation).
        
    dt : int
        Number of minutes between different layers in first dimension of vararr
    
    """
    
    ## Put input data into lists (if not provided as list):
    if type(data_arr)        is not list: data_arr = [data_arr]
    if type(var_name)        is not list: var_name = [var_name]
    if type(var_type)        is not list: var_type = [var_type]
    if type(var_unit)        is not list: var_unit = [var_unit]
    if type(longname)        is not list: longname = [longname]
    if type(datetime_object) is not list: datetime_object = [datetime_object]
    
    ## Check length of input data:
    if len(set(map(np.shape, data_arr))) > 1:
        raise ValueError('variable arrays are not of the same shape (%s)' %
                          map(np.shape, data_arr))
    if len(datetime_object)!=1 and len(datetime_object)!=data_arr[0].shape[0]:
        raise ValueError('length of datetime object (%s) is unequal '+
                          'to one or the first dimension of the variable array (%s)' %
                          (len(datetime_object),data_arr[0].shape[0]))
    elif len(var_name)!=len(data_arr):
        raise ValueError('length of var_name list (%s) is unequal '+
                          'to the number of variable arrays (%s)' %
                          (len(var_name),len(data_arr)))
    #elif len(var_name)!=1 and len(var_name)!=len(data_arr):
    #    raise ValueError('length of var_name list (%s) is unequal '+
    #                      'to one or the first dimension of the variable array (%s)' %
    #                      (len(var_name),len(data_arr)))
    elif len(var_type)!=1 and len(var_type)!=len(data_arr):
        raise ValueError('length of var_type list (%s) is unequal '+
                          'to one or the first dimension of the variable array (%s)' %
                          (len(var_name),len(data_arr)))
    elif len(var_unit)!=1 and len(var_unit)!=len(data_arr):
        raise ValueError('length of var_unit list (%s) is unequal '+
                          'to one or the first dimension of the variable array (%s)' %
                          (len(var_unit),len(data_arr)))
    elif len(longname)!=1 and len(longname)!=len(data_arr):
        raise ValueError('length of longname list (%s) is unequal '+
                          'to one or the first dimension of the variable array (%s)' %
                          (len(longname),len(data_arr)))
    if len(datetime_object)==1 and len(datetime_object)!=len(data_arr) and dt is None:
        raise ValueError('length of time step has to be provided')

    if not exists(dirname(output_file_path_str)):
        print('... create output directory: ' + dirname(output_file_path_str))
        makedirs(dirname(output_file_path_str))
    
    ## Create NetCDF file:
    dataset = Dataset(output_file_path_str,
                      'w', format='NETCDF4_CLASSIC')
    dataset.history = 'Created ' + datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    dataset.description = description
                      
    ## Dimension creation:
    x    = dataset.createDimension('x', data_arr[0].shape[2])
    y    = dataset.createDimension('y', data_arr[0].shape[1])
    time = dataset.createDimension('time', None) # data_arr.shape[0])
    
    ## Auxilary variable creation:
    x_axis = dataset.createVariable('x', np.float32, ('x',))
    y_axis = dataset.createVariable('y', np.float32, ('y',))
    times  = dataset.createVariable('time', np.int16, ('time',)) # u8 or i8 does not work...
    times.calendar = 'standard'
    times.units = 'minutes since %s' % datetime_object[0].strftime("%Y-%m-%d %H:%M:%S")
    #times.units = 'seconds since 1970-01-01 00:00:00.0'
    
    ## Create time stamps variable:
    if len(datetime_object)==1:
        datetime_list = datetime_object - np.arange(data_arr[0].shape[0])*datetime.timedelta(minutes=dt)
    else: datetime_list = datetime_object
    times[:] = date2num(datetime_list,units=times.units)
    
    ## Create spatial coordinate variable:
    y_axis.units = 'Swiss northing CH1903 [km]'
    x_axis.units = 'Swiss easting CH1903 [km]'
    x_axis[:]    = np.arange(255,965)+0.5
    y_axis[::-1] = np.arange(-160,480)+0.5
    
    ## Data variable creation:
    var_name_list = var_name #if len(var_name)==1 else var_name*len(data_arr)
    var_type_list = var_type if len(var_name)==1 else var_type*len(data_arr)

    ## Write data into variables:
    id_var_list=[]
    for i in range(len(data_arr)):
        #if "int" in var_type_list[i]:
        #    id_var_list.append(dataset.createVariable(var_name_list[i],var_type_list[i],
        #                                              ('time','y','x'),zlib=True))
        #else:
        id_var_list.append(dataset.createVariable(var_name_list[i],var_type_list[i],
                                                 ('time','y','x'),zlib=True,
                                                 least_significant_digit=3)) #,least_significant_digit=2
        id_var_list[i].setncatts({'long_name': longname[i],'units': var_unit[i]})
        id_var_list[i][:,:,:] = data_arr[i]

    ## Close file:
    dataset.close()
    if verbose: print("   Written NetCDF file for: %s (%s)" %
                      (description,datetime_object[0].strftime("%d.%m.%y %H:%M")))
    
    
## Read variable array (data_arr) as NetCDF file:
def read_nc(input_file_path_str,var_name=None):
    """Read variable array (data_arr) as NetCDF file.

    Parameters
    ----------
        
    input_file_path_str : str
        File path defining location of .nc file to be read.
        
    var_name : (list of) strings
        (List of) strings stating the names of the datasets.
    
    Returns
    -------
    
    varrarr : (list of) numpy arrays
        (List of) numpy arrays containing the data to be read.

    """

    if not exists(dirname(input_file_path_str)):
        print('*** ERROR, input file path does not exist: ' + dirname(input_file_path_str))
        exit()

    if var_name is None:
        nc_file = Dataset(input_file_path_str,'r')
        return nc_file
    else: 
        nc_file = Dataset(input_file_path_str,'r')
        file = nc_file.variables[var_name][:,:,:]
        nc_file.close()
        return(file)
