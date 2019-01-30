#!/opt/users/common/packages/anaconda2//bin

""" Little script to convert hourly THX ascii files (.prod) into daily NetCDF fiels:
This script calls the function to convert an ascii file (.prod) with information 
on every lightning during a day into a NetCDF file with 6 fields (dens, densCG, densIC,
curr_abs, curr_pos, curr_neg) on ccs4 grid every 5minutes.

"""

# ===============================================================================
# Import packages and functions

import datetime
import sys
import numpy as np
sys.path.insert(0, '/opt/users/jmz/monti-pytroll/packages/mpop')
#import mpop
from mpop.satin import swisslightning_jmz

## Run script:
THX_path_in  = "/data/lom/WOL/foudre/data/THX"
THX_path_out = "/data/COALITION2/database/THX"

date0 = datetime.datetime.strptime("201804140000", "%Y%m%d%H%M")
dates = date0+np.arange(172)*datetime.timedelta(days=1)

#date0 = datetime.datetime.strptime("201507070000", "%Y%m%d%H%M")
#dates = [date0]

#swisslightning_jmz.convert_THXprod_nc_daily(THX_path_in,THX_path_out,dates)
swisslightning_jmz.convert_THXprod_nc_daily_direct(THX_path_in,THX_path_out,dates)
    
