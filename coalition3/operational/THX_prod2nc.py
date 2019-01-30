""" Little script to convert hourly THX ascii files (.prod) into daily NetCDF fiels:
This script calls the function to convert an ascii file (.prod) with information 
on every lightning during a day into a NetCDF file with 6 fields (dens, densCG, densIC,
curr_abs, curr_pos, curr_neg) on ccs4 grid every 5minutes."""

## Import packages and functions
import sys
import datetime
import numpy as np

## Check location of mpop package (THIS PART OF THE SCRIPT IS HARD-CODED!):
mpop_path = '/opt/users/jmz/monti-pytroll/packages/mpop'
bool_continue = raw_input("\nIs this mpop.satin package path '%s' correct? [y/n] " % mpop_path)=="y"
if not bool_continue: print("  Script aborted, change input mpop path in script"); sys.exit()
sys.path.insert(0, mpop_path)
from mpop.satin import swisslightning_jmz

## Input dates
start_date = datetime.datetime.strptime(sys.argv[1], "%Y%m%d") #.strftime("%Y%m%d%H%M")
end_date   = datetime.datetime.strptime(sys.argv[2], "%Y%m%d") #.strftime("%Y%m%d%H%M")
dates      = [start_date + datetime.timedelta(days=x) for x in range((end_date-start_date).days + 1)]
print("\nStart converting THX prod files between\n  %s to\n  %s\ninto 5min aggregates in daily NetCDF files\n" % 
      (start_date,end_date))

## Set paths (THIS PART OF THE SCRIPT IS HARD-CODED!):
THX_path_in  = "/data/lom/WOL/foudre/data/THX"
THX_path_out = "<insert your output path here>"

bool_continue = raw_input("Is this input path '%s' correct? [y/n] " % THX_path_in)=="y"
if not bool_continue: print("  Script aborted, change input path in script"); sys.exit()
bool_continue = raw_input("Is this output path '%s' correct? [y/n] " % THX_path_out)=="y"
if not bool_continue: print("  Script aborted, change input path in script"); sys.exit()

## Convert prod files to nc files:
swisslightning_jmz.convert_THXprod_nc_daily_direct(THX_path_in,THX_path_out,dates)