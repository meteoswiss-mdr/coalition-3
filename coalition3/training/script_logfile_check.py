""" [COALITION3] Analyse the log file 'Training_Dataset_Processing_Status.pkl'
    and print some diagnostics."""

# ===============================================================================
# Import packages and functions

from __future__ import division
from __future__ import print_function

import sys
import datetime as dt
import pickle
import numpy as np

import coalition3.inout.paths as pth

print("\nPrint diagnostics on the progress of the stats/pixel count dataset generation:")
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
log_path       = pth.get_log_path(user_argv_path)

with open(log_path, "rb") as path: df = pickle.load(path)

print("\n---------------------------------------------------------------------------------------------------")
print("Current time: %s" % dt.datetime.now())

n_processed_TRT = np.sum(df["Processed"])
tot_TRT         = df.shape[0]
print("  Percentage of already processed TRT cells: ....................... %02d%% (%s/%s)" % \
      (100.*n_processed_TRT/tot_TRT, n_processed_TRT, tot_TRT))

n_processed_dt  = len(np.unique(df["date"].loc[df["Processed"]]))
tot_dt          = len(np.unique(df["date"]))
print("  Percentage of already processed time points: ..................... %02d%% (%s/%s)" % \
      (100.*n_processed_dt/tot_dt,n_processed_dt,tot_dt))

n_processing_TRT = np.sum(df["Processing"])
tot_TRT          = df.shape[0]
print("  Percentage of processing TRT cells: .............................. %02d%% (%s/%s)" % \
      (100.*n_processing_TRT/tot_TRT, n_processing_TRT, tot_TRT))

n_processing_dt  = len(np.unique(df["date"].loc[df["Processing"]]))
tot_dt           = len(np.unique(df["date"]))
print("  Percentage of processing time points: ............................ %02d%% (%s/%s)" % \
      (100.*n_processing_dt/tot_dt,n_processing_dt,tot_dt))
      
dates_recent = np.unique(df["Processing_End"].values[np.where(np.logical_and(df["Processing_End"]>dt.datetime.now()-dt.timedelta(hours=2),
                                                                             df["Processing_End"].notnull()))])
dates_recent_dt = dt.datetime.utcfromtimestamp(max(dates_recent).astype(int) * 1e-9)
if len(dates_recent)>0:
    print("  Newest xarray object created on: ................................. %s" % dates_recent_dt)
print("  Number of xarray objects created in the past 2h: ................. %s" % len(dates_recent))
if len(dates_recent)>5:
    timedeltas = [dates_recent[i-1]-dates_recent[i] for i in range(1, len(dates_recent))]
    average_timedelta = dt.timedelta(seconds=np.abs((sum(timedeltas) / len(dates_recent)).item()/10**9))
    print("  Average time delta between xarray object generated in the past 2h: %s" % average_timedelta)
    time_remaining = average_timedelta*(tot_dt-n_processed_dt)
    time_finishing = dt.datetime.now()+time_remaining
    print("  Expected time remaining: ......................................... %s" % time_remaining)
    print("  Expected time finishing: ......................................... %s" % time_finishing)
else:
    print("  No more than five xarray objects created in the past 2h")

print("---------------------------------------------------------------------------------------------------\n")








