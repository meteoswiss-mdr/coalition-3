""" [COALITION3] Functions to prepare generation of training dataset."""

from __future__ import division
from __future__ import print_function

import configparser
import datetime
import numpy as np
import pysteps as st

## =============================================================================
## FUNCTIONS:

## Create list of datetime objects, at which samples should be generated for the training dataset:
def create_dt_sampling_list(cfg_set_tds):
    """Create list of datetime objects, at which samples should be
    generated for the training dataset.
    """
    print("Creating list of sampling datetime objects")
    
    ## Append further datetime objects accroding to settings in cfg_set_tds:
    dt_temp = datetime.datetime.combine(cfg_set_tds["tds_period_start"],datetime.time(0,0))
    
    ## Insert starting date as first element:
    dt_sampling_list = []
    while dt_temp.date()<=cfg_set_tds["tds_period_end"]:
        dt_sampling_list.append(dt_temp)
        day_temp = dt_temp.day
        dt_temp = dt_temp+datetime.timedelta(minutes=cfg_set_tds["dt_samples"])
        if day_temp < dt_temp.day:
            dt_temp = dt_temp+datetime.timedelta(minutes=cfg_set_tds["dt_daily_shift"])
            day_temp = dt_temp.day
    print("  Number of sampling times points: %s" % len(dt_sampling_list))
    return dt_sampling_list












