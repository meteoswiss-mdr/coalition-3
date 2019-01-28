""" Functions for NOSTRADAMUS_1_input_prep.py:

The following functions are assembled here for the input data generation
of NOSTRADAMUS:
"""

from __future__ import division
from __future__ import print_function

import sys
import os
import configparser
import datetime
import pdb

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import pysteps as st
import ephem
from netCDF4 import Dataset
from scipy import ndimage, signal, interpolate, spatial

from joblib import Parallel, delayed
import multiprocessing

#from contextlib import contextmanager
#import warnings
#import ast

sys.path.insert(0, '/data/COALITION2/database/radar/ccs4/python')
import metranet
sys.path.insert(0, '/opt/users/jmz/monti-pytroll/packages/mpop')
from mpop.satin import swisslightning_jmz, swisstrt, swissradar

## ===============================================================================
## FUNCTIONS:



    

  
    

    
    
    
    
    
