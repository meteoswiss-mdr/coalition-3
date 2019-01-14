""" Functions for NOSTRADAMUS_1_animate.py:

Plot animation of ccs4 field.
"""

from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import pysteps as st

import NOSTRADAMUS_1_input_prep_fun as Nip

CONFIG_PATH = "/data/COALITION2/PicturesSatellite/results_JMZ/2_input_NOSTRADAMUS_ANN/"
CONFIG_FILE_set = "NOSTRADAMUS_1_input_prep.cfg" # "NOSTRADAMUS_1_input_prep_shorttest.cfg"
CONFIG_FILE_var = "cfg_var.csv"
CONFIG_FILE_var_combi = "cfg_var_combi.csv"

t_end_str = sys.argv[1]

cfg_set, cfg_var, cfg_var_combi = Nip.get_config_info(CONFIG_PATH,CONFIG_FILE_set,CONFIG_FILE_var,CONFIG_FILE_var_combi,t_end_str)
vars = [sys.argv[2]]
if len(sys.argv)==4: future = True if sys.argv[3]=="future" else False
else: future = False

if vars[0] == "selection": vars =["RZC","BZC","IR_108","THX_dens","CAPE_MU","EZC45",]

for var in vars: Nip.plot_displaced_fields(var,cfg_set,future=future,animation=False,TRT_form=True)


## Old version of plotting routine:
# Nip.plot_displaced_fields_old(var,cfg_set,resid=resid,animation=False,TRT_form=True)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
