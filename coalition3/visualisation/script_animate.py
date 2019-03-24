""" [COALITION3] Print animations of CCS4 fields in the current /tmp directory """

from __future__ import division
from __future__ import print_function

import sys

import coalition3.inout.readconfig as cfg
import coalition3.visualisation.dispplot as dpl

## Get user arguments
t_end_str = sys.argv[1]

cfg_set, cfg_var, cfg_var_combi = cfg.get_config_info_op()
cfg_set = cfg.cfg_set_append_t0(cfg_set,t_end_str)

vars = [sys.argv[2]]
if len(sys.argv)==4: future = True if sys.argv[3]=="future" else False
else: future = False

if vars[0] == "selection": vars =["RZC","BZC","IR_108","THX_dens","CAPE_ML","EZC45",]

## Do the plots
for var in vars: dpl.plot_displaced_fields(var,cfg_set,future=future,animation=False,TRT_form=True)

## Old version of plotting routine:
# dpl.plot_displaced_fields_old(var,cfg_set,resid=resid,animation=False,TRT_form=True)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
