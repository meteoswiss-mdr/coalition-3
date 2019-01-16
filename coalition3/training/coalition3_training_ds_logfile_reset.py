#!/opt/users/common/packages/anaconda2//bin

""" Correct the log file 'Training_Dataset_Processing_Status.pkl' after one run of
the training dataset generation by resetting the entries 'Processing=True' to
'Processing=False', which can then be processed again."""

# ===============================================================================
# Import packages and functions

from __future__ import division
from __future__ import print_function

import sys
import os
import pandas as pd
import NOSTRADAMUS_0_training_ds_fun as Nds
import pickle

print("\nThis scipt is used during the generation of the training dataset.\n \
It happens that the displacement and statistics-calculation process \n \
is interrupted and no results are produced. When all time steps are \n \
processed, one can use this script to reset the entry in the log file \n \
from 'Processing=True' back to 'Processing=False', so that in a second \n \
run these time steps are again processed. \n\n \
  --> Do not forget to clean up the tmp/ directory! <--")
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
log_path      = Nds.get_log_path(user_argv_path)

with open(log_path, "rb") as path: df = pickle.load(path)
#df = pd.read_pickle(log_path)

print("These time steps have entry 'Processing=True':")
print(df.loc[df["Processing"]])
print("Set 'Processing=False' everywhere")
df["Processing"]=False

new_filename = log_path[:-4]+"_reset.pkl"
#df.to_pickle(new_filename)
with open(new_filename, "wb") as output_file: pickle.dump(df, output_file, protocol=-1)
print("\nSaved to new pickle file (%s), replace old one manually." % os.path.basename(new_filename))