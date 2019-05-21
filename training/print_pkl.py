from __future__ import print_function
import sys
import pickle

if len(sys.argv) != 2:
    print ("use this script: python print_pkl.py filename.pkl")
else:
    print ('decode:', str(sys.argv[1]))

    #with open('Training_Dataset_Sampling.pkl', 'rb') as f:
    with open(str(sys.argv[1]), 'rb') as f:
        data = pickle.load(f)
    print (data)
