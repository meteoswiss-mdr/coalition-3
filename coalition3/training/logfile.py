""" [COALITION3] Functions to generate, read, and edit logfile which
    captures information gathered during the generation of the training
    dataset and keeps track of what has already been processed."""

from __future__ import division
from __future__ import print_function

import os
import datetime
import pickle
import pdb
import numpy as np
import pandas as pd

import coalition3.training.processing as prc

## =============================================================================
## FUNCTIONS:

## Make log file which is containing information on training dataset generation:
def setup_log_file(cfg_set_tds,cfg_set_input):    
    """Make log file which is containing information on training dataset generation."""
    
    print("Create file containing processing status of training dataset generation.")
    samples_df = pd.read_pickle(os.path.join(cfg_set_tds["root_path_tds"],"Training_Dataset_Sampling_enhanced.pkl"))
    samples_df = samples_df.reset_index(drop=True)
        
    ## Only keep RANKs bigger than minimum value:
    samples_df = samples_df.loc[samples_df["RANKr"]>=cfg_set_input["min_TRT_rank"]*10]
    
    ## Add columns indicating whether time step has already been processed, 
    ## is currently being processed, and whether all input data is available.
    samples_df["Processing"]          = False 
    samples_df["Processing_Start"]    = None 
    samples_df["Processing_End"]      = None 
    samples_df["Processed"]           = False 
    samples_df["COSMO_CONV_missing"]  = False 
    samples_df["SEVIRI_missing"]      = False 
    samples_df["RADAR_missing"]       = False 
    samples_df["THX_missing"]         = False 
    samples_df["Border_cell"]         = False 
    samples_df["Non_NAN_stat"]        = False 
    
    ## Check input data availability:
    for counter, date_i in enumerate(np.unique(samples_df["date"])):
        perc_complete = 100*int(counter)/float(len(np.unique(samples_df["date"])))
        str_print = "  Completed %02d%% of dates" % (perc_complete)
        print('\r',str_print,end='') #, end='\r'
        date_i_dt = datetime.datetime.strptime(date_i,"%Y%m%d%H%M")
        #date_i_dt = datetime.datetime.utcfromtimestamp(date_i.astype(int) * 1e-9)
        samples_df, not_avail = check_input_data_availability(samples_df,date_i_dt,cfg_set_input,cfg_set_tds)
        #samples_df, not_avail = check_input_data_availability(samples_df,date_i_dt,None,None,
        #                                                      "/opt/users/jmz/0_training_NOSTRADAMUS_ANN/MissingInputData.pkl",
        #                                                      10,5)
        if not_avail:
            samples_df.loc[samples_df["date"]==date_i, "Processed"] = True
    
    ## Add TRT/Date ID:
    #samples_df["DATE_TRT_ID"]    =  np.array([date.strftime("%Y%m%d%H%M")+"_"+ID for date, ID in
    #                                          zip(samples_df["date"].astype(datetime).values,samples_df["traj_ID"].values)],
    #                                         dtype=np.object)
    samples_df["DATE_TRT_ID"] =  np.array([date+"_"+ID for date, ID in zip(samples_df["date"].values,samples_df["traj_ID"].values)],
                                          dtype=np.object)
    samples_df["date"]        =  [datetime.datetime.strptime(date,"%Y%m%d%H%M") for date in samples_df["date"].values]
    
    ## Save new dataset
    samples_df.to_pickle(os.path.join(cfg_set_tds["root_path_tds"],u'Training_Dataset_Processing_Status.pkl'))
    print("\n   Dataframe saved in: %s" % os.path.join(cfg_set_tds["root_path_tds"],u'Training_Dataset_Processing_Status.pkl'))
     
## Read and edit log file which is containing information on training dataset generation:
def read_edit_log_file(cfg_set_tds,cfg_set_input,process_point,t0_object=None,log_file=None,
                       samples_df=None,check_input_data=True):    
    """Read and edit log file which is containing information on training dataset generation."""
    
    print("Reading training dataset generation log file at %s of input generation process" % process_point)
    if log_file is None:
        log_file_path = os.path.join(cfg_set_tds["root_path_tds"],u'Training_Dataset_Processing_Status.pkl')
    if log_file is None and samples_df is None:
        log_file_path = os.path.join(cfg_set_tds["root_path_tds"],u'Training_Dataset_Processing_Status.pkl')
        with open(log_file_path, "rb") as path: samples_df = pickle.load(path)
    elif samples_df is None: 
        log_file_path = log_file
        with open(log_file_path, "rb") as path: samples_df = pickle.load(path)
    #samples_df = pd.read_pickle(log_file_path)
    
    if process_point=="start":
        ## Check all time steps which have not been processed yet and are not currently processed:
        samples_df_subset = samples_df.loc[(samples_df["Processed"].values==False) & \
                                           (samples_df["Processing"].values==False)]
        
        ## (Unfinished) Read list of time-points which are currently processed:
        #try:
        #    file_dates_currently_processing = "%s%s" % (cfg_set_tds["root_path_tds"],"Training_Dataset_Processing_Dates.npy")
        #    np.load(file_dates_currently_processing)
        #except IOError:
                    
        ## Pick random time step:
        if len(np.unique(samples_df_subset["date"]))==0:
            return_value = 200
            return(return_value)
        chosen_date = np.random.choice(np.unique(samples_df_subset["date"]))
        chosen_date = datetime.datetime.utcfromtimestamp(chosen_date.astype(int) * 1e-9)

        ## Set process starting time to current time:
        samples_df.loc[samples_df["date"]==chosen_date, "Processing_Start"] = datetime.datetime.now()
                                                   
        ## Check whether SEVIRI and COSMO input data is available:
        if check_input_data:
            samples_df, not_avail = check_input_data_availability(samples_df,chosen_date,cfg_set_input,cfg_set_tds)

        if not_avail:
            ## If any input data is missing, skip this time point and set 'Processed' to True:
            samples_df.loc[samples_df["date"]==chosen_date, "Processing_End"] = datetime.datetime.now()
            samples_df.loc[samples_df["date"]==chosen_date, "Processed"] = True
            #samples_df.to_pickle(log_file_path) 
            #with open(log_file_path, "wb") as output_file: pickle.dump(samples_df, output_file, protocol=-1)
            print("   For timepoint %s not all input data exists, skip this timepoint\n" % chosen_date.strftime("%d.%m.%Y %H:%M"))         
            return(None, samples_df)
        else:
            ## If all input data is available, actually start process (setting 'Processing' to True)
            ## by returning the respective time point:
            samples_df.loc[samples_df["date"]==chosen_date, "Processing"] = True
            #samples_df.to_pickle(log_file_path)
            with open(log_file_path, "wb") as output_file: pickle.dump(samples_df, output_file, protocol=-1)
            print("   Determine statistics for timepoint %s\n" % chosen_date.strftime("%d.%m.%Y %H:%M")) 
            return(chosen_date, None)
    
    elif process_point=="end":
        ## Note end-time of process, set 'Processing' back to False but 'Processed' to True:
        samples_df.loc[samples_df["date"]==t0_object, "Processing_End"] = datetime.datetime.now()
        samples_df.loc[samples_df["date"]==t0_object, "Processed"]      = True
        samples_df.loc[samples_df["date"]==t0_object, "Processing"]     = False
        print("\n",samples_df.loc[samples_df["date"]==t0_object],"\n")
        
        ## Delete files in /tmp directory of time steps where processing started more than 2h ago:
        dates_old = samples_df.loc[samples_df["Processing_End"]<datetime.datetime.now()-datetime.timedelta(hours=2),["Processing_Start"]]
        if len(dates_old)>0:
            dates_old_unique_str = np.unique(dates_old.iloc[:,0].dt.strftime('%Y%m%d%H%M'))
            for date_str in dates_old_unique_str[:-1]: prc.clean_disparr_vararr_tmp(cfg_set_input,fix_t0_str=date_str)
        
        ## Save the log file as pickle and as csv:
        with open(log_file_path, "wb") as output_file: pickle.dump(samples_df, output_file, protocol=-1)
        #samples_df.to_pickle(log_file_path)
        samples_df.to_csv(log_file_path[:-3]+"csv")

        ## Return percentage of processed files plus 100:
        print("   Finished process for timepoint %s." % t0_object.strftime("%d.%m.%Y %H:%M"))
        return_value = int(100.*np.sum(samples_df["Processed"])/len(samples_df["Processed"]))+100
        return(return_value)
    else: raise ValueError("process_point argument must either be 'start' or 'end'")

## Function which checks whether the input data at the respective time point is available:
def check_input_data_availability(samples_df,time_point,cfg_set_input,cfg_set_tds,
                                  missing_date_df_path=None,n_integ=None,timestep=None):
                                  
    if missing_date_df_path is None:
        missing_date_df_path = "%s%s" % (cfg_set_tds["root_path_tds"],"MissingInputData.pkl")
    if n_integ is None:
        n_integ = cfg_set_input["n_integ"]
    if timestep is None:
        timestep = cfg_set_input["timestep"]
    
    dates_of_needed_input_data = pd.date_range(time_point-n_integ*datetime.timedelta(minutes=timestep),
                                               time_point+n_integ*datetime.timedelta(minutes=timestep),
                                               freq=str(timestep)+'Min')
    with open(missing_date_df_path, "rb") as path: missing_date_df = pickle.load(path)
    t_ind = (missing_date_df.index >= dates_of_needed_input_data[0]) & \
            (missing_date_df.index <= dates_of_needed_input_data[-1])

    missing_COSMO_CONV = np.any(missing_date_df.loc[t_ind,"COSMO_CONV"])
    missing_SEVIRI     = np.any(missing_date_df.loc[t_ind,"SEVIRI"])
    missing_THX        = np.any(missing_date_df.loc[t_ind,"THX"])
    missing_RADAR      = np.any(missing_date_df.loc[t_ind,["RZC","BZC","LZC","MZC","EZC15","EZC20","EZC45","EZC50","CZC"]])  
            
    samples_df.loc[samples_df["date"]==time_point, "COSMO_CONV_missing"]  = missing_COSMO_CONV
    samples_df.loc[samples_df["date"]==time_point, "SEVIRI_missing"]      = missing_SEVIRI
    samples_df.loc[samples_df["date"]==time_point, "THX_missing"]         = missing_THX
    samples_df.loc[samples_df["date"]==time_point, "RADAR_missing"]       = missing_RADAR
    
    not_all_input_data_available = np.any([missing_COSMO_CONV,
                                           missing_SEVIRI,
                                           missing_THX,
                                           missing_RADAR])
    return(samples_df, not_all_input_data_available)    
     
## Select only dates which have cells with TRT rank larger than threshold:
def select_high_TRTrank_dates(log_file_path, new_filename, TRT_treshold):
    """From log-file, select only time points which contain a TRT cell higher than a specific value"""
    print("Only keep time points with TRT cells above rank %s" % TRT_threshold)
    
    df = pd.read_pickle(log_file_path)
    dates_high_rank = np.unique(df["date"].loc[df["RANKr"]>=TRT_treshold*10.])
    df_high_rank    = df_merge.iloc[np.in1d(df["date"], dates_high_rank)]

    n_processed_TRT = np.sum(df_high_rank["Processed"])
    tot_TRT         = df_high_rank.shape[0]
    print("  Percentage of already processed TRT cells:  %02d%%" % (100.*n_processed_TRT/tot_TRT))

    n_processed_dt  = len(np.unique(df_high_rank["date"].loc[df_high_rank["Processed"]]))
    tot_dt          = len(np.unique(df_high_rank["date"]))
    print("  Percentage of already processed time poins: %02d%%" % (100.*n_processed_dt/tot_dt))

    df_high_rank.to_pickle(new_filename)
    print("  Log-file dataframe with restricted dates saved in: %s" % (new_filename))
    
## Copy entries from old log_file into new log_file:
def copy_log_file_entries(file_old_path, file_new_path, filename_merge):
    """Copy entries from old log_file into new log_file. New log_file dateframe must contain old
    dataframe columns (DATE_TRT_ID, Processing, Processed, Processing_Start, Processing_End,
    COSMO_CONV_missing, SEVIRI_missing, RADAR_missing, THX_missing, Border_cell, Non_NAN_stat)"""
    
    df_old = pd.read_pickle(file_old_path)
    df_new = pd.read_pickle(file_new_path)

    df_old.Processing_Start = df_old.Processing_Start.astype(np.object)
    df_old.Processing_End   = df_old.Processing_End.astype(np.object)

    column_ls = ["Processing", "Processed", "Processing_Start", "Processing_End", \
                 "COSMO_CONV_missing", "SEVIRI_missing", "RADAR_missing", "THX_missing", "Border_cell", "Non_NAN_stat"]

    df_old_red = df_old[column_ls+["DATE_TRT_ID"]]
    df_new_red = df_new.drop(column_ls,axis=1)
    df_merge   = pd.merge(df_new_red,df_old_red,on="DATE_TRT_ID",how="left")
    for column_name in ["Processing", "Processed", "COSMO_CONV_missing", "SEVIRI_missing", \
                        "RADAR_missing", "THX_missing", "Border_cell", "Non_NAN_stat"]:
        df_merge[column_name].fillna(False,inplace=True)

    df_merge.to_pickle(filename_merge)
    print("   Merged log-file dataframe saved in: %s" % (filename_merge))
    
    """    
    for DATE_TRT_ID in df_old["DATE_TRT_ID"].values:
        #print_str = "Working on DATE_TRT_ID: %s" % DATE_TRT_ID
        print_str = "Processed %02d%%" % (100.*np.where(df_old["DATE_TRT_ID"].values==DATE_TRT_ID)[0][0]/len(df_old["DATE_TRT_ID"].values))
        print(print_str, end="\r")
        
        
        if DATE_TRT_ID not in df_new["DATE_TRT_ID"].values:
            #raise ValueError("DATE_TRT_ID not found in new dataframe")
            missing_DATE_TRT_ID_ls.append(DATE_TRT_ID)
        else:
            for column in column_ls:
                df_new[column].loc[df_new["DATE_TRT_ID"]==DATE_TRT_ID] = df_old[column].loc[df_old["DATE_TRT_ID"]==DATE_TRT_ID].values
    """







