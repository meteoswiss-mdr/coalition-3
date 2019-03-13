""" [COALITION3] Code to get long lasting TRT cells """

import pickle
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pylab as plt
from collections import Counter
import coalition3.statlearn.inputprep as ipt
from sklearn.neural_network import MLPRegressor

def shift_values_to_timestep(df, var_name, TRT_var=None):
    all_timesteps = pd.date_range(start = df.index[ 0] - dt.timedelta(minutes=45),
                                  end   = df.index[-1] + dt.timedelta(minutes=45),
                                  freq = '5min')
    df = pd.concat([pd.DataFrame([], index = all_timesteps),df], axis=1, sort=True)
    
    df_var = df[[colname_var for colname_var in df.columns if var_name+"|" in colname_var]]
    if df_var.shape[1]==0:
        print("   *** Warning: No variable %s found in column names, returning empty df ***" % var_name)
        return df_var
    
    ls_var_new = []
    if TRT_var is not None:
        ls_var_new.append(df[TRT_var])
    for i, time_del in enumerate(np.arange(-45,50,5)):
        df_var_dt = df_var[[colname_dt for colname_dt in df_var.columns if "|%i" % time_del in colname_dt]]
        if df_var_dt.shape[1]!=0:
            df_var_dt_short = df_var_dt.iloc[9:-9,:].values[:,0]
            ls_var_new.append(pd.DataFrame(df_var_dt_short, columns=df_var_dt.columns,
                                           index=df_var_dt.index[i:i+len(df_var_dt_short)]))
    df_var_shift = pd.concat(ls_var_new, axis=1, sort=True)
    return df_var_shift
#xr_loc = xr.open_dataset("<Add your path here>")
#xr_loc
#TRT_ID = xr_loc.DATE_TRT_ID

path_to_df = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/training_dataset/stat_output_20190214/diam_23km/nc/Combined_stat_pixcount_df_t0diff_nonnan.h5"
df_nonnan  = pd.read_hdf(path_to_df,key="df_nonnan")
pred_dt = 20
X_train, X_test, y_train, y_test, scaler = ipt.get_model_input(df_nonnan,
    del_TRTeqZero_tpred=True, split_Xy_traintest=True, X_normalise=True,
    pred_dt=pred_dt)
TRT_ID = X_test.index


TRT_ID = [TRT_ID_i[13:] for TRT_ID_i in TRT_ID.values] 

len(np.unique(TRT_ID))
TRT_ID_count = Counter(TRT_ID)
TRT_ID_count_sort = [(key,value) for key, value in sorted(TRT_ID_count.iteritems(), key=lambda (k,v): (v,k))]

TRT_ID_count_sort_pd = pd.DataFrame(np.array(TRT_ID_count_sort),columns=["TRT_ID","Count"])
TRT_ID_count_sort_pd["Count"] = TRT_ID_count_sort_pd["Count"].astype(np.uint16,inplace=True)
TRT_ID_count_sort_pd.info()

TRT_ID_long = TRT_ID_count_sort_pd.loc[TRT_ID_count_sort_pd["Count"]>30]
for i in [8,10]: #range(len(TRT_ID_long)): #[12]: #
    TRT_ID_sel  = TRT_ID_long.iloc[-i,:]


    ## Find cells where the there are loads of similar TRT Ranks:
    #DTI_long = [dti for dti in xr_loc.DATE_TRT_ID.values if dti[13:] in TRT_ID_long["TRT_ID"].values]
    #DTI_max  = [dti for dti in xr_loc.DATE_TRT_ID.values if dti[13:] in TRT_ID_max["TRT_ID"].values]
    DTI_long = [dti for dti in X_test.index.values if dti[13:] in TRT_ID_long["TRT_ID"].values]
    DTI_sel  = [dti for dti in X_test.index.values if dti[13:] in TRT_ID_sel["TRT_ID"]]
    cell_sel = df_nonnan.loc[DTI_sel]
    #cell_sel["TIME"] = pd.to_datetime([dt.datetime.strptime(date[:12],"%Y%m%d%H%M") for date in cell_sel.index])
    cell_sel.set_index(pd.to_datetime([dt.datetime.strptime(date[:12],"%Y%m%d%H%M") for date in cell_sel.index]),
                       drop=True,append=False,inplace=True)

    #cell_sel_allt = pd.concat([pd.DataFrame([], index = all_timesteps),cell_sel], axis=1, sort=True)
    #df_TRT_shift = shift_values_to_timestep(cell_sel,"LZC_stat","VIL")
    #df_TRT_shift = shift_values_to_timestep(cell_sel,"TRT_Rank","RANKr")
    #df_TRT_shift = shift_values_to_timestep(cell_sel,"TRT_Rank","RANKr")
    df_TRT_shift = shift_values_to_timestep(cell_sel,"TRT_Rank","RANKr")
    df_TRT_shift["RANKr"]/=10.
    print(np.mean(cell_sel["RANKr"]), np.max(cell_sel["RANKr"]), i)
    fig = plt.figure(figsize = [14,9])
    ax = fig.add_subplot(1,1,1)
    df_TRT_shift[[col for col in df_TRT_shift.columns if "TRT" in col]].plot.line(ax = ax, cmap="Spectral",linewidth=1)#; plt.show()
    df_TRT_shift["RANKr"].plot.line(ax=ax, color="black",linestyle="--",linewidth=2)
    #df_TRT_pred_shift.plot.line(ax=ax, color="red",linestyle="-.",linewidth=2)
    plt.grid()
    plt.pause(4)
    plt.close()

xgb_model_path = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/statistical_learning/feature_selection/models/diam_23km/all_samples/model_20_t0diff_maxdepth6.pkl"
with open(xgb_model_path,"rb") as file:
    xgb_model = pickle.load(file)
mlp_model_path_nfeat = "/data/COALITION2/PicturesSatellite/results_JMZ/0_training_NOSTRADAMUS_ANN/statistical_learning/ANN_models/models/diam_23km/model_20_t0diff_mlp_500feat.pkl"
with open(mlp_model_path_nfeat,"rb") as file:
    mlp_model = pickle.load(file) #[-1].best_estimator_
    
top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                           orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                           ascending=False)

X_test_sel = X_test.loc[DTI_sel]
y_test_sel = y_test.loc[DTI_sel]
pred_sel = mlp_model.predict(X_test_sel[top_features_gain.index[:500]])
cell_sel["TRT_Rank_pred|%i" % pred_dt] = cell_sel["TRT_Rank|0"]+pred_sel
df_TRT_pred_shift = shift_values_to_timestep(cell_sel,"TRT_Rank_pred")


ax = df_TRT_shift[[col for col in df_TRT_shift.columns if "TRT" in col]].plot.line(cmap="Spectral",linewidth=1)#; plt.show()
df_TRT_shift["RANKr"].plot.line(ax=ax, color="black",linestyle="--",linewidth=2)
df_TRT_pred_shift.plot.line(ax=ax, color="red",linestyle="-.",linewidth=2)
plt.grid()
plt.show()

#DTI_eq_TRTRank = [dti for dti in DTI_long if len(np.unique(xr_loc["TRT_Rank"].sel(DATE_TRT_ID=dti)))<10]
#TRT_Ranks = [np.float(xr_loc["TRT_Rank"].sel(DATE_TRT_ID=dti, time_delta=0).values) for dti in DTI_eq_TRTRank]
#DTI_long_ex = [dti for dti in DTI_eq_TRTRank if "_2018080718050095" in dti]
#TRT_Rank_ex = [np.float(xr_loc["RANKr"].sel(DATE_TRT_ID=DTI_ex).values)/10. for DTI_ex in DTI_long_ex]