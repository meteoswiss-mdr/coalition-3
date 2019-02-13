""" [COALITION3] Code to get long lasting TRT cells """

import xarray as xr
import numpy as np
import pandas as pd
from collections import Counter

xr_loc = xr.open_dataset("<Add your path here>")
xr_loc

TRT_ID = xr_loc.DATE_TRT_ID
TRT_ID = [TRT_ID_i[13:] for TRT_ID_i in TRT_ID.values] 
len(TRT_ID)

len(np.unique(TRT_ID))
TRT_ID_count = Counter(TRT_ID)
TRT_ID_count_sort = [(key,value) for key, value in sorted(TRT_ID_count.iteritems(), key=lambda (k,v): (v,k))]

TRT_ID_count_sort_pd = pd.DataFrame(np.array(TRT_ID_count_sort),columns=["TRT_ID","Count"])
TRT_ID_count_sort_pd["Count"] = TRT_ID_count_sort_pd["Count"].astype(np.uint16,inplace=True)
TRT_ID_count_sort_pd.info()

TRT_ID_long = TRT_ID_count_sort_pd.loc[TRT_ID_count_sort_pd["Count"]>30]


## Find cells where the there are loads of similar TRT Ranks:
DTI_long = [dti for dti in xr_loc.DATE_TRT_ID.values if dti[13:] in TRT_ID_long["TRT_ID"].values]
DTI_eq_TRTRank = [dti for dti in DTI_long if len(np.unique(xr_loc["TRT_Rank"].sel(DATE_TRT_ID=dti)))<10]

TRT_Ranks = [np.float(xr_loc["TRT_Rank"].sel(DATE_TRT_ID=dti, time_delta=0).values) for dti in DTI_eq_TRTRank]

DTI_long_ex = [dti for dti in DTI_eq_TRTRank if "_2018080718050095" in dti]
TRT_Rank_ex = [np.float(xr_loc["RANKr"].sel(DATE_TRT_ID=DTI_ex).values)/10. for DTI_ex in DTI_long_ex]