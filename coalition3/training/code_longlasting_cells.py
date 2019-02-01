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
TRT_ID_count_sort_pd
TRT_ID_count_sort_pd.info()
