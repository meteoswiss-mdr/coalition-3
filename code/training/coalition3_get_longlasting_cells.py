# coding: utf-8
import xarray as xr
import numpy as np
xr_loc = xr.open_dataset("diam_16km/Combined_stat_pixcount.nc")
xr_loc
get_ipython().magic(u'll -h diam_16km/')
TRT_ID = xr_loc.DATE_TRT_ID
TRT_ID = 
TRT_ID 
TRT_ID = [TRT_ID_i[12:] for TRT_ID_i in TRT_ID] 
TRT_ID = [TRT_ID_i[12:] for TRT_ID_i in TRT_ID.values] 
TRT_ID
TRT_ID = [TRT_ID_i[1:] for TRT_ID_i in TRT_ID.values] 
TRT_ID = [TRT_ID_i[1:] for TRT_ID_i in TRT_ID] 
TRT_ID
len(TRT_ID)
len(TRT_ID.shape)
TRT_ID.shape
np.unique(TRT_ID)
len(np.unique(TRT_ID))
from collections import Counter
TRT_ID_count = Counter(TRT_ID)
TRT_ID_count
TRT_ID_count_sort = [(key,value) for key, value in sorted(TRT_ID_count.iteritems(), key=lambda (k,v): (v,k))]
TRT_ID_count_sort
import pandas as pd
pd.DataFrame.from_dict(TRT_ID_count)
np.array(TRT_ID_count_sort)
np.array(TRT_ID_count_sort).shape
TRT_ID_count_sort_pd = pd.DataFrame(np.array(TRT_ID_count_sort)))
TRT_ID_count_sort_pd = pd.DataFrame(np.array(TRT_ID_count_sort))
TRT_ID_count_sort_pd
TRT_ID_count_sort_pd.info
TRT_ID_count_sort_pd.info()
xrange(10)
get_ipython().magic(u'save get_longlasting_cells.py ~0/')
