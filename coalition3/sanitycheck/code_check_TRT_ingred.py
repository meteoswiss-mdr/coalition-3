# coding: utf-8
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("201805091905_stat_pixcount.pkl","rb") as file: ds= pickle.load(file)
ds.to_netcdf("201805091905_stat_pixcount.nc")

ds
ds["TRT_Rank"].sel(DATE_TRT_ID="201805091905_2018050918200166")
ds["TRT_cellcentre_indices"].sel(DATE_TRT_ID="201805091905_2018050918200166")
ds["lat"].sel(DATE_TRT_ID="201805091905_2018050918200166")
ds["lon"].sel(DATE_TRT_ID="201805091905_2018050918200166")
ds["iCH"].sel(DATE_TRT_ID="201805091905_2018050918200166")
ds["jCH"].sel(DATE_TRT_ID="201805091905_2018050918200166")
ds_ex = ds.sel(DATE_TRT_ID="201805091905_2018050918200166")


da_EZC = xr.open_dataset("201805091905_EZC45_orig.nc")
da_CZC = xr.open_dataset("201805091905_CZC_orig.nc")
da_VIL = xr.open_dataset("201805091905_LZC_orig.nc")

ar_EZC = da_EZC["EZC45"].values[0,:,:]
ar_CZC = da_CZC["CZC"].values[0,:,:]
ar_VIL = da_VIL["LZC"].values[0,:,:]
plt.imshow(ar_VIL); plt.show()

for i, dt in enumerate(np.arange(0,-50,-5)):
    print(da_VIL["LZC"].values[i,:,:].flatten()[ds["TRT_domain_indices"].sel(time_delta=dt,DATE_TRT_ID="201805091905_2018050919050039")])
    print(np.max(da_VIL["LZC"].values[i,:,:].flatten()[ds["TRT_domain_indices"].sel(time_delta=dt,DATE_TRT_ID="201805091905_2018050919050039")]))

    flat_ar_VIL = da_VIL["LZC"].values[i,:,:].flatten()
    flat_ar_VIL[ds["TRT_domain_indices"].sel(time_delta=dt,DATE_TRT_ID="201805091905_2018050919050039")] = 1000
    nonflat_ar_VIL = flat_ar_VIL.reshape(ar_VIL.shape)
    plt.clf()
    plt.imshow(da_VIL["LZC"].values[i,:,:])
    plt.contour(nonflat_ar_VIL)
    plt.pause(2)
plt.close()

for dt in np.arange(-45,5,5): print(ar_CZC.flatten()[ds["TRT_domain_indices"].sel(time_delta=dt,DATE_TRT_ID="201805091905_2018050918200166")])
for dt in np.arange(-45,5,5): print(np.max(ar_CZC.flatten()[ds["TRT_domain_indices"].sel(time_delta=dt,DATE_TRT_ID="201805091905_2018050918200166")]))


