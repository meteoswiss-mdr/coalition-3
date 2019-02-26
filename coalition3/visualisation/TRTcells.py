""" [COALITION3] Plotting locations of TRT cells within training dictionary
    and histograms of different TRT statistics"""

from __future__ import division
from __future__ import print_function

import os
import datetime
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.colors import LightSource
from scipy import ndimage
from PIL import Image
import shapefile
import scipy.ndimage.morphology as morph


## =============================================================================
## FUNCTIONS:
             
## Plot contour lines in 2D histogram, showing the fraction of points within contour line:
def contour_of_2dHist(hist2d_1_data,percentiles=[0,40,60,80,95,100],smooth=True):
    if True:
        counts_total = np.sum(hist2d_1_data)
        hist2d_1_cumsum = np.zeros(len(hist2d_1_data.flatten()))
        hist2d_1_cumsum[np.argsort(hist2d_1_data.flatten())] = \
            np.cumsum(hist2d_1_data.flatten()[np.argsort(hist2d_1_data.flatten())])
        hist2d_1_data = hist2d_1_cumsum.reshape(hist2d_1_data.shape)
        hist2d_1_data = 100-hist2d_1_data/np.sum(counts_total)*100.
    else:
        non_zero_perc_vals = np.percentile(hist2d_1_data[hist2d_1_data>0],
                                           percentiles[1:-1])
    if smooth:
        hist2d_1_data_smooth = ndimage.gaussian_filter(hist2d_1_data,hist2d_1_data.shape[0]//100)
        hist2d_1_data_smooth[hist2d_1_data==0] = 0
        hist2d_1_data = hist2d_1_data_smooth

    if True:
        hist_2d_perc = hist2d_1_data; levels = percentiles[1:-1]
    else:
        hist_2d_perc = np.searchsorted(non_zero_perc_vals,hist2d_1_data)
        for val_old, val_new in zip(np.unique(hist_2d_perc), percentiles):
            hist_2d_perc[hist_2d_perc==val_old] = val_new
        levels = np.unique(hist_2d_perc)[1:]

    return hist_2d_perc.T, levels
             
## Print histogram of TRT cell values:
def print_TRT_cell_histograms(samples_df,cfg_set_tds):
    """Print histograms of TRT cell information."""
    
    fig_hist, axes = plt.subplots(3, 2)
    fig_hist.set_size_inches(12, 15)

    ## Analyse distribution of ranks
    nw = np.sum(np.logical_and(samples_df["RANKr"]>=12, samples_df["RANKr"]<15))
    ng = np.sum(np.logical_and(samples_df["RANKr"]>=15, samples_df["RANKr"]<25))
    ny = np.sum(np.logical_and(samples_df["RANKr"]>=25, samples_df["RANKr"]<35))
    nr = np.sum(np.logical_and(samples_df["RANKr"]>=35, samples_df["RANKr"]<=40))
    print("  The number of Cells with TRT Rank w is: %s" % nw)
    print("  The number of Cells with TRT Rank g is: %s" % ng)
    print("  The number of Cells with TRT Rank y is: %s" % ny)
    print("  The number of Cells with TRT Rank r is: %s" % nr)
    samples_df["RANKr"] = samples_df["RANKr"]/10.
    pw = patches.Rectangle((1.2, 65000), 0.3, 10000, facecolor='w')
    pg = patches.Rectangle((1.5, 65000),   1, 10000, facecolor='g')
    py = patches.Rectangle((2.5, 65000),   1, 10000, facecolor='y')
    pr = patches.Rectangle((3.5, 65000), 0.5, 10000, facecolor='r')
    axes[0,0].add_patch(pw); axes[0,0].add_patch(pg); axes[0,0].add_patch(py); axes[0,0].add_patch(pr)
    axes[0,0].annotate(str(nw),(1.35,70000),(1.25,90500),ha='center',va='center',color='k',arrowprops={'arrowstyle':'->'}) #,arrowprops={arrowstyle='simple'}
    axes[0,0].annotate(str(ng),(2,70000),ha='center',va='center',color='w') 
    axes[0,0].annotate(str(ny),(3,70000),ha='center',va='center',color='w')
    axes[0,0].annotate(str(nr),(3.75,70000),ha='center',va='center',color='w') 
    samples_df["RANKr"].hist(ax=axes[0,0],bins=np.arange(0,4.25,0.25),facecolor=(.7,.7,.7),alpha=0.75,grid=True)
    axes[0,0].set_xlabel("TRT rank")
    axes[0,0].set_title("TRT Rank Distribution")
    
    samples_df["area"].hist(ax=axes[0,1],bins=np.arange(0,650,50),facecolor=(.7,.7,.7),alpha=0.75,grid=True)
    axes[0,1].set_xlabel("Cell Area [km$^2$]")
    axes[0,1].set_title("Cell Size Distribution")
    
    samples_df["date"] = samples_df["date"].astype(np.datetime64)
    
    samples_df["date"].groupby(samples_df["date"].dt.month).count().plot(kind="bar",ax=axes[1,0],facecolor=(.7,.7,.7),
                                                                         alpha=0.75,grid=True)
    #axes[1,0].set_xlabel("Months")
    axes[1,0].set_xlabel("")
    axes[1,0].set_xticklabels(["Apr","May","Jun","Jul","Aug","Sep"],rotation=45)
    axes[1,0].set_title("Monthly Number of Cells")

    samples_df["date"].groupby([samples_df["date"].dt.month,
                                samples_df["date"].dt.day]).count().plot(kind="bar",
                                ax=axes[1,1],facecolor=(.7,.7,.7),alpha=0.75,edgecolor=(.7,.7,.7),grid=True)
    axes[1,1].get_xaxis().set_ticks([])
    axes[1,1].set_xlabel("Days over period")
    axes[1,1].set_title("Daily Number of Cells")
    
    samples_df["date"].groupby(samples_df["date"]).count().hist(ax=axes[2,0],bins=np.arange(0,150,10),
                                                                facecolor=(.7,.7,.7),alpha=0.75,grid=True)
    axes[2,0].set_xlabel("Number of cells")
    axes[2,0].set_title("Number of cells per time step")
    
    #samples_df["date"].loc[samples_df["RANKr"]>=1].groupby(samples_df["date"]).count().hist(ax=axes[2,1],bins=np.arange(0,65,5),
    #                                                            facecolor=(.7,.7,.7),alpha=0.75,grid=True)
    #axes[2,1].set_xlabel("Number of cells")
    #axes[2,1].set_title("Number of cells (TRT Rank >= 1)\n per time step")
    axes[2,1].axis('off')
    
    fig_hist.savefig(os.path.join(cfg_set_tds["fig_output_path"],u"TRT_Histogram.pdf"))

## Print map of TRT cells:
def print_TRT_cell_map(samples_df,cfg_set_tds):
    """Print map of TRT cells."""

    fig, axes = ccs4_map()
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "DEVELOPING"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "DEVELOPING"],c='w',edgecolor=(.7,.7,.7),s=18)
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "MODERATE"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "MODERATE"],c='g',edgecolor=(.7,.7,.7),s=22)
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "SEVERE"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "SEVERE"],c='y',edgecolor=(.7,.7,.7),s=26)
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "VERY SEVERE"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "VERY SEVERE"],c='r',edgecolor=(.7,.7,.7),s=30)
    
    fig.savefig(os.path.join(cfg_set_tds["fig_output_path"],u"TRT_Map.pdf"))
    
## Print map of TRT cells:
def ccs4_map(cfg_set_tds,figsize_x=12,figsize_y=12,hillshade=True,radar_loc=True,radar_vis=True):
    """Print map of TRT cells."""
    
    ## Load DEM and Swiss borders
    shp_path_CH      = os.path.join(cfg_set_tds["root_path"],u"data/shapefile/swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.shp")
    shp_path_Kantone = os.path.join(cfg_set_tds["root_path"],u"data/shapefile/swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.shp")
    shp_path_count   = os.path.join(cfg_set_tds["root_path"],u"data/shapefile/CCS4_merged_proj_clip_G05_countries.shp")
    dem_path         = os.path.join(cfg_set_tds["root_path"],u"data/DEM/ccs4.png")
    visi_path        = os.path.join(cfg_set_tds["root_path"],u"data/radar/radar_composite_visibility.npy")

    dem = Image.open(dem_path)
    dem = np.array(dem.convert('P'))

    sf_CH = shapefile.Reader(shp_path_CH)
    sf_KT = shapefile.Reader(shp_path_Kantone)
    sf_ct = shapefile.Reader(shp_path_count)

    ## Setup figure
    fig_extent = (255000,965000,-160000,480000)
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(figsize_x, figsize_y)
    
    ## Plot altitude / hillshading
    if hillshade:
        ls = LightSource(azdeg=315, altdeg=45)
        axes.imshow(ls.hillshade(-dem, vert_exag=0.05),
                    extent=fig_extent, cmap='gray', alpha=0.5)
    else:
        axes.imshow(dem*0.6, extent=fig_extent, cmap='gray', alpha=0.5)
        
    ## Get borders of Cantons
    try:
        shapes_KT = sf_KT.shapes()
    except UnicodeDecodeError:
        print("   *** Warning: No country shape plotted (UnicodeDecodeErrror)")
    else:
        for KT_i, shape in enumerate(shapes_KT):
            x = np.array([i[0] for i in shape.points[:]])
            y = np.array([i[1] for i in shape.points[:]])
            endpoint = np.where(x==x[0])[0][1]
            x = x[:endpoint]
            y = y[:endpoint]
            axes.plot(x,y,color='darkred',linewidth=1,zorder=5)

    ## Get borders of neighbouring countries
    try:
        shapes_ct = sf_ct.shapes()
    except UnicodeDecodeError:
        print("   *** Warning: No country shape plotted (UnicodeDecodeErrror)")
    else:
        for ct_i, shape in enumerate(shapes_ct):
            if ct_i in [0,1]:
                continue
            x = np.array([i[0] for i in shape.points[:]])
            y = np.array([i[1] for i in shape.points[:]])
            x[x<=255000] = 245000
            x[x>=965000] = 975000
            y[y<=-159000] = -170000
            y[y>=480000] = 490000
            if ct_i in [3]:
                axes.plot(x[20:170],y[20:170],color='black',linewidth=1)
            else:
                axes.plot(x,y,color='black',linewidth=1,zorder=4)

    ## Get Swiss borders
    try:
        #shp_records = sf_CH.shapeRecords()
        shapes_CH = sf_CH.shapes()
    except UnicodeDecodeError:
        print("   *** Warning: No country shape plotted (UnicodeDecodeErrror)")
    else:
        for ct_i, shape in enumerate(shapes_CH): #sf_CH.shapeRecords():
            if ct_i!=0: continue
            x = np.array([i[0]-2000000 for i in shape.points[:]])
            y = np.array([i[1]-1000000 for i in shape.points[:]])
            endpoint = np.where(x==x[0])[0][1]
            x = x[:endpoint]
            y = y[:endpoint]
            
            ## Convert to swiss coordinates
            #x,y = lonlat2xy(lon, lat)
            axes.plot(x,y,color='darkred',linewidth=2,zorder=3)

    ## Add weather radar locations:
    if radar_loc:
        weather_radar_y = [237000,142000,100000,135000,190000]
        weather_radar_x = [681000,497000,708000,604000,780000]
        axes.scatter(weather_radar_x,weather_radar_y,
                     color='orange',marker="D",zorder=10)
            
    ## Add radar visibility:
    if radar_vis:
        arr_visi = np.load(visi_path)
        arr_visi[arr_visi<9000] = 0
        arr_visi2 = morph.binary_opening(morph.binary_erosion(arr_visi, structure=np.ones((4,4))), structure=np.ones((4,4)))
        arr_visi[arr_visi<9000] = np.nan
        axes.imshow(arr_visi, cmap="gray", alpha=0.2, extent=fig_extent)
        arr_visi[np.isnan(arr_visi)] = 1
    #axes.contour(arr_visi[::-1,:], levels=[2], cmap="gray", linewidths=2,
    #             linestyle="solid", alpha=0.5, extent=fig_extent)
    #arr_visi = arr_visi[::4, ::4]
    #ys, xs = np.mgrid[arr_visi.shape[0]:0:-1,
    #                  0:arr_visi.shape[1]]
    #axes.scatter(xs.flatten(), ys.flatten(), s=4,
    #             c=arr_visi.flatten().reshape(-1, 3), edgecolor='face')
            
    ## Add further elements:
    axes.set_xlim([255000,965000])
    axes.set_ylim([-160000,480000])
    axes.grid()
    axes.set_ylabel("CH1903 Northing")
    axes.set_xlabel("CH1903 Easting")
    axes.get_xaxis().set_major_formatter( \
        ticker.FuncFormatter(lambda x, p: format(int(x), ",").replace(',', "'")))
    axes.get_yaxis().set_major_formatter( \
        ticker.FuncFormatter(lambda x, p: format(int(x), ",").replace(',', "'")))
    plt.yticks(rotation=90)
    return fig, axes, fig_extent

## Convert lat/lon-values in decimals to values in seconds:
def dec2sec(angles):
    """Convert lat/lon-values in decimals to values in seconds.
    
    Parameters
    ----------
    
    angles : list of floats
        Location coordinates in decimals.   
    """
    angles_ = np.zeros_like(angles)
    for i in range(len(angles)):
        angle = angles[i]
        ## Extract dms
        deg = float(str(angle).split(".")[0])
        min = float(str((angle - deg)*60.).split(".")[0])
        sec = (((angle - deg)*60.) - min)*60.
        angles_[i] = sec + min*60. + deg*3600.
    return angles_
    
## Convert lat/lon-values (in seconds) into LV03 coordinates:
def lonlat2xy(s_lon, s_lat): # x: easting, y: northing
    """Convert lat/lon-values (in seconds) into LV03 coordinates.
    
    Parameters
    ----------
    
    s_lon, s_lat : float
        Lat/Lon locations in seconds (not decimals!).   
    """
    # convert decimals to seconds...
    s_lon = dec2sec(s_lon)
    s_lat = dec2sec(s_lat)

    ## Auxiliary values 
    # i.e. differences of latitude and longitude relative to Bern in the unit [10000'']
    s_lng_aux = (s_lon - 26782.5)/10000.
    s_lat_aux = (s_lat - 169028.66)/10000.
    
    # easting
    s_x =   (600072.37 
        +  211455.93*s_lng_aux 
        -   10938.51*s_lng_aux*s_lat_aux 
        -       0.36*s_lng_aux*(s_lat_aux**2)  
        -      44.54*(s_lng_aux**3))
    
    # northing
    s_y =   (200147.07 
        + 308807.95*s_lat_aux 
        +   3745.25*(s_lng_aux**2) 
        +     76.63*(s_lat_aux**2) 
        -    194.56*(s_lng_aux**2)*s_lat_aux 
        +    119.79*(s_lat_aux**3))

    return s_x, s_y

    
    
    