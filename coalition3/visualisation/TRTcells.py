""" [COALITION3] Plotting locations of TRT cells within training dictionary
    and histograms of different TRT statistics"""

from __future__ import division
from __future__ import print_function

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from PIL import Image
import shapefile


## =============================================================================
## FUNCTIONS:
             
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
    ## Load DEM and Swiss borders
    
    shp_path = "%s%s" % (os.path.join(cfg_set_tds["root_path"],
                         u"data/shapefile/Shapefile_and_DTM/CHE_adm0.shp"))
    #shp_path = "%s%s" % (cfg_set_tds["CONFIG_PATH_set_train"],"Shapefile_and_DTM/CCS4_merged_proj_clip_G05_countries.shp")
    dem_path = "%s%s" % (os.path.join(cfg_set_tds["root_path"],u"data/DEM/ccs4.png"))

    dem = Image.open(dem_path)
    dem = np.array(dem.convert('P'))

    sf = shapefile.Reader(shp_path)
    # for shape in sf.shapeRecords():
        # x = [i[0] for i in shape.shape.points[:]]
        # y = [i[1] for i in shape.shape.points[:]]
        # plt.plot(x,y)
       
    fig_map, axes = plt.subplots(1, 1)
    fig_map.set_size_inches(12, 12)
    axes.imshow(dem, extent=(255000,965000,-160000,480000), cmap='gray')
        
    ## Plot in swiss coordinates (radar CCS4 in LV03 coordinates)
    for shape in sf.shapeRecords():
        lon = [i[0] for i in shape.shape.points[:]]
        lat = [i[1] for i in shape.shape.points[:]]
        
        ## Convert to swiss coordinates
        x,y = lonlat2xy(lon, lat)
        #x = lon
        #y = lat
        axes.plot(x,y,color='b',linewidth=1)
        
    ## Convert lat/lon to Swiss coordinates:
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "DEVELOPING"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "DEVELOPING"],c='w',edgecolor=(.7,.7,.7),s=18)
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "MODERATE"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "MODERATE"],c='g',edgecolor=(.7,.7,.7),s=22)
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "SEVERE"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "SEVERE"],c='y',edgecolor=(.7,.7,.7),s=26)
    axes.scatter(samples_df["LV03_x"].loc[samples_df["category"] == "VERY SEVERE"],
                 samples_df["LV03_y"].loc[samples_df["category"] == "VERY SEVERE"],c='r',edgecolor=(.7,.7,.7),s=30)
    axes.set_xlim([255000,965000])
    axes.set_ylim([-160000,480000])
    fig_map.savefig(os.path.join(cfg_set_tds["fig_output_path"],u"TRT_Map.pdf"))

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

    
    
    