#Code by J. Mackrory to make "quick" heatmaps for latitude/longitude data.
# Specialized to ODOT crash data gathered for PDSG by Seth Urbach.
#
#Assumes pandas dataframes with 'Latitude minutes' etc
#as columns.
#Lots of longwindedness for simple idea:
# 1) convert degrees-minutes-seconds to decimal degrees.
# 2) make 2D array for lat/lon.
# 3) count up number of events in each bin.
#    (really include a smoothing kernel)
# 4) make a "pretty" plot with log10(x+1) scaling.
#
# N.B. Approximate zero error checking here.
#
# Can adapt input dataframes to select out certain conditions.
# Could also restrict xydict to certain lat/lons, like downtown.
# Or use something less sucky like Plotly/Bokeh/Gmaps for actual
# maps?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

def dec_degree(x):
    """dec_degree
    Converts 2d numpy array to decimal degrees.
    Assumes 0 column is degree
    1  column is minutes
    2 column is second.
    Outputs 1D numpy array.
    """
    return x[:,0]+np.sign(x[:,0])*(x[:,1]/60.0+x[:,2]/3600.0)

def make_latlon(df):
    """make_latlon(df)
    Converts latitude/longitude to decimal degrees.
    Only keeps actual values.
    
    Input: df - pandas dataframe
    Output lats,lons - numpy arrays with latitude/longitude
    
    """
    lat_list=['Latitude Degrees',
    'Latitude Minutes',
    'Latitude Seconds']

    lon_list=['Longitude Degrees',
    'Longitude Minutes',
    'Longitude Seconds']
    msk1=df['Latitude Degrees'].astype(str).str.contains(r'[0-9]')
    lons=df.loc[msk1,lon_list].values.astype(float)
    lats=df.loc[msk1,lat_list].values.astype(float)
    msk=(lons[:,1]==lons[:,1])
    lons=dec_degree(lons[msk])
    lats=dec_degree(lats[msk])
    return lats,lons

def make_xydict(lats,longs,Nx=1000):
    """make_xydict
    Make dict to store max/min  for x/y and number of x points. 
    Can then exactly reconstruct grid, and ranges
    even for different data.
    """
    #find 99% percentile edges
    xd={}
    xd['xmin']=np.percentile(longs,0.5);
    xd['xmax']=np.percentile(longs,99.5);

    xd['ymin']=np.percentile(lats,0.5);
    xd['ymax']=np.percentile(lats,99.5);
    xd['Nx']=Nx;
    xd['dx']=(xd['xmax']-xd['xmin'])/Nx
    xd['Ny']=int((xd['ymax']-xd['ymin'])/xd['dx'])
    
    return xd

@jit
def make_eff_heatmap(lats,longs,xd,sigma_fac=1):
    """make_eff_heatmap

    Make a heatmap with a given kernel for smearing
    given a list of lats/longitudes.  

    Input: lats - numpy array of decimal latitudes
           longs - numpy array of decimal longitudes
           xd - dict containing max/min values, Nx and dx for X(lon) and Y(lat) dimensions
           sigma_fac - number of bins for blurring
    Output: heatmap - (Nx, Ny) numpy array 
    """
    Nv = len(lats)

    pad = sigma_fac*3;

    Nx_tot=xd['Nx']+2*pad;    
    Ny_tot=xd['Ny']+2*pad;

    #Make the kernel to add up everywhere.
    Np=2*pad+1;
    kernel=np.zeros((2*pad+1,2*pad+1));
    mu=pad+1
    for i in range(Np):
        for j in range(Np):
            kernel[i,j]=np.exp(-((i-mu)**2+(j-mu)**2)/(2*sigma_fac**2));
    
    #make the heatmap
    heat_tot = np.zeros((Nx_tot,Ny_tot));
    for i in range(Nv):
        #find the pixel corresponding to upper edge.
        #Note max/min for x/y.  Assuming (0,0) is upper left.
        xi = int((longs[i]-xd['xmin'])/xd['dx'])+pad;
        yi = int((xd['ymax']-lats[i])/xd['dx'])+pad;
        #check that box falls inside allowed region.
        if ((xi>0) & ((xi+Np)<Nx_tot) & (yi>0) & ((yi+Np)<Ny_tot)):
           heat_tot[xi:xi+Np, yi:yi+Np]+=kernel
    #now remove the padding so max/min are as specified.
    #hopefully counting is not off by one.
    #Take transpose because of way python counts, so y goes along rows, x along columns.
    heat_tot=heat_tot[pad:-pad,pad:-pad].T
    return heat_tot

def make_heatmap_plot(heat,xd,title_str,savename=None):
    """make_heatmap_plot
    Now plot that heatmap.
    Use custom xticks/yticks.
    Can also save pdf if uncomment that line.

    Inputs: heat - 2d numpy array
    xd - dict with x/y parameters
    title_str - string 

    Outputs: none

    Side effect: numpy plot.  Can save pdf too.

    """
    plt.figure(figsize=(20,10))
    #shift by one so that zero counts return zero, since log(1)=0.
    plt.imshow(np.log10(heat+1),cmap=plt.cm.hot)
    #Make ticks to 4d.p.
    #Could just choose round numbers, but meh to that. 
    fmt_4dp= lambda y: list(map(lambda x: '{:.4f}'.format(x),y))
    xticks_bin=np.linspace(0,heat.shape[1],5)
    xticks_val=np.linspace(xd['xmin'],xydict['xmax'],5)
    yticks_bin=np.linspace(0,heat.shape[0],5)
    yticks_val=np.linspace(xd['ymax'],xydict['ymin'],5)
    plt.xticks(xticks_bin,fmt_4dp(xticks_val),rotation='vertical')
    plt.yticks(yticks_bin,fmt_4dp(yticks_val))
    #Labelling
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title_str)
    plt.colorbar()
    if (savename is not None):
        plt.savefig(savename)
    plt.show()

#Commands to actually run this thing.
#df_tot is some pandas dataframe with latitude/longitude columns.
def run_heatmap(df):
    lats,lons=make_latlon(df)
    xd=make_xydict(lats,lons,Nx=1000)
    heat = make_eff_heatmap(lats,lons,xd,sigma_fac=1)
    t1=time.time()
    print('time taken',t1-t0)
    make_heatmap_plot(heat,xd,'Log10-number of ALL accidents from 2012-2015 in Portland Area')    
