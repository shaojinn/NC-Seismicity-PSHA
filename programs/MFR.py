#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate Magnitude-Frequency Relation (MFR) from synthetic/real data and find 
magnitude of completeness (Mc) and coresponding a-, b- values.

@author: shaojinn.chin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import geopandas
from shapely.geometry import Polygon
import copy


def mag_N(mode='synthetic', a=8, b=1, IncludeLargest=True, mBinWidth=0.1,
          filename='EarthquakeCatalog.csv', region=False):
    """
    Extract numbers of events at designed magnitude intervals from a real dataset 
    or generate synthetic ones from assigned a- and b-values. 
    Region of interest can be assigned when using a real dataset.

    Parameters
    ----------
    mode : string, optional
        Use 'synthetic' or 'realdata' dataset. The default is 'synthetic'.
    a : real, optional
        Used in synthetic mode. Assigned a-value. The default is 8.
    b : real, optional
        Used in synthetic mode. Assigned b-value. The default is 1.
    IncludeLargest : string, optional
        Used in synthetic mode. If True, force to have an event at M = a + 0.5.
        The default is True.
    mBinWidth : real, optional
        magnitude interval to calculate number of events. The default is 0.1.
    filename : string, optional
        Used in realdata mode. csv file contains earthquake catalogue.
        The default is 'earthquakeCatalog.csv'.
    region : string, optional
        Used in realdata mode to assign a region of interest and extract MFR 
        from the earthquakes in the region.
        If not given, Will use the whole dataset.
        The default is False.

    Returns
    -------
    M : array
        Magnitudes with at least one event.
    N : array
        Number of events at each magnitude interval.
    N_cumul : Array
        Cumulative number of events at each magnitude interval.

    """
    if(mode == 'synthetic'):

        # Assign the maximum magnitude = (a + MGreaterThan_a) 
        # if IncludeLargest = True
        if(IncludeLargest):
            MGreaterThan_a = 0.5
        else:
            MGreaterThan_a = 0

        ### Random missing data
        RandomMissingData = True
        MissingFactor = 1  # Linear amplification of percentage of missing data

        ### Create discrete number of event at each magnitude with real a, b and 
        ### assigned maximum magnitude
        Mmax = a + MGreaterThan_a 
        M = np.arange(Mmax,-0.1,-mBinWidth)
        Real_N_cumul = a - b * M
        Real_N = 10**Real_N_cumul[1:] - 10**Real_N_cumul[:-1]
        Real_N = np.flip(np.append(np.flip(Real_N),0))

        ### Create synthetic data through whole range of magnitude (0 - Mmax)
        if(RandomMissingData):
            random_missing_data = []
            for i in np.arange(0,len(M)):
                random_missing_data = (np.append(random_missing_data,
                                        np.random.triangular(-1,0,0) 
                                         * ((Mmax-M[i])/Mmax) * MissingFactor))
            syn_N = np.rint(Real_N + Real_N * random_missing_data)
        else:
            syn_N = Real_N
    
        if not IncludeLargest:
            sel = (syn_N > 0)
            syn_N = np.rint(syn_N[sel])
            M = M[sel]

        else:
            syn_N[0] = 1.0  ## Force to have an event with the largest magnitude
            sel = (syn_N > 0)
            syn_N = np.rint(syn_N[sel])
            M = M[sel]
                   
        M = np.flip(M)
        syn_N = np.flip(syn_N)
        syn_N_cumul = np.flip(np.log10(np.cumsum(syn_N)))
        return(M,syn_N,syn_N_cumul)
    
    elif(mode == 'realdata'):

        # read the catalog into a pandas dataframe
        df = pd.read_csv(filename,skiprows=5)
        # Create a geopandas dataframe using lon,lat values
        gdf = geopandas.GeoDataFrame(df,
              geometry=geopandas.points_from_xy(df.longitude, df.latitude))
        # load and define a closed polygon geographic region of interest 
        if region:
            polygonFilename = 'polygon_' + region + '.txt'
            poly_lonlat = np.loadtxt(polygonFilename)
            # Create shapely.geometry.polygon.Polygon object with topology 
            poly = Polygon(poly_lonlat)
            # Use geopandas clip function to extract gdf of events within the polygon
            gdf_clip = gdf.clip(poly)
        else:
            gdf_clip = gdf
            
        # Extract and bin magnitude values
        # extract values in polygon, sorted large to small (flipped)
        M = np.flip(np.sort(gdf_clip['magnitude'].values))
        # create magnitude bins with centres on multiples of magnitudeBinWidth
        # and min and max edges that contain all data
        mBinEdgeMin = M.min() - np.mod(M.min()+ 0.5*mBinWidth,mBinWidth)
        mBinEdgeMin = M.min() - 0.5 * mBinWidth
        mBinEdgeMax = M.max() - np.mod(M.max()+ 0.5*mBinWidth,mBinWidth) + mBinWidth
        mBinEdgeMax = M.max() + 0.5 * mBinWidth
        # Bin magnitude values - histogram
        mBins = np.arange(mBinEdgeMin,mBinEdgeMax+mBinWidth,mBinWidth)
        N,bins = np.histogram(M,mBins)
        binCenters = (mBins[1:] + mBins[:-1]) / 2
        # Cumulative sum N >= M ; need to reverse order of bins then flip back
        N_cumul = np.flip(np.cumsum(np.flip(N)))
        # Selection of values with number of events in bin > 0 and M<maxM
        NNotZero = (N > 0)
    
        N = N[NNotZero]
        M = binCenters[NNotZero]
        N_cumul = np.log10(N_cumul[NNotZero])
        return(M,N,N_cumul)
       
def syn_Mc(M,N,Mc=5.0,NEQ=False,a=False,b=False,randomMc=False):
    """
    Assign Mc to a synthetic MFR and randomly trim number of earthquake with 
    M < Mc.

    Parameters
    ----------
    M : array
        Magnitudes with at least one event.
    N : array
        Number of events at each magnitude interval.
    Mc : real, optional
        Assigned Mc. The default is 5.0.
    NEQ : integer, optional
        Assigned number of earthquakes to modeling. Mc is calculated 
        correspondingly if NEQ, a, b are given. The default is False.
    a : real, optional
        Real a-value in the synthetic catalogue. Must be given with NEQ and b
        to model the number of events. The default is False.
    b : real, optional
        Real b-value in the synthetic catalogue. Must be given with NEQ and a
        to model the number of events. . The default is False.
    randomMc : string, optional
        If True, randomly assign Mc with (Mmin + 0.5 <= Mc <= Mmax - 1). 
        The default is False.

    Returns
    -------
    M : array
        Magnitudes with at least one event.
    N : array
        Number of events at each magnitude interval.
    N_cumul : Array
        Cumulative number of events at each magnitude interval.

    """
    N_syn = copy.deepcopy(N)
    
    if NEQ and a and b:
        Mc = np.round((a - np.log10(NEQ)) / b,decimals=1)
        print('Mc= {:.1f}'.format(Mc))
    if randomMc:
        Mc = np.random.triangular(np.min(M)+0.5, np.mean(M), np.max(M)-1)
        print('Random Mc= {:.1f}'.format(Mc))
    #print("Mc={:.1f}".format(Mc))
    sel = (M < Mc)
    for i, v in enumerate(sel):
        if v:
            N_syn[i] = np.round(np.random.uniform(0,0.01**((Mc-M[i]))) * 
                                N_syn[i])
    sel = (N_syn > 0)
    M_syn = M[sel]
    N_syn = N_syn[sel]           
    N_cumul = np.flip(np.log10(np.cumsum(np.flip(N_syn))))
    return(M_syn,N_syn,N_cumul)

def fit_MFR(M,N,N_cumul,Mmax=10):
    """
    Fit magnitudes and numbers of events to find magnitude-frequency relation 
    (MFR) by using least-square solutions (LSQ) or Maximum Likelihood 
    Estimate (MLE, Aki, 1965; Utsu, 1965; 1966).

    Parameters
    ----------
    M : array
        Magnitudes.
    N : array
        Numbers of events at each magnitude interval.
    N_cumul : array
        Cumulative numbers of events at each magnitude interval.
    Mmax : real, optional
        Cutoff of the maximum magnitude. Magnitudes greater than Mamx will not 
        be used to fit. Set a reasonable value to avoid unusual big earthquakes
        or a very large one to include all events.
        The default is 10.

    Returns
    -------
    afit : array
        a-values at M assuming M is the magnitude of completeness of 
        the catalogue (Mc).
    bfit : array
        b-values at M assuming M is Mc.
    Rs : array
        goodness of fit at M assuming M is Mc (designed by Wiemer and Wyss, 2000).

    """
    afit = []
    bfit = []
    Rs = []    
    for minM in M:
        #minM = np.round(minM,decimals=3) # numerical precision problem fixed
        sel = (M >= minM) & (M <= Mmax)
        n_ind = np.where(M == minM)[0]
        ac = N_cumul[n_ind]
        afit = np.append(afit,ac)
        M_average = np.sum(M[sel] * N[sel]) / np.sum(N[sel])
        b_MLE = 1 / (np.log(10) * (M_average - (minM - 0.1 / 2)))
        bfit = np.append(bfit,b_MLE)
        n_predict = ac - b_MLE * (M[sel] - minM)
        R_MLE = 100 - 100 * ((np.sum(abs(10**N_cumul[sel] - 
                            10**n_predict))) / np.sum(10**N_cumul[sel]))
        Rs = np.append(Rs,R_MLE)
    return(afit,bfit,Rs)

def Mc_finder(a,b,M,Rs,threshold=90,R90=False):
    """
    Find best-fit MFR in a set of a-, b-values and goodness of fit.

    Parameters
    ----------
    a : array
        a-values at M assuming M is Mc.
    b : array
        b-values at M assuming M is Mc.
    M : array
        Magnitudes.
    Rs : array
        Goodness of fit (R) at M assuming M is Mc (designed by Wiemer and Wyss, 2000).
    threshold : real, optional
        Threshold of goodness of fit to decide if the fit is accepted. 
        Mc will be picked at M with maximum R if all of R < threshold.
        The default is 90.
    R90 : string, optional
        If True, use the original definition of acceptance.
        i.e. Mc at minimum R >= threshold (90 by Wiemer and Wyss, 2000). 
        If False, find Mc at the first pick of R and R >=threshold.
        i.e. Find peaks in Rs first, then find the first peak with R >= threshold
        to avoid potential underestimates reported by (Woessner and Wiemer, 2005).
        The default is False.

    Returns
    -------
    ac : real
        Best-fit a-value at Mc.
    bc : real
        Best-fit b-values.
    Mc : real
        Best-fit magnitude of completeness.
    Rc : real
        Goodness of fit at Mc (designed by Wiemer and Wyss, 2000).

    """
    
    foundMc = False
    if R90:
        for R in Rs:
            if R >= threshold and not foundMc:
                ind = [i for i,v in enumerate(Rs) if v == R][0]
                foundMc = True
                Rc = R
                ac = a[ind]
                bc = b[ind]
                Mc = M[ind]
    else:
        R_ind = find_peaks(Rs)[0]
        for ind in R_ind:
            if Rs[ind] >= threshold and not foundMc:
                foundMc = True
                Rc = Rs[ind]
                ac = a[ind]
                bc = b[ind]
                Mc = M[ind]
    # If all of R < threshold, pick Mc at M with maximum R 
    if not foundMc:
        Rc = np.max(Rs[:-2])
        ind = [i for i, v in enumerate(Rs) if v == Rc][0]
        ac = a[ind]
        bc = b[ind]
        Mc = M[ind]
    return(ac,bc,Mc,Rc)

def plot_MFR(M,N,N_cumul,ac,bc,Mc,Rc,mode=False,region=False,xmax=False,
             ymax=False):
    """
    Plot numbers of events at magnitudes and best-fit MFR.

    Parameters
    ----------
    M : array
        Magnitudes.
    N : array
        Numbers of events at each magnitude interval.
    N_cumul : array
        Cumulative numbers of events at each magnitude interval.
    ac : real
        Best-fit a-value at Mc.
    bc : real
        Best-fit b-values.
    Mc : real
        Best-fit magnitude of completeness.
    Rc : real
        Goodness of fit at Mc (designed by Wiemer and Wyss, 2000).
    mode : string, optional
        Label of data type. The default is False.
    region : string, optional
        Label of data type if used. The default is False.
    xmax : real, optional
        Maximum magnitude to plot. The default is False.
    ymax : TYPE, optional
        Maximum number of events to plot. The default is False.

    Returns
    -------
    None.

    """
    
    plt.figure(figsize=(6,4))
    plt.xlabel('Magnitude, M')
    plt.ylabel('Number of events, N')
    plt.yscale('log')
    if(mode): plt.plot([],[],' ',label="Mode: {:s}".format(mode))
    if(region): plt.plot([],[],' ',label="Region: {:s}".format(region))
    if ymax: plt.ylim(0.1,ymax)
    if xmax: plt.xlim(1,xmax)
    if (type(ac) is np.float64):
        N_predict_cumul = ac - bc * (M - Mc)
        plt.plot(M,10**N_predict_cumul,color='red',label='{:.2f} - '
             '{:.2f} * (M-Mc),\na0= {:.1f}, Mc= {:.1f}, Rc= {:.1f}'.
             format(ac,bc,ac+bc*Mc,Mc,Rc))
    if (type(ac) is np.ndarray):
        for i in range(ac.size):         
            N_predict_cumul = ac[i] - bc[i] * (M - Mc[i])
            plt.plot(M,10**N_predict_cumul,label='{:.2f} - {:.2f} *'
            ' (M-Mc)\na0= {:.1f}, Mc= {:.1f}, Rc= {:.1f}'.
            format(ac[i],bc[i],ac[i]+bc[i]*Mc[i],Mc[i],Rc[i]))
    plt.plot(M,10**N_cumul,marker='+',linestyle='',color='black')
    plt.bar(M,height=N,color='grey',width=0.1)
    plt.legend()

    

#------------ Functions for specific implementation ---------------------
def synthetic_test(a_ori=8,b_ori=1,IncludeLargest='True',R90='False'):
    """
    Plot a synthetic MFR and fit with LSQ and MLE 

    Parameters
    ----------
    a_ori : real, optional
        Assigned a-value. The default is 8.
    b_ori : real, optional
        Assigned b-value. The default is 1.
    IncludeLargest : string, optional
        Used in synthetic mode. If True, force to have an event at M = a + 0.5.
        The default is True. The default is 'True'.
    R90 : string, optional
        If True, use the original definition of acceptance.
        i.e. Mc at minimum R >= threshold (90 by Wiemer and Wyss, 2000). 
        If False, find Mc at the first pick of R and R >=threshold.
        i.e. Find peaks in Rs first, then find the first peak with R >= threshold
        to avoid potential underestimates reported by (Woessner and Wiemer, 2005).
        The default is 'False'.

    Returns
    -------
    None.

    """
# Plot a synthetic MFR and fit with LSQ and MLE    
    aclist = []
    bclist = []
    Mclist = []
    Rclist = []
    mode = 'synthetic'
    M,N,N_cumul = mag_N(mode=mode,a=a_ori,b=b_ori,IncludeLargest=IncludeLargest)

    M_syn,N_syn,N_cumul = syn_Mc(M,N,randomMc=True)
    
    a,b,Rs = fit_MFR(M_syn,N_syn,N_cumul)
    ac,bc,Mc,Rc = Mc_finder(a, b, M_syn, Rs,R90=True)
    aclist = np.append(aclist,ac)
    bclist = np.append(bclist,bc)
    Mclist = np.append(Mclist,Mc)
    Rclist = np.append(Rclist,Rc)
    a,b,Rs = fit_MFR(M_syn,N_syn,N_cumul)
    ac,bc,Mc,Rc = Mc_finder(a, b, M_syn, Rs,R90=False)
    aclist = np.append(aclist,ac)
    bclist = np.append(bclist,bc)
    Mclist = np.append(Mclist,Mc)
    Rclist = np.append(Rclist,Rc)
    plot_MFR(M_syn,N_syn,N_cumul,aclist,bclist,Mclist,Rclist,mode=mode)

# Plot comparison of sample sizes and a, b, Mc, R that are revealed by the 4 
# combinations of fit types and definitions of R. 
# (i.e. MLE90, MLEpeak, LSQ90, LSQpeak.)
    MLE90 = [[0,0,0,0]]
    MLEpeak = [[0,0,0,0]]
    N_cumuls = []
    mmin = 3.5
    mmax = 6.6
    for MMc in np.arange(mmin,mmax,0.2):
        M_syn,N_syn,N_cumul = syn_Mc(M,N,Mc=MMc)
        ind = [i for i, v in enumerate(M_syn) if np.round(v,decimals=3) == 
               np.round(MMc,decimals=3)][0]
        N_cumuls = np.append(N_cumuls,N_cumul[ind])
        a,b,Rs = fit_MFR(M_syn,N_syn,N_cumul)
        ac,bc,Mc,Rc = Mc_finder(a, b, M_syn, Rs,R90=False)
        MLEpeak.extend(([[ac,bc,Mc,Rc]]))
        ac,bc,Mc,Rc = Mc_finder(a, b, M_syn, Rs,R90=True)
        MLE90.extend(([[ac,bc,Mc,Rc]]))
    MLEpeak = np.array(MLEpeak[1:])
    MLE90 = np.array(MLE90[1:])

    plt.figure(figsize=(6,4))
    plt.xlabel('Mc')
    plt.ylabel('a0')
    plt.yscale('linear')
    plt.ylim(6.5,9)
    plt.axhline(y=a_ori,color='gray',linestyle='--')
    plt.plot(a_ori-N_cumuls,MLEpeak[:,0] + MLEpeak[:,1] * MLEpeak[:,2],
             color='red',label='MLEpeak')
    plt.plot(a_ori-N_cumuls,MLE90[:,0] + MLE90[:,1] * MLE90[:,2],color='blue',
             linestyle='-',label='MLE90')
    plt.legend()
    
    plt.figure(figsize=(6,4))
    plt.xlabel('Mc')
    plt.ylabel('b')
    plt.yscale('linear')
    plt.ylim(0.8,1.1)
    plt.axhline(y=b_ori,color='gray',linestyle='--')
    plt.plot(a_ori-N_cumuls,MLEpeak[:,1],color='red',label='MLEpeak')
    plt.plot(a_ori-N_cumuls,MLE90[:,1],color='blue',linestyle='-',label='MLE90')
    plt.legend()

    plt.figure(figsize=(6,4))
    plt.xlabel('True Mc')
    plt.ylabel('Predicted Mc')
    plt.yscale('linear')
    plt.ylim(mmin-0.2,mmax+0.2)
    plt.plot(a_ori-N_cumuls,MLEpeak[:,2],color='red',label='MLEpeak')
    plt.plot(a_ori-N_cumuls,MLE90[:,2],color='blue',linestyle='-',label='MLE90')
    plt.plot(a_ori-N_cumuls,np.arange(mmin,mmax,0.2),color='black',
             linestyle='--')
    plt.legend()
    
    plt.figure(figsize=(6,4))
    plt.xlabel('Mc')
    plt.ylabel('Rc')
    plt.yscale('linear')
    plt.ylim(80,100)
    plt.plot(a_ori-N_cumuls,MLEpeak[:,3],color='red',label='MLEpeak')
    plt.plot(a_ori-N_cumuls,MLE90[:,3],color='blue',linestyle='-',label='MLE90')
    plt.legend()

def fit_SNC(region='SNC',R90=False,plot_fig=False):
    """
    Fit MFR in SNC and NHV with ITOPNC catalogue.

    Parameters
    ----------
    region : string, optional
        SNC or NHV. The default is 'SNC'.
    R90 : string, optional
        If True, use the original definition of acceptance.
        i.e. Mc at minimum R >= threshold (90 by Wiemer and Wyss, 2000). 
        If False, find Mc at the first pick of R and R >=threshold.
        i.e. Find peaks in Rs first, then find the first peak with R >= threshold
        to avoid potential underestimates reported by (Woessner and Wiemer, 2005).
        The default is False.
    plot_fig : string, optional
        If True, plot numbers and cumulative numbers of earthquakes at each 
        magnitude interval and best-fit MFR.
        The default is False.

    Returns
    -------
    a0 : float
        a-value at M = 0.
    bc : float
        b-value

    """
    mode = 'realdata'
    M,N,N_cumul = mag_N(mode=mode,region=region)
    a,b,Rs = fit_MFR(M, N, N_cumul)
    ac,bc,Mc,Rc = Mc_finder(a, b, M, Rs,R90=False,threshold=95)
    a0 = ac + bc * Mc
    np.savetxt('MFR_' + region +'.txt', np.transpose([M,N,N_cumul,a,b,Rs]),
               header='M N N_cumul a b R',fmt='%.3f')
    if plot_fig: plot_MFR(M, N, N_cumul, ac, bc, Mc, Rc, mode=mode, 
                          region=region)
    
    return(a0,bc)
    
if __name__ == '__main__':
    
    synthetic_test()
    fit_SNC('SNC',plot_fig=True)
    fit_SNC('NHV',plot_fig=True)