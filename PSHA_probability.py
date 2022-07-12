#!/usr/bin/env python
# python3
"""
    Probabilistic seismic hazard analysis.
    For Noumea, consider local sources within 250 km and regional sources from 
    southern New Hebrides-Vanuatu subduction zone.
    
    @author: shaojinn.chin
"""
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod
from shapely.geometry import Polygon
import geopandas
import pandas
from matplotlib import rc
rc("pdf", fonttype=42)
#rc("font",size=10,family='Arial')
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True

import GMPE 
import MFR

def a_normalise(a,time_ref,area_ref=False,R_max=200,method='circular'):
    """
    Normalise a-value with area and time

    Parameters
    ----------
    a : float
        Original a-value.
    time_ref : float
        Time for obtaining the original a-value (year).
    area_ref : float, optional
        Area for obtaining the original a-value (km**2).
        The default is False.
    R_max : float, optional
        Radius of area of interest (km). Only used in 'circular' method.
        The default is 200.
    method : string, optional
        circular, linear, or from_file. method to normalise area. Area will be
        calculated only if method = circular, otherwise will be assummed 
        identical to area_ref. The default is circular.

    Returns
    -------
    a : float
        Adjusted a-value.

    """
    a_origin = a
    if method == 'linear' or method == 'from_file' or method == 'poly':
        area_ref = 1
        a = a + np.log10((area_ref * 1) / (area_ref * time_ref))
    elif method == 'circular' and (area_ref):
        a = a + np.log10((R_max**2 * np.pi * 1) / (area_ref * time_ref))
        
    print(('a-value is normalised by area and time to %.2f' % a + ' (was %.2f)' 
           % a_origin))
    
    return(a)

def prob_mag(mag,M_step,Year=1,alpha=1000,beta=np.log(10),a=False,b=False):
    """
    Occurrence probability of magnitude (Epstein and Lomnitz, 1966)
    Occurrence prob. (mag) = Exceedance prob.(mag) - Exceedance prob.(mag+M_step)
    The default is to return probability with alpha = 1000 and beta = ln(10) 
    that is equivalent to a = 4, b = 1.0 (log(N) = 4 - 1 * M).
    
    Parameters
    ----------
    mag : numpy array
        Magnitude of interest.
    M_step : float
        Interval of magnitudes. Must equal to the interval in mag array or the 
        resulting probability will be overlapped and higher than expected.
    Year : TYPE, optional
        Time period of interest (year). The default is 1 and should be 1 to 
        calculate "Annual probability".Still can be changed.
    alpha : TYPE, optional
        alpha either assigned or converted from a if given. 
        The default is 1000.
    beta : TYPE, optional
        beta either assigned or converted from b if given. 
        The default is np.log(10).
    a : TYPE, optional
        a-value in Magnitude-frequency relation (GR relation). 
        The default is False.
    b : TYPE, optional
        b-value in Magnitude-frequency relation (GR relation). 
        The default is False.

    Returns
    -------
    prob_mag : numpy array
        Occurrence probability of magnitude.

    """
    ab = a * b
    if ab:
        alpha = np.power(10,a)
        beta = b * np.log(10)

    prob_mag = 1-np.exp(-Year * alpha * np.exp(-beta*mag))
    prob_mag_one_step  = 1-np.exp(-Year * alpha * np.exp(-beta*(mag+(M_step))))
    prob_mag = prob_mag - prob_mag_one_step

    return prob_mag

def prob_r(R_min=0,R_max=200,filename=False,method='circular'):
    """
    Obtain matrix R and prob_R in a circular area with a radius of R_max, or
    in a designed area, which used to obtain a-value, with a read-in file
    containing pre-calculated probabilities.

    Parameters
    ----------
    R_min : float, optional
        Minimum distance of interest. The default is 0.
        Used if method == circular or linear
    R_max : float, optional
        Maximum distance of interest or radius of a circular area of interest. 
        The default is 200.
        Used if method == circular or linear
    filename : string, optional
        Read-in file with distance and pre-calculated probability. 
        The default is False.
        Used if method == from_file
    method : string, optional
        Method to calculate probability of distance (r).
        Aveliable methods: circular, linear, from_file. 

    Returns
    -------
    R : numpy array
        Distances within area of interest.
    R_prob : numpy array
        Probability of distances within area of interest.
    R_min : float
        Minimum distance within area of interest.
    R_max : numpy array
        Maximum distances within area of interest.
    nR : integral
        Number of elements in R.

    """
    
    if method == 'from_file':
        # Read Probability of distance from a file 
        # File format: Distance(km) Probability(0-1)
        R = np.loadtxt(filename)
        R_prob = R[:,1]
        R = R[:,0]
        R_min = R[0]
        R_max = R[-1]
        nR = len(R)
    elif method == 'circular':
        if R_min == 0: R_min = 1
        R = np.arange(R_min,R_max+1,1)
        R_prob = []
        for i in R:
            R_prob = np.append(R_prob, i / (R_max**2 / 2))
        nR = len(R)
    elif method == 'linear':
        R = np.arange(R_min,R_max+1,1)
        nR = len(R)
        R_prob = np.ones(nR) * (1/nR)
        
    return(R,R_prob,R_min,R_max,nR)

def prob_r_poly(lon_ref,lat_ref,filename=False,poly_lonlat=False,gridX=1,
                gridY=1,dist_step=1,geo_ref="WGS84"):
    """
    Obtain matrix R and prob_R from a polygon area to a reference point.   

    Parameters
    ----------
    lon_ref : float
        Longitude of the reference point.
    lat_ref : float
        Latitude of the reference point.
    filename : string, optional
        File name to read the points of a polygon. The default is False.
    poly_lonlat : np.array or list, optional
        Array or list of the points of a polygon. The default is False.
    gridX : float, optional
        Grid size along longitude in km. The default is 1.
    gridY : float, optional
        Grid size along latitude in km. The default is 1.
    dist_step : float, optional
        Stop of distance in km to calculate probability. The default is 1.
    geo_ref : string, optional
        Geoid to calculate distance. The default is "WGS84".

    Returns
    -------
    R : numpy array
        Distances within area of interest.
    R_prob : numpy array
        Probability of distances within area of interest.
    R_min : float
        Minimum distance within area of interest.
    R_max : numpy array
        Maximum distances within area of interest.
    nR : integral
        Number of elements in R.

    """
    geo_ref = "+ellps=" + geo_ref
    geod = Geod(geo_ref)
    if filename:
        poly_lonlat = np.loadtxt(filename)
        poly = Polygon(poly_lonlat)
    elif poly_lonlat:
        poly = Polygon(poly_lonlat)
    
    lonW = poly_lonlat[:,0].min()-0.1
    lonE = poly_lonlat[:,0].max()+0.1
    latN = poly_lonlat[:,1].max()+0.1
    latS = poly_lonlat[:,1].min()-0.1
    lons = [lonW,lonW]
    lats = [latN,latS]
    npt_lat = (int(geod.line_length(lons, lats) / 1000)) / gridY
    lons = [lonW,lonE]
    grids = [(lonW,latN)]
    for WW in geod.npts(lonW,latN,lonW,latS,npt_lat):
        lats = [WW[1],WW[1]]
        npt_lon = (int(geod.line_length(lons, lats) / 1000)) / gridX
        grids = np.append(grids,geod.npts(lonW,WW[1],lonE,WW[1],npt_lon),axis=0)
    
    df = pandas.DataFrame({'lon' : [i[0] for i in grids],
                       'lat' : [i[1] for i in grids]})
    gdf = geopandas.GeoDataFrame(df,
                                 geometry=geopandas.points_from_xy(df.lon, df.lat))
    gdf_clip = gdf.clip(poly)
    refpt = np.zeros((len(gdf_clip),2))
    refpt[:,0] = lon_ref
    refpt[:,1] = lat_ref
    az1,az2,dist = geod.inv(refpt[:,0],refpt[:,1],gdf_clip.lon,gdf_clip.lat)
    dist = np.rint(dist / 1000)
    Dbins = np.arange(np.rint(dist.min())-1.5,np.rint(dist.max())+1.5,dist_step)
    R_prob,bins = np.histogram(dist,Dbins)
    R = (bins[1:] + bins[:-1]) / 2
    R_prob = R_prob / sum(R_prob)
    return(R,R_prob,R.min(),R.max(),len(R))

def Prob_occurrence(M_min,M_max,M_step,alpha,beta,Depth,R,R_prob,D_max=False):
    """
    Create matrices of Magnitude(M), Distance(r) and Depth(d), and calculate 
    probability of earthquake occurrence with magnitudes at a range 
    of distance and depth (Prob_Mdr)
    
    Magnitude(M), Distance(r), and Depth(d) matrices
    Example of Magnitude(M), Distance(r), and Depth(d) matrices:
            M       r       d       Prob_Mdr
            4       1       1       Prob_M(4) * prob_r(1) * prob_d(1)
            4       1       2       Prob_M(4) * prob_r(1) * prob_d(2)
            4       1       3       Prob_M(4) * prob_r(1) * prob_d(3)
            4       2       1       Prob_M(4) * prob_r(2) * prob_d(1)
            4       2       2       Prob_M(4) * prob_r(2) * prob_d(2)
            4       2       3       Prob_M(4) * prob_r(2) * prob_d(3)
            5       1       1       :
            5       1       2       :
            5       1       3       :
            5       2       1       :
            5       2       2       :
            5       2       3       :
            :       :       :       :

    Parameters
    ----------
    M_min : float
        Minimum magnitude.
    M_max : float
        Maximum magnitude.
    M_step : float
        Magnitude interval.
    alpha : float
        alpha converted from a-value.
    beta : float
        beta converted from a-value.
    Depth : numpy array
        Depth of interest (km).
    R : numpy array
        Distance of interest (km).
    R_prob : numpy array
        Distance probability (0-1).

    Returns
    -------
    prob_Mdr : numpy array
        Occurrence probability of an earthquake with magnitude M at distance r 
        and depth d.
    M : numpby array
        Earthquake magnitude.
    d : numpby array
        Depth (km).
    r : numpby array
        Distance (km).
    

    """
    nM = len(np.arange(M_min,M_max+M_step,M_step))
    nR = len(R)
    nDepth = len(Depth)
    
    M = np.arange(M_min,M_max+0.1,0.1)
    M_ori = M
    for i in np.arange(1,nR * nDepth,1):
        M = np.append(M, M_ori)
    M.sort()
    d = Depth
    for i in np.arange(1,nM * nR,1):
        d = np.append(d, Depth)
    r = R
    for i in np.arange(1,nDepth,1):
        r = np.append(r,R)
    r.sort()
    r_to_append = r
    for i in np.arange(1,nM,1):
        r = np.append(r,r_to_append)
## Check the length of matrices M, r, d are identical
    if len(M) == len(r) and len(M) == len(d):
        N_events = len(M)
    else:
        print('Noumber of events is incorrect')

## Probability matrices of M, r, and d
    prob_M = prob_mag(M, M_step, alpha=alpha, beta=beta)
    prob_d = np.ones(len(d)) / nDepth
    prob_r = []
    for i in r:
        prob_r = np.append(prob_r,R_prob[int(i-R[0])])
    prob_Mdr = prob_M * prob_d * prob_r
    return(prob_Mdr, M, d, r)

def prob_pga_ASC(M,r,d,prob_Mdr,V30=850,outProb='individual'):
    """
    Combine probabilities of annual earthquake occurrence and of exceedance of 
    pga to calculate annual exceedance probability of PGA with GMPEs:
    ZH06: Zhao et al. (2006)
    BA08: Boore and Atkinson (2008)
    CY08: Chiou and Youngs (2008)

    Parameters
    ----------
    M : numpy array
        Magnitude.
    r : numpy array
        Distance (km).
    d : numpy array
        Depth (km).
    prob_Mdr : numpy array
        Occurrence probability of an earthquake with magnitude M at distance r 
        and depth d.
    outProb : string, optional
        Output type of probability. 
        If individual, output three matrices that are probability calculated 
        with three GMPEs, respectively.
        If combined, output one matrix that combines weighted results from the 
        three individual matrices.
        The default is individual.        

    Returns
    -------
    prob_pgaZH06 : numpy array
        Probability calculated with Zhao et al. (2006) GMPE.
    prob_pgaBA08 : numpy array
        Probability calculated with Boore and Atkinson (2008) GMPE.
    prob_pgaCY08 : numpy array
        Probability calculated with Chiou and Youngs (2008) GMPE.
    prob_combined : numpy array
        Weighted probability (0.4 * ZH06 + 0.3 * BA08 + 0.3 * CY08)
    """
    
    ## Compute pga for each event and convert unit from m/s/s to g
    lnpgaZH06,sigmaZH06 = GMPE.lnpgaZH06(M,r,depth=d)
    lnpgaBA08,sigmaBA08 = GMPE.lnpgaBA08(M,r,V30=V30)
    lnpgaCY08,sigmaCY08 = GMPE.lnpgaCY08(M,r,depth=d,V30=V30)
    ## Convert to g
    lnpgaZH06 = lnpgaZH06 - np.log(9.8) 
    lnpgaBA08 = lnpgaBA08 - np.log(9.8) 
    lnpgaCY08 = lnpgaCY08 - np.log(9.8) 

    ## Calculte probability of exceedance of PGA
    lnpga_threshold = np.log(np.power(10,np.arange(-2,1.8,0.1)))
    pga = np.exp(lnpga_threshold)
    prob_pgaZH06 = []
    prob_pgaBA08 = []
    prob_pgaCY08 = []
    for i in lnpga_threshold[:]:
        prob = GMPE.prob_exceedance(i, lnpgaZH06, sigmaZH06)
        prob = prob * prob_Mdr
        prob_pgaZH06 = np.append(prob_pgaZH06,np.sum(prob))
        prob = GMPE.prob_exceedance(i, lnpgaBA08, sigmaBA08)
        prob = prob * prob_Mdr
        prob_pgaBA08 = np.append(prob_pgaBA08,np.sum(prob))
        prob = GMPE.prob_exceedance(i, lnpgaCY08, sigmaCY08)
        prob = prob * prob_Mdr
        prob_pgaCY08 = np.append(prob_pgaCY08,np.sum(prob))
    prob_combined = prob_pgaZH06 * 0.4 + prob_pgaBA08 * 0.3 + prob_pgaCY08 * 0.3 
    
    if outProb == 'individual':
        return(pga,prob_pgaZH06,prob_pgaBA08,prob_pgaCY08)
    elif outProb == 'combined':
        return(pga,prob_combined)
    
def prob_pga_SUB(M,r,d,prob_Mdr,V30=850,outProb='individual'):
    """
    Combine probabilities of annual earthquake occurrence and of exceedance of 
    pga to calculate annual exceedance probability of PGA with GMPEs:
    ZH06: Zhao et al. (2006)
    BA08: Boore and Atkinson (2008)
    CY08: Chiou and Youngs (2008)

    Parameters
    ----------
    M : numpy array
        Magnitude.
    r : numpy array
        Distance (km).
    d : numpy array
        Depth (km).
    prob_Mdr : numpy array
        Occurrence probability of an earthquake with magnitude M at distance r 
        and depth d.
    outProb : string, optional
        Output type of probability. 
        If individual, output three matrices that are probability calculated 
        with three GMPEs, respectively.
        If combined, output one matrix that combines weighted results from the 
        three individual matrices.
        The default is individual.        

    Returns
    -------
    prob_pgaZH06 : numpy array
        Probability calculated with Zhao et al. (2006) GMPE.
    prob_pgaBA08 : numpy array
        Probability calculated with Boore and Atkinson (2008) GMPE.
    prob_pgaCY08 : numpy array
        Probability calculated with Chiou and Youngs (2008) GMPE.
    prob_combined : numpy array
        Weighted probability (0.4 * ZH06 + 0.3 * BA08 + 0.3 * CY08)
    """
    
    ## Compute pga for each event and convert unit from m/s/s to g
    lnpgaAB03,sigmaAB03 = GMPE.lnpgaAB03(M,r,depth=d)
    lnpgaAn16,sigmaAn16 = GMPE.lnpgaAn16(M,r,depth=d,V30=V30)
    lnpgaYs97,sigmaYs97 = GMPE.lnpgaYs97(M,r,depth=d)
    ## Convert to g
    lnpgaAB03 = lnpgaAB03 - np.log(9.8) 
    lnpgaAn16 = lnpgaAn16 - np.log(9.8) 
    lnpgaYs97 = lnpgaYs97 - np.log(9.8) 

    ## Calculte probability of exceedance of PGA
    lnpga_threshold = np.log(np.power(10,np.arange(-2,1.8,0.1)))
    pga = np.exp(lnpga_threshold)
    prob_pgaAB03 = []
    prob_pgaAn16 = []
    prob_pgaYs97 = []
    for i in lnpga_threshold[:]:
        prob = GMPE.prob_exceedance(i, lnpgaAB03, sigmaAB03)
        prob = prob * prob_Mdr
        prob_pgaAB03 = np.append(prob_pgaAB03,np.sum(prob))
        prob = GMPE.prob_exceedance(i, lnpgaAn16, sigmaAn16)
        prob = prob * prob_Mdr
        prob_pgaAn16 = np.append(prob_pgaAn16,np.sum(prob))
        prob = GMPE.prob_exceedance(i, lnpgaYs97, sigmaYs97)
        prob = prob * prob_Mdr
        prob_pgaYs97 = np.append(prob_pgaYs97,np.sum(prob))
    prob_combined = (prob_pgaAB03 * 0.33 + prob_pgaAn16 * 0.34 + 
                     prob_pgaYs97 * 0.33) 
    
    if outProb == 'individual':
        return(pga,prob_pgaAB03,prob_pgaAn16,prob_pgaYs97)
    elif outProb == 'combined':
        return(pga,prob_combined)

def PoE(region=False,V30=760,M_step=0.1,M_min=4.5,M_max=7.5,a=8,b=1,time_ref=1,
        area_ref=250**2 * np.pi,R_max=250,D_max=20,method='circular',
        filename=False,lon_ref=0,lat_ref=0,weighted=True,tectonic='ASC'):
    """
    Calculate probability of exceedance (PoE).

    Parameters
    ----------
    region : TYPE, optional
        Region of interest. 
        Options are 'SNC' or 'NHV'. 
        Using 'SNC' will give a result from ITOPNC catalogue.
        Using 'NHV' will give one from USGS catalogue with an averaged a-value 
        in 2001 - 2020.
        The default is False.
    a : float, optional
        a- value. Activated when region == False. The default is 8.
    b : float, optional
        b-value. Activated when region == False. The default is 1.
    V30 : float, optional
        Averaged shear velocity in the upper 30 m. The default is 760.
    M_step : float, optional
        Magnitude interval from M_min to M_max. The default is 0.1.
    M_min : float, optional
        Minimum magnitude of interest. Used when region == False.
        The default is 4.5.
    M_max : float, optional
        Maximum magnitude of interest. Used when region == False.
        The default is 7.5.
    time_ref : float, optional
        Time length of the catalogue used to find a-, b- values. Used when 
        region == False. The default is 1.
    area_ref : TYPE, optional
        Area for obtaining the original a-value (km**2). 
        used when method == 'circular' and region == False.
        The default is 250**2 * np.pi.
    R_max : float, optional
        Maximum distance of interest or radius of a circular area of interest 
        (km). 
        Used when region == False. The default is 250.
    D_max : float, optional
        Maximum depth (km). Used when region == False. The default is 20.
    method : string, optional
        Method to calculate probability of distance(r).
        Avaliable methods: circular, linear, from_file, poly. 
        The default is circular.
    filename : string, optional
        File name to read pre-determined probability of distance or to read 
        longitudes and latitudes of a polygon.
        Requested when method is from_file or poly. The default is False.
    lon_ref : float, optional
        Longitude of the reference point to calculat probability of distance.
        Requested when method is poly. The default is 0.
    lat_ref : float, optional
        Latitude of the reference point to calculat probability of distance.
        Requested when method is poly. The default is 0.
    weighted : string, optional
        If True, combine results of Zh06, BA08, and CY08 with weights 0.4, 0.3,
        0.3. (follow Ghasemi et al., 2006 and Johnson et al., 2020)
        If False, return the results individually. 
        The default is True.
    tectonic : string, optional
        Tectonic region to assign GMPE. ASC or SUB. The default is ASC.

    Returns
    -------
    pga : numpy array
        exp(pga) from 10**(-2) to 10**(1.8) (g).
    prob_pgaZH06 : numpy array
        PoE calculated with GMPE from Zhao et al. (2006).
    prob_pgaBA08 : numpy array
        PoE calculated with GMPE from Boore and Atkinson (2008).
    prob_pgaCY06 : numpy array
        PoE calculated with GMPE from Chiou and Youngs (2008).
    prob_combined : numpy array
        Weighted PoE (0.4 * ZH06 + 0.3 * BA08 + 0.3 * CY08).    

    """
    savefile = 'PGA.txt'
    if region == 'SNC':  #parameters for SNC from ITOPNC catalogue
        a,b = MFR.fit_SNC(region='SNC')
        a = np.round(a,decimals=2)
        b = np.round(b,decimals=2)
        time_ref = 1.08 # years
        area_ref = 38700. # sq km
        R_max = 250
        D_max=20
        M_min=4.5
        M_max=7.5
        method = 'circular'
        filename = False
        savefile = 'PGA_' + region + '.txt'
    elif region == 'NHV':  #parameters for NHV from USGS catalogue 2011 - 2020
        a = 8.16
        b = 1.33
        time_ref = 1.0 # years
        area_ref = False
        R_max = False
        D_max = 50
        M_min = 6.0 
        M_max = 9.0
        savefile = 'PGA_' + region + '.txt'
        method = 'poly'
        lon_ref = 166.4416
        lat_ref = -22.2711
        #filename='dist-percentage_NHV.dat'
        filename = 'polygon_NHV.txt'
            
# Normalise a-value and convert to alpha, beta
    a = a_normalise(a,time_ref,R_max=R_max,area_ref=area_ref,
                method=method)
    alpha = 10**(a) # 10**a0
    beta = b*np.log(10)
    Depth = np.arange(0,D_max + 1,1)
# Area of interest
    if method == 'poly':
        R,R_prob,R_min,R_max,nR = prob_r_poly(lon_ref,lat_ref,
                                              filename=filename)
    else:
        R,R_prob,R_min,R_max,nR = prob_r(R_max=R_max,method=method,
                                     filename=filename)
#Calculate probability
    prob_Mdr, M, d, r = Prob_occurrence(M_min,M_max,M_step,alpha,beta,
                                        Depth,R,R_prob)       
    if tectonic == 'ASC':
        pga, prob_pgaZH06, prob_pgaBA08, prob_pgaCY08 = prob_pga_ASC(M,r,d,
                                                                     prob_Mdr,
                                                                     V30=V30)
    elif tectonic == 'SUB':
        pga, prob_pgaAB03, prob_pgaAn16, prob_pgaYs97 = prob_pga_SUB(M,r,d,
                                                                     prob_Mdr,
                                                                     V30=V30)

    if weighted:
        if tectonic == 'ASC':
            prob_combined = (prob_pgaZH06 * 0.4 + prob_pgaBA08 * 0.3 + 
                             prob_pgaCY08 * 0.3)
            np.savetxt(savefile,(pga,prob_combined))
            zzz=np.loadtxt(savefile)
            np.savetxt(savefile,zzz.T,header='pga, ZH06 * 0.4 + BA08 * 0.3 +'
                       'CY08 * 0.3')
        elif tectonic == 'SUB':
            prob_combined = (prob_pgaAB03 * 0.33 + prob_pgaAn16 * 0.34 + 
                             prob_pgaYs97 * 0.33)
            np.savetxt(savefile,(pga,prob_combined))
            zzz=np.loadtxt(savefile)
            np.savetxt(savefile,zzz.T,header='pga, AB03 * 0.33 + An16 * 0.34 +'
                       'Ys97 * 0.33')
        return(pga,prob_combined)
    else:
        if tectonic == 'ASC':
            np.savetxt(savefile,(pga,prob_pgaZH06,prob_pgaBA08,prob_pgaCY08))
            zzz=np.loadtxt(savefile)
            np.savetxt(savefile,zzz.T,header='pga, ZH06, BA08, CY08')
            return(pga,prob_pgaZH06,prob_pgaBA08,prob_pgaCY08)
        elif tectonic == 'SUB':
            np.savetxt(savefile,(pga,prob_pgaAB03,prob_pgaAn16,prob_pgaYs97))
            zzz=np.loadtxt(savefile)
            np.savetxt(savefile,zzz.T,header='pga, AB03, An16, Ys97')
            return(pga,prob_pgaAB03,prob_pgaAn16,prob_pgaYs97)

def plot_Jo21(file='./hazard_curves_Johnson2021.txt'):
    """
    Plot hazard curves of Noumea and Port Vila from Johnson et al. (2020) to 
    compare with.
    One should request a numerical copy of the result of Johnson et al. (2020) 
    to make the plot.
    
    Parameters
    ----------
    file : string, optional
        The file contains the result from Johnson et al. (2020). 
        The default is './hazard_curves_Johnson2021.txt'.

    Returns
    -------
    None.

    """
## Plot results from Johnson et al. (2021) to compare

    pga_JS = []
    pga_JS_NC = []
    pga_JS_VU = []
    n = 0
    with open(file) as f:
        for line in f.readlines():
            sp=line.split(' ')
            if n > 2:
                pga_JS.append(float(sp[0]))
                pga_JS_NC.append(float(sp[1]))
                pga_JS_VU.append(float(sp[2]))
            n = n + 1
    pga_JS = pga_JS[3:]
    pga_JS_NC = pga_JS_NC[3:]
   # pga_JS_NC = np.array(pga_JS_NC)
   # pga_JS_NC = pga_JS_NC / 2
    pga_JS_VU = pga_JS_VU[3:]
    plt.plot(pga_JS,pga_JS_NC,color='black',linestyle='--')
    plt.plot(pga_JS,pga_JS_VU,color='black',linestyle='--',
             label='Johnson et al. (2021)')

if __name__ == '__main__':
    
    ## Plot results and compare to Johnson et al. (2020)
    pga,prob_SNC = PoE(region='SNC',tectonic='ASC')
    pga_sub,prob_NHV = PoE(region='NHV',tectonic='ASC')
    plt.plot(pga,prob_SNC,color='black',label='Local')
    plt.plot(pga_sub,prob_NHV,color='gray',label='Regional')
    plt.plot(pga,prob_SNC + prob_NHV,color='red',label='Combined')
    plot_Jo21()
    plt.yscale('log')
    plt.xscale('log')
    plt.minorticks_on()
    plt.ylim(5e-7,1)
    plt.xlim(0.01,3)
    plt.xlabel('PGA (g)')
    plt.ylabel('Annual probability of exceedence')
    plt.legend()
