#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict peak ground acceleration (PGA) with ground motion prediction equations
(GMPEs) for active shallow crust (ASC) (Zhao et al., 2006, Boore and Atkinson, 
2008, and Chiou and Youngs, 2008) and for subduction interface (SUB)
(Atkinson and Boore, 2003, Abrahamson et al., 2016, and Youngs et al., 1997).

Convert between magnitudes and numbers of earthquakes with given a- and b- 
values.

@author: shaojinn.chin
 
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def GR_M(N,a,b):
    """
    Gutenberg-Richter relation logN = a - bM with parameters a,b 
    and N being the cumulative number of earthquakes with magnitude > M.   

    Parameters
    ----------
    N : numpy array
        Number of events >= M.
    a : float
        a value.
    b : float
        b value.

    Returns
    -------
    M : numpy array
        Magnitude values.

    """
    M = (a - np.log10(N)) / b
    return M
    
def GR_N(M,a,b,integral=False):
    """
    Gutenberg-Richter relation logN = a - bM with parameters a,b 
    and N being the cumulative number of earthquakes with magnitude > M.   

    Parameters
    ----------
    M : numpy array
        Magnitude values.
    a : float
        a value.
    b : float
        b value.

    Returns
    -------
    N : numpy array
        Number of events >= M.

    """
    if integral:
        N = np.rint(np.power(10,a - b*M))
    else:
        N = np.power(10,a - b*M)
    return N
    
def GR_dN(M,a,b):
    """
    Given a Gutenberg-Richter relation logN = a - bM with parameters a,b 
    and N being the cumulative number of earthquakes with magnitude > M.
    Assumes an input set of equally-spaced magnitudes M (bin centers) and
    computes predicted numbers in each bin.

    Parameters
    ----------
    M : numpy array
        Magnitude bin midpoints.
    a : float
        a value.
    b : float
        b value.

    Returns
    -------
    dN : numpy array
        Number of events in each bin.

    """
    dM = M[1] - M[0]
    dN = b *  np.log(10) * np.power(10,a-b*M) * dM
    return dN

def lnpgaZH06(mag,dist,depth=15.,C_site=1.111,Fr=0):
    """
    Peak ground accelleration (PGA) prediction equation, Zhao et al. (2006).
    Bulletin of the Seismological Society of America, 96(3), 898-913.

    Parameters
    ----------
    mag : numpy array
        moment magnitude.
    dist : numpy array
        distance in km.
    depth : numpy array
        depth in km.
    C_site : float, optional
        Site class term. The default is rock 1.111.
        0.298 solid rock V30 > 1100
        1.111 rock 600 < V30 < 1100
        1.344 hard soil 300 < V30 < 600
        1.355 medium soil 200 < V30 < 300
        1.420 soft soil V30 < 200
    Fr : float or numpy array, optional
        Fault parameter, default is 0 for all earthquake events.
        Reverse fault events have Fr = 0.251
    
    Returns
    -------
    ln_pga : numpy array
        Natural log of PGA in m/s/s for each event.
    sigma : float
        Standard error of ln_pga.

    """
    ## coefficient list (Table 4)
    a = 1.101
    b = -0.00564
    c = 0.0055
    d = 1.080
    e = 0.01412 
    
    r = dist + c * np.exp(d * mag)
    
    deep = (depth > 15)
    
    # ignore Si + Ss + Ssl*np.log(dist) terms 
    ln_pga = ( a*mag + b*dist - np.log(r) + 
              deep * e * (depth - 15)  + Fr +  C_site )
    
    # standard error of model prediction (sigmaT, Table 5)
    sigma = 0.723 
    
    # convert pga from cm/s/s in paper to m/s/s
    # log(pga*0.01) = log(pga) + log(0.01)
    ln_pga = ln_pga + np.log(0.01)
    
    return  ln_pga,sigma

def lnpgaBA08(mag,dist,V30=850.):
    """
    Peak ground accelleration (PGA) prediction equation. 
    Boore and Atkinson (2008). Earthquake spectra, 24(1), 99-138.
    
    Parameters
    ----------
    mag : numpy array
        DESCRIPTION.
    dist : numpy array
        DESCRIPTION.
    V30 : float, optional
        Average shear wave velocity (m/s) to 30 m depth. The default is 850.
        solid rock V30 > 1100
        rock 600 < V30 < 1100
        hard soil 300 < V30 < 600
        medium soil 200 < V30 < 300
        soft soil V30 < 200

    Returns
    -------
    ln_pga : numpy array
        PGA in m/s/s.
    sigma : float
        Standard error of ln_pga.

    """
    g = 9.8
    
    def F_d(dist,mag,R_ref=1.0):
        # Equations 3-4: distance scaling
        # distance scaling - Table 6
        h = 1.35
        c1 = -0.66050
        c2 = 0.11970
        c3 = -0.01151
        M_ref = 4.5
        R = np.sqrt(dist**2 + h**2)
        return ((c1 + c2 * (mag - M_ref)) * np.log(R / R_ref) + 
                c3 * (R - R_ref))

    def F_m(mag):
        ## Equation 5: magnitude scaling (uses only 'unspecified' at present) 
        # Magnitude scaling, Table 7
        U = 1
        Mh = 6.75
        e1 = -0.53804
        e5 = 0.28805
        e6 = -0.10164
        e7 = 0.0
        dM = mag - Mh
        smallEQ = (dM <= 0)
        largeEQ = (dM > 0)    
        return (smallEQ * (e1 * U + e5 * dM + e6 * dM**2) +
                largeEQ * (e1 * U + e7 * dM))

    def F_lin(V30):
        # site amplification coefficient, Table 3
        v_ref = 760
        b_lin = -0.360
        # Equation 7: site amplification linear term 
        return b_lin * np.log(V30 / v_ref)  
    
    def F_nl(V30,dist,mag):
        # site amplification coefficients, Tables 3,4
        b1 = -0.64
        b2 = -0.14
        a1 = 0.03 * g
        a2 = 0.09 * g
        pga_low = 0.06 * g
        v1 = 180
        v2 = 300
        v_ref = 760
        # Equation 13: non-linear slope
        tinyV = (V30 <= v1)
        smallV = (V30 > v1) & (V30 <= v2)
        mediumV = (V30 > v2) & (V30 < v_ref)
        largeV = (V30 >= v_ref)
        b_nl = (tinyV * b1 +
                smallV * ((b1-b2) * np.log(V30/v2) / np.log(v1/v2) + b2) +
                mediumV * (b2 * np.log(V30/v_ref) / np.log(v2/v_ref)) +
                largeV * 0.0 )
        
        # Equation 1 ground motion prediction equation; no non-linear term
        # see Table 6 - R_ref = 5 km for pga4nl
        R_ref = 5.0
        ln_pga4nl = F_d(dist,mag,R_ref) + F_m(mag) 
        pga4nl = np.exp(ln_pga4nl)

        # Equations 8-12: Site amplification non-linear term 
        smallPGA = (pga4nl <= a1)
        mediumPGA = np.logical_and(pga4nl > a1, pga4nl <= a2)
        largePGA = (pga4nl > a2)
        delx = np.log(a2/a1)
        dely = b_nl * np.log(a2/pga_low)
        c = (3 * dely - b_nl * delx) / (delx**2)
        d = -(2 * dely - b_nl * delx) / (delx**3)
               
        return (smallPGA * b_nl * np.log(pga_low/0.1) +
                mediumPGA * (b_nl * np.log(pga_low/0.1) +
                             c * (np.log(pga4nl/a1)**2) +
                             d * (np.log(pga4nl/a1)**3)) +
                largePGA * b_nl * np.log(pga4nl/0.1))
    
    # Equation 1 ground motion prediction equation
    # see Table 6 - R_ref = 1.0 for linear
    R_ref = 1.0
    ln_pga = (F_m(mag) + F_d(dist,mag,R_ref) + 
              F_lin(V30) + F_nl(V30,dist,mag))
    
    # return ln(pga) and the standard deviation of ln(pga) residuals
    # Table 8 - uncertainty
    sigma = 0.566
    
    # convert from units of g to SI units of m/s/s
    ln_pga = ln_pga + np.log(g)
    
    return ln_pga,sigma

def lnpgaCY08(mag,dist,depth=15.,V30=850.):
    """
    Peak ground accelleration (PGA) prediction equation. 
    Chiou and Youngs (2008). Earthquake spectra, 24(1), 173-215.
    

    Parameters
    ----------
    mag : numpy array 
        magnitude.
    dist : numpy array 
        distance.
    depth : numpy array or float, optional
        depth. The default is 15.
    V30 : numpy array or float, optional
        Shear wave velocity to 30 m depth. The default is 850.0 m/s.

    Returns
    -------
    ln_pga : numpy array
        PGA in m/s/s.
    sigma : numpy array
        Standard error of ln_pga.

    """
    # Chiou, B. J., & Youngs, R. R. (2008). Earthquake spectra, 24(1), 173-215
    ## coefficient list for PGA prediction equation from Chiou and Youngs (2008)
    ## coefficients for reference amplitude equation 
    c1 = -1.2687
    c1a = 0.1
    c1b = -0.2550
    c2 = 1.06
    c3 = 3.45
    c4 = -2.1
    c4a = -0.5
    c5 = 6.1600
    c6 = 0.4893
    c7 = 0.0512
    c9 = 0.7900
    c9a = 1.5005
    cn = 2.996
    cM = 4.184
    cRB = 50
    cHM = 3
    cr1 = -0.00804
    cr2 = -0.00785
    cr3 = 4
    ## depth to top of rupture = focal depth - 1/2(rupture width)*cos(angle)
    ## rupture width is estimated based on the regression result for all 
    ## earthquake types from Table 2A in Wells and Coppersmith (1994)
    fault_width = (10 ** (-1.01 + 0.32 * mag))
    mean_cos_angle = 0.6351 # avearge of cos from 0 to 90 degree 
    Z_TOR = depth - 0.5 * mean_cos_angle * fault_width
    ## Modify Z_TOR while Z_TOR < 0 
    while (Z_TOR[0] < 0):
        Z_TOR = (Z_TOR + depth) / 2
    r_rup = (dist ** 2 + Z_TOR ** 2) ** (1 / 2) # distance to rupture
    mean_cos_angle2 = 0.5 # average of cos square from 60 to 90 degree (normal faults)
    ## coefficients for prediction with reference amplitude equation
    ph1 = -0.4417
    ph2 = -0.1417
    ph3 = -0.007010
    ph4 = 0.102151
    ph5 = 0.2289
    ph6 = 0.014996
    ph7 = 580
    ph8 = 0.07
    ## coefficients for variance
    tau1 = 0.3437
    tau2 = 0.2637
    sigma1 = 0.4458
    sigma2 = 0.3459
    sigma3 = 0.8
    sigma4 = 0.0663
    eta = 0 # Equations (20,21) and description
    b = (ph2 * (np.exp(ph3 * np.minimum(V30,1130) - 360) - 
                np.exp(ph3 * (1130-360))))
    c = ph4
    ## END of coefficient list
    lnZ1 = 28.5 - (3.82 / 8) * np.log(V30**8 + 378.7**8)
    Z1 = np.exp(lnZ1)
    
    
    ln_pga_ref = (c1 + (c1a + c1b) / 2 +  c7 * (Z_TOR - 4) + 
                  c2 * (mag - 6) + 
                  ((c2 - c3) / cn) * np.log(1 + np.exp(cn * (cM - mag))) + 
                  c4 * np.log(r_rup + c5 * np.cosh(c6 * (mag - cHM))) + 
                  (c4a - c4) * np.log((r_rup ** 2 + cRB ** 2)**0.5) + 
                  (cr1 + (cr2 / np.cosh(mag - cr3))) * r_rup + 
                  c9 * (2 / 2)*np.tanh(dist * mean_cos_angle2 / c9a) * 
                  (1 - (((dist ** 2 + Z_TOR ** 2)**0.5) / (r_rup + 0.001))))
    
    pga_ref = np.exp(ln_pga_ref)
    NL0 = b * pga_ref / (pga_ref + c)                        
    ln_pga = (ln_pga_ref + ph1 * np.log(V30 / 1130) + 
              ph2 * (np.exp(ph3 * (V30 - 360)) - np.exp(ph3 * (1130 - 360))) * 
              np.log((np.exp(ln_pga_ref) * np.exp(eta) + ph4) / ph4) + 
              ph5 * (1 - (1 / (np.cosh(ph6 * 0)))) + 
              ph8 / np.cosh(0.15 * (Z1 - 15)))
        
    tau = tau1 + (((tau2 - tau1) / 2) * (np.minimum(np.maximum(mag,5),7) - 5))
    sigma = (((sigma1 + (((sigma2 - sigma1) / 2) * 
                         (np.minimum(np.maximum(mag,5),7) - 5))) * 
             (sigma3 + (1+NL0)**2)**0.5))
    sigma = ((1 + NL0)**2 * tau**2 + sigma**2)**0.5 
    
    # convert to SI units of m/s/s from multiples of g
    ln_pga = ln_pga + np.log(9.8)    
    
    return ln_pga,sigma

def lnpgaBSSA14(mag,dist,depth=15,V30=850):
    """
    Peak ground accelleration (PGA) prediction equation. 
    Boore et al. (2014). Earthquake spectra, 30(3), 1057-1085.   

    Parameters
    ----------
    mag : numpy array 
        magnitude.
    dist : numpy array 
        distance.
    depth : numpy array or float, optional
        depth. The default is 15.
    V30 : numpy array or float, optional
        Shear wave velocity to 30 m depth. The default is 850.0 m/s.

    Returns
    -------
    ln_pga : numpy array
        PGA in m/s/s.
    sigma : numpy array
        Standard error of ln_pga.
    """
    ## Parameters
    e0 = 0.4473
    e4 = 1.431
    e5 = 0.05053
    e6 = -0.1662
    Mh = 5.5
    c1 = -1.134
    c2 = 0.1917
    c3 = -0.008088
    delta_c3 = 0 # Global, CA, TW, NZ
    Mref = 4.5
    Rref = 1
    h = 4.5
    c = -0.6
    Vc = 1500
    Vref = 760
    f1 = 0
    f3 = 0.1
    f4 = -0.15
    f5 = -0.00701
    R1 = 110
    R2 = 270
    delta_phiR = 0.1
    delta_phiV = 0.07
    V1 = 225
    V2 = 300
    phi1 = 0.695
    phi2 = 0.495
    tau1 = 0.398
    tau2 = 0.348
    
    # R in the paper, assume Rjb = dist (random direction)
    r_rup = (dist**2 + depth**2)**(1 / 2)
    
    # Unspecified events only (i.e. Not use e1, e2, e3)
    Fe = e0 + (e4 * ((mag <= Mh) * (mag - Mh)) + 
               e5 * ((mag <= Mh) * (mag  - Mh))**2 + 
               e6 * ((mag > Mh) * (mag - Mh)))
    Fp = ((c1 + c2 * (mag - Mref)) * np.log(r_rup / Rref) + 
          (c3 + delta_c3) * (r_rup - Rref))
    # Calcilate reference PGA for nonlinear term Fnl    
    pga_ref = np.exp(Fe + Fp)
    
    Flin = c * np.log(np.minimum(V30,Vc) / Vref)
    f2 = (f4 * (np.exp(f5 * (np.minimum(V30,760) - 360)) - 
                np.exp(f5 * (760 - 360))))
    Fnl = f1 + f2 * np.log((pga_ref + f3) / f3)
    Fs = Flin + Fnl # Does not consider basin effect for PGA in BSSA14
    ln_pga = Fe + Fp + Fs 
    #Convet pga from g to SI units m/s/s
    ln_pga = ln_pga + np.log(9.8)
    
    tau = ((mag <= 4.5) * tau1 + 
           ((mag > 4.5) & (mag < 5.5)) * 
           (tau1 + (tau2 - tau1) * (((mag > 4.5) & (mag < 5.5)) * (mag - 4.5)))
           + (mag >= 5.5) * tau2)
    phiM = ((mag <= 4.5) * phi1 + 
           ((mag > 4.5) & (mag < 5.5)) * 
           (phi1 + (phi2 - phi1) * (((mag > 4.5) & (mag < 5.5)) * (mag - 4.5)))
           + (mag >= 5.5) * phi2)
    phiR = ((dist <= R1) * 0 +
            ((dist > R1) & (dist <= R2)) * delta_phiR * 
            (np.log(dist / R1) / np.log(R2 / R1)) +
            (dist > R2) * delta_phiR)
    phiV = ((V30 >= V2) * 0 -
            ((V30 > V1) & (V30 <= V2)) * delta_phiV * 
            (np.log(V2 / V30) / np.log(V2 / V1)) -
            (V30 < V1) * delta_phiV)
    phi = phiM + phiR + phiV
    sigma = (phi**2 + tau**2)**(1 / 2)
    
    return ln_pga,sigma

def lnpgaYs97(mag,dist,depth=15,Zt=False):
    """
    Peak ground accelleration (PGA) prediction equation for rock sites and 
    subducton zones earthquakes.
    Youngs et al. (1997). Seismological Research Letters, 68(1), 58-73
    
    Parameters
    ----------
    mag : numpy array 
        magnitude.
    dist : numpy array 
        distance.
    depth : numpy array or float, optional
        depth. The default is 15.
    Zt : bool, optional
        True for intraslab, False for interface . The default is False.

    Returns
    -------
    ln_pga : numpy array
        PGA in m/s/s.
    sigma : numpy array
        Standard error of ln_pga.

    """
    ## coefficient list (Table 2 for rock sites)
    c1 = 0.0
    c2 = 0.0
    c3 = -2.552
    c4 = 1.45
    c5 = -0.1
    r = (dist**2 + depth**2)**(1 / 2)
    ln_pga = (0.2418 + 1.414 * mag + c1 + c2 * (10 - mag)**3 +
              c3 * np.log(r + 1.7818 * np.exp(0.554 * mag)) +
              0.00607 * depth + 0.3846 * Zt)
    
    #Convet pga from g to SI units m/s/s
#    ln_pga = np.log(np.exp(ln_pga) * 9.8)
    ln_pga = ln_pga + np.log(9.8)   
    
    sigma = c4 + c5 * np.minimum(mag,8)
    
    return ln_pga,sigma

def lnpgaAB03(mag,dist,depth=15,Zt=False):
    """
    Peak ground accelleration (PGA) prediction equation for rock sites and 
    subducton zones earthquakes. 
    Atkinson and Boore (2003). Bulletin of the Seisological Society of America,
    93(4), 1703-1729.

    Parameters
    ----------
    mag : numpy array 
        magnitude.
    dist : numpy array 
        distance.
    depth : numpy array or float, optional
        depth. The default is 15.
    Zt : bool, optional
        True for intraslab, False for interface . The default is False.

    Returns
    -------
    ln_pga : numpy array
        PGA in m/s/s.
    sigma : numpy array
        Standard error of ln_pga.

    """
    c1 = 2.991
    c2 = 0.03525
    c3 = 0.00759
    c4 = -0.00206
    sigma = 0.23
    
    if Zt: #Intraslab events
        g = 10**(0.301 - 0.01 * mag)
        mag = np.minimum(mag,8)
    else: #Interplate events
        g = 10**(1.2 - 0.18 * mag)
        mag = np.minimum(mag,8.5)
        
    #Closest dist to fault surface    
    D = (dist**2 + (np.minimum(depth,100))**2)**(1 / 2) 
    delta = 0.00724 * 10**(0.507 * mag)
    R = (D**2 + delta**2)**(1 / 2)

    #Predict log10(pga) in cm/s/s (equation 1)
    log_pga = c1 + c2 * mag + c3 * depth + c4 * R - g * np.log10(R)
    #Convert units from cm/s/s to m/s/s
    log_pga = log_pga - np.log10(100)
    #Convert log10 to ln
    ln_pga = np.log(10**(log_pga))
    
    return ln_pga, sigma

def lnpgaAn16(mag,dist,depth=15,V30=850,Zt=False):
    """
    Peak ground accelleration (PGA) prediction equation for rock sites and 
    subducton zones earthquakes. 
    Abrahamson et al. (2016). Earthquake Spectra, 32(1), 23-44.

    Parameters
    ----------
    mag : numpy array 
        magnitude.
    dist : numpy array 
        distance.
    depth : numpy array or float, optional
        depth. The default is 15.
    V30 : numpy array or float, optional
        Shear wave velocity to 30 m depth. The default is 850.0 m/s.
    Zt : bool, optional
        True for intraslab, False for interface. The default is False. Not used
        in this version.

    Returns
    -------
    ln_pga : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.

    """
    
    # Period-independent coefficients
    n = 1.18
    c = 1.88
    theta3 = 0.1
    theta4 = 0.9
    theta5 = 0.0
    theta9 = 0.4
    c4 = 10
    # Period dependent coefficients
    Vlin = 865.1
    b = -1.186
    theta1 = 4.2203
    theta2 = -1.350
    theta6 = -0.0012
    theta12 = 0.980
    theta13 = -0.0135
    sigma = 0.74
    c1 = 7.8
    delta_c1 = 0.2
    r_rup = (dist**2 + depth**2)**(1 / 2)
    fmag = theta4 * (mag - (c1 + delta_c1)) + theta13 * (10 - mag)**2 # M <= 8
    ffaba = 0 # For forearc and unknow sites
    Vlin = 865.1 # For PGA
    Vs = 1000
    fsite = theta12 * np.log(Vs / Vlin) + b * n * np.log(Vs / Vlin)
    ln_pga1000 = (theta1 + theta4 * delta_c1 + (theta2 + theta3 * (mag - 7.8)) 
                  * np.log(r_rup + c4 * np.exp(theta9 * (mag - 6))) + theta6 * 
                  r_rup + fmag + ffaba + fsite)
    pga1000 = np.exp(ln_pga1000)
    Vs = np.minimum(V30,1000)
    if V30 >= Vlin:
        fsite = theta12 * np.log(Vs / Vlin) + b * n * np.log(Vs / Vlin)
    else:
        fsite = (theta12 * np.log(Vs / Vlin) - b * np.log(pga1000 + c) + 
                 b * np.log(pga1000 + c * (Vs / Vlin)**n))
    ln_pga = (theta1 + theta4 * delta_c1 + (theta2 + theta3 * (mag - 7.8)) * 
              np.log(r_rup + c4 * np.exp(theta9 * (mag - 6))) + theta6 * r_rup 
              + fmag + ffaba + fsite)
    #Convet pga from g to SI units m/s/s
    ln_pga = ln_pga + np.log(9.8)
    
    return ln_pga,sigma

def pga_g(ln_pga):
    """
    Convert ln(pga) in SI units (m/s/s) to pga in g.

    Parameters
    ----------
    ln_pga : numpy array
        ln(pga) in m/s/s.

    Returns
    -------
    numpy array
        pga in g.

    """
    return np.exp(ln_pga) / 9.8

def prob_exceedance(pga_threshold,mean_pga,sigma):
    """
    Probability of exceedance with given mean, sigma, and threshold.

    Parameters
    ----------
    pga_threshold : float
        Threshold to calculate exceedance probability.
    mean_pga : numpy array
        Mean pga.
    sigma : float or numpy array
        Standard error of predicted pga.

    Returns
    -------
    prob : numpy array
        Probability of exceedance of pga.

    """
    prob = 1 - norm.cdf(pga_threshold,mean_pga,sigma)
    return prob

def comparePGA(mag,dist,depth=10,V30=850,comparison=False,labeling=False):
    """
    Plot pga predicted with Zhao et al. (2006), Boore and Atkinson (2008) and 
    Chiou and Youngs (2008) against distance and optionally compare to results 
    from OpenQuake.

    Parameters
    ----------
    mag : numpy array
        Magnitudes.
    dist : numpy array
        Distances.
    depth : float, optional
        Depth in km to predict pga. The default is 10.
    V30 : float, optional
        Average shear velocity of the upper 30 m. The default is 850.
    comparison : bool, optional
        If True, plot results results from OpenQuake from existing files. 
        The default is False.
    labeling : bool, optional
        If True, plot labels. The default is False.

    Returns
    -------
    None.

    """
    
#    r_rup = (dist**2 + depth**2)**(1 / 2)
    ln_pga,s = lnpgaZH06(mag,dist,depth=depth)
    ASC = 0.4 * ln_pga
    plt.plot(dist,pga_g(ln_pga),color='gray',label='Zhao et al (2006)')
    
    ln_pga,s = lnpgaBA08(mag,dist,V30=V30)
    ASC = ASC + 0.3 * ln_pga
    plt.plot(dist,pga_g(ln_pga),color='gray',linestyle='-.',
             label='Boore & Atkinson (2008)')
    if mag[0] == 7.5:
        plt.plot(dist,pga_g(ln_pga + 2 * s),color='blue',linestyle=':')
        plt.plot(dist,pga_g(ln_pga - 2 * s),color='blue',linestyle=':')
        
    ln_pga,s = lnpgaCY08(mag,dist,depth=depth,V30=V30)
    ASC = ASC + 0.3 * ln_pga
    plt.plot(dist,pga_g(ln_pga),color='gray',linestyle=':',
             label='Chiou & Youngs (2008)')
       
    ln_pga,s = lnpgaAB03(mag,dist,depth=depth)
    SUB = 0.34 * ln_pga
    plt.plot(dist,pga_g(ln_pga),color='pink',linestyle=':',
             label='Atkinson & Boore (2003)')
   
    ln_pga,s = lnpgaYs97(mag,dist,depth=depth)
    SUB = SUB + 0.33 * ln_pga
    plt.plot(dist,pga_g(ln_pga),color='pink',label='Youngs et al.(2008)')
    
    ln_pga,s = lnpgaAn16(mag,dist,depth=depth)
    SUB = SUB + 0.33 * ln_pga
    plt.plot(dist,pga_g(ln_pga),color='pink',linestyle='-.',
             label='Abrahamsom et al.(2016)')
   
    ln_pga,s = lnpgaBSSA14(mag,dist,depth=depth)
    plt.plot(dist,pga_g(ln_pga),color='lightblue',linestyle='-.',
             label='Boore et al.(2014)')
    if mag[0] == 7.5:
        plt.plot(dist,pga_g(ln_pga + 2 * s),color='blue',linestyle=':')
        plt.plot(dist,pga_g(ln_pga - 2 * s),color='blue',linestyle=':')
    
    plt.plot(dist,pga_g(ASC),color='black')
    plt.plot(dist,pga_g(SUB),color='red')
    
    if (comparison):
        file = './pga_M7.5_dep10km_Zh.dat' 
        pga_M75_dep10km_Zh = []
        with open(file) as f:
            for line in f.readlines():
                sp=line.split(' ')
                pga_M75_dep10km_Zh.append(float(sp[2]))
        file = './pga_M7.5_dep10km_BA.dat' 
        pga_M75_dep10km_BA = []
        with open(file) as f:
            for line in f.readlines():
                sp=line.split(' ')
                pga_M75_dep10km_BA.append(float(sp[2]))
        file = './pga_M7.5_dep10km_CY.dat' 
        pga_M75_dep10km_CY = []
        with open(file) as f:
            for line in f.readlines():
                sp=line.split(' ')
                pga_M75_dep10km_CY.append(float(sp[2]))
        dist = np.arange(0.,208.,1)
        plt.plot(dist,pga_M75_dep10km_Zh[:],color='red',linestyle='--',
                 label='OpenQuake: Zhao et al (2006)')
        plt.plot(dist,pga_M75_dep10km_BA[:],color='blue',linestyle='--',
                 label='OpenQuake: Boore & Atkinson (2008)')
        plt.plot(dist,pga_M75_dep10km_CY[:],color='green',linestyle='--',
                 label='OpenQuake: Chiou & Youngs (2008)')

    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,500)
    plt.ylim(1e-5,1)
    plt.xlabel('Distance (km)')
    plt.ylabel('PGA (g)')
    if(labeling):
        plt.legend()
   # plt.show()
    
    
if __name__ == '__main__':
    """ Unit tests """  
    
    # ---------------------------------------------------------------------
    # plot distance scaling
    
    dist = np.arange(1.,801.,1)
    depth = 20
    mag = np.ones_like(dist) * 5
    comparePGA(mag,dist,depth=depth,labeling=True,comparison=False)
    mag = np.ones_like(dist) * 6
    comparePGA(mag,dist,depth=depth,comparison=False)
    mag = np.ones_like(dist) * 7 
    comparePGA(mag,dist,depth=depth,comparison=False)
    mag = np.ones_like(dist) * 8 
    comparePGA(mag,dist,depth=depth,comparison=False)
    mag = np.ones_like(dist) * 9 
    comparePGA(mag,dist,depth=depth,comparison=False)
    plt.savefig('GMPE_CrustalSubduction.png',dpi=600)

    
   
