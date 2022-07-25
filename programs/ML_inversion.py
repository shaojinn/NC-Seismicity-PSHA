#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create A and B matrices from phases and stations and solve Ax = B to obtain 
local magnitude function.

@author: shaojinn.chin
"""

import numpy as np
import ReadHyp

def LocalMag_matricesAB(phases,stlist):
    """
    Construct matrices A and B specifically for ML inversion Ax = B

    Parameters
    ----------
    phases : Pandas DataFrame
        Data to be processed. 
        Column names readed: station,channel,event,distance,amplitude.
    stlist : list or Pandas DataFrame
        Stations from which the records will be in the matrices.

    Returns
    -------
    mat_a : Numpy array
        Matrix A.
    mat_b : Numpy array
        Matrix B.

    """
    n_phase = len(phases)
    n_ch = len(stlist) * 2
    mat_a = np.zeros((n_phase,max(phases.event)+n_ch+2))
    mat_b = np.zeros(n_phase)

    for i in range(0,len(phases)):
        phase = phases.iloc[i]
        mat_b[i] = np.log10(phase['amplitude'])
        mat_a[i,0] = np.log10(phase['distance'] / 17)
        mat_a[i,1] = phase['distance'] - 17
        nev = phase['event'] - 1 + 2
        mat_a[i,nev] = 1
        if phase['channel'] == 'N':
            nch = 1
        elif phase['channel'] == 'E':
            nch = 0
        else:
            print('Waring: Channel is neither N nor E')
        try:
            nst = 1 + max(phases.event) + stlist.index(phase['station']) * 2 + 1
            mat_a[i,nst + nch] = 1
        except Exception:
            #print(("Ignore", phase['station']))
            continue
        
    return(mat_a,mat_b)

def MlInversion(MatrixA,MatrixB,Stations):
    """
    Do ML inversion Ax=B and determine constant (in displacement in nm) and 
    station correcstions with sum of corrections = 0.
    * This is an alpha version that produce results identical to matlab without
    uncertainty estimates. Magnitude inversion results in Chin et al. (2022)
    were done with Matlab.

    Parameters
    ----------
    MatrixA : Numpy array
        Matrix A.
    MatrixB : Numpy array
        Matrix A.
    Stations : list
        Stations from which the records will be in the invertion and for which
        corrections will be calculated.
        
        * Stations are assumed operating together so the sum of corrections is
          calculated with all stations. If one or part stations are added or 
          removed during the time period, corrections should be recalculated.
          (e.g. VAL1 and VAL2 in ITOPNC)

    Returns
    -------
    n : float
        Coefficient n for empirical attenuation function.
    k : float
        Coefficient k for empirical attenuation function.
    c : float
        Coefficient c for log(A0) in displacement in nm.
    StationCorrection : Numpy array
        Array of station corrections. One per station. The order follow that 
        of Stations.
    CorrectionShift : float
        Shift that makes the sum of corrections equals to zero.
        May be used to recalculate corrections if stations in Stations are not
        operating together all the time. (e.g. VAL1 and VAL2 in ITOPNC)

    """
    x, residual, rank_a, singluar_a = np.linalg.lstsq(mat_a,mat_b,rcond=None)
    n = -x[0]
    k = -x[1]
    c = 2 - np.log10(480) - n * np.log10(17) - 17 * k
        
    NChannel = len(Stations) * 2
    Correction = -x[-NChannel:]
    CorrectionShift = sum(Correction) / len(Correction)
    Correction = Correction - CorrectionShift
    StationCorrection = np.zeros(len(Stations))
    for i in np.arange(0,NChannel,2):
        StationCorrection[int(i/2)] = sum(Correction[i:i+2]) / 2
        
    return(n,k,c,StationCorrection,CorrectionShift)

if __name__ == '__main__':
    
    filename = "hyp.out_NHV_SNC"
    IgnoreStationList = ['LIFNC','MA2NC','KOUNC']
    
    events,phases = ReadHyp.ReadHyp(filename,
                                      IgnoreStationList=IgnoreStationList)

    stlist = ['MA2NC','KOUNC','LIFNC','MARNC','PINNC','ONTNC','OUENC','YATNC',
              'DZM','NOUC','CAMP','CHAM','DAVA','ENER','KIKI','PAPA','PPRB',
              'ROPT','ROUX','TRIB','VIAL','VAL1','VAL2']
    
    stlist = ReadHyp.IgnoreStation(stlist,IgnoreStationList)
    
    mat_a,mat_b = LocalMag_matricesAB(phases,stlist)
    
    #np.savetxt("matrix_a.txt",mat_a,fmt='%s')
    #np.savetxt("matrix_b.txt",mat_b,fmt='%f')
    
    n,k,c,Corrections,Correction_shift = MlInversion(mat_a,mat_b,stlist)
    
    print((n,k,c))

    
