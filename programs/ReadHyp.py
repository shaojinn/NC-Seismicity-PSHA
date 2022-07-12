#!/usr/bin/env python
# python3
"""
    Read earthquake catalog generated by SEISAN - HYP program and rewrite
    summary catalog file.
    
    @author: shaojinn.chin
"""
from datetime import datetime
import pandas as pd


def ReadHyp(InputFileName,IgnoreStationList=[],Epicentre=False,
            OutputCatalog=False,OutputFileName='EarthquakeCatalog.csv'):
    """
    Read from a SEISAN file to generate a Pandas Data frame for ML inversion.

    Parameters
    ----------
    filename : string
        File name of the input SEISAN file.
    IgnoreStationList : list, optional
        A list of stations from which the records will not be listed in the 
        output and thus will not go in to the inversion. The default is [].
    Epicentre : bool, optional
        Use epicentre distance if Ture. The default is False.
    OutputCatalog : bool, optional
        Output an summary earthquake catalogue if true. The default is False.
    OutputFileName : string. optional
        Output file name. The default is EarthquakeCatalog.csv

    Returns
    -------
    phases : Pandas DataFrame
        Pandas Data Frame contains station, channel, event number, distance,
        amplitude.
    events : Pandas DataFrame
        Pandas Data Frame contains the summary of earthquakes.

    """
    phases = []
    events = []
    nev = 0
    for line in open(InputFileName, "r"):
        if line[79:80] == '1' and line[59:63] == 'LTES':
            nev = nev + 1
            lon = float(line[30:38])
            lat = float(line[23:30])
            depth = float(line[38:43])
            magnitude = float(line[55:59])
            year = int(line[1:5])
            month = int(line[6:8])
            day = int(line[8:10])
            hr = int(line[11:13])
            mm = int(line[13:15])
            sec = int(line[16:18])
            microsec = int(line[19:20])*100000
            time = datetime(year,month,day,hr,mm,sec,microsec)
            n = int(line[47:51])
            rms = float(line[52:55])
            events.append((nev,time,lon,lat,depth,magnitude,n,rms))
        if line[10:14] == 'IAML':
            st = line[1:6]
            st = st.strip()
            ch = line[7:8]
            amp = float(line[33:40])
            dist = float(line[71:75])
            if Epicentre:
                dist = dist
            else:
                dist = (dist**2 + depth**2)**(1 / 2)
            phases.append((st,ch,nev,dist,amp))
                
    phases = pd.DataFrame({'station' : [i[0] for i in phases],
                           'channel' : [i[1] for i in phases],
                           'event':[i[2] for i in phases],
                           'distance':[i[3] for i in phases],
                           'amplitude':[i[4] for i in phases]})
    events = pd.DataFrame({'No' : [i[0] for i in events],
                          'origin' : [i[1] for i in events],
                          'longitude' : [i[2] for i in events],
                          'latitude' : [i[3] for i in events],
                          'depth' : [i[4] for i in events],
                          'magnitude' : [i[5] for i in events],
                          'N_phases' : [i[6] for i in events],
                          'rms' : [i[7] for i in events]})
    if IgnoreStationList != []:
        phases = IgnoreStation(phases, IgnoreStationList)
        
    if OutputCatalog:
        with open(OutputFileName,'w') as fOut:
            fOut.write(
                'Earthquake catalog extracted from SEISAN HYP output file.\n'+
                'CSV text with 5 header lines, including col names in last.\n'+
                'longitude, latitude (WGS 84); depth in km.\n'+
                'mmagnitude = local magnitude determined by SEISAN HYP.\n' +
                'n = number of stations used; rms = fit to arrival times\n')
            events.to_csv(fOut,index=False)
    
    return(events,phases)

def IgnoreStation(datalist,station):
    """
    Remove stations or records from stations.

    Parameters
    ----------
    datalist : Pandas DataFrame or list
        Data to be processed.
    station : list
        Stations from which the records will be removed.

    Returns
    -------
    datalist : Pandas DataFrame or list
        Data without records from ignored stations.

    """
    if isinstance(datalist, pd.DataFrame):
        for i in station:
            datalist = datalist.drop(datalist[datalist.station == i].index)
    elif isinstance(datalist, list):
        for i in station:
            datalist.remove(i)
        
    return (datalist)


# generate a summary catalog text file from HYP output file
if __name__ == '__main__':
    ReadHyp('hyp.out_NHV_SNC',OutputCatalog=True)


