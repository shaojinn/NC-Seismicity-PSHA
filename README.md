# NC-seismicity-PSHA
NC earthquakes catalogue and hazard analysis
## Table of contents
* [General info](#general-info)
* [Earthquake catalogue](#earthquake-catalogue)
* [programs](#programs)
## General info
This is the archive of the earthquake catalogue and python programs to generate the results in Chin et al (2022).
## Earthquake catalogue
The earthquake catalogue is archived in nordic format (SEISAN). Detailed description can be found in Chin et al. (2022).
Polygons are coordinates defining southern New Caledonia (SNC) and New Hebrides-Vanuatu subduction zone (NHV) in Chin et al. (2022).
Velocity1D.txt- 1D velocity model determined with the earthquake catelogue with VELEST.
## Programs
### [ReadHyp.py](programs/ReadHyp.py)
* Read an earthquake catalogue in nordic format to event and phase lists and optionally output a summary earthquake catalogue in csv format.
### [ML_inversion.py](programs/ML_inversion.py)
* Build A and B matrices and solve Ax = B to obtain parameters of a local magnitude function.
* Earthquakes should be selected before creating matrices.
### [MFR.py](programs/MFR.py)
* Find Magnitude-frequency relation (MFR, as known as Gutenberg-Richter relation, a- and b-values) from synthetic or real data by determining magnitude of completeness (Mc) and goodness of fit.
### [GMPE.py](programs/GMPE.py)
* Ground motion Equations (GMPE) to predict peak ground accelerations (PGA) for active shallow crust (ASC, Zhao et al., 2006, Boore and Atkinson, 2008, and Chiou and Youngs, 2008) and for subduction interface (SUB, Atkinson and Boore, 2003, Abrahamson et al., 2016, and Youngs et al., 1997).
### [PSHA_probability.py](programs/PSHA_probability.py)
* Calculate annual probability of exceedance (PoE) of PGA with given MFR, magnitude range, site condition (Vs30), source zone, and site of interest.
### pga_M7.5_depth10km_*.dat
* PGA(g) aggainst distances (km) predicted with OpenQuake using set parameters (M=7.5, depth=10 km, Vs30=850 m/s) and corresponding GMPEs (Zh- Zhao et al., 2006; BA- Boore and Atkinson, 2008; CY- Chiou and Youngs, 2008) for comparing output from GMPE.py.
### [hazard_curves_Johnson2021.txt](programs/hazard_curves_Johnson2021.txt)
* Hazard curves for Noumea and Port Vila in Johnson et al. (2021). Kindly provided by Dr. Kendra Johnson in Global Earthquake Model Foundation, Italy. One should ask them for permission before using it for other purposes.
