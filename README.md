# NC-seismicity-PSHA
NC earthquakes catalogue and hazard analysis
## Table of contents
* General info
* Earthquake catalogue
* programs
## General info
This is the archive of the earthquake catalogue python programs to generate the results in Chin et al (2022).
## Earthquake catalogue
The earthquake catalogue is archived in nordic format (SEISAN). Detailed description can be found in Chin et al. (2022).
## Programs
### ReadHyp.py
* Read an earthquake catalogue in nordic format to event and phase lists and optionally output a summary earthquake catalogue in csv format.
### ML_inversion.py
* Build A and B matrices and solve Ax = B to obtain parameters of a local magnitude function.
* Earthquakes should be selected before creating matrices.
### MFR.py
* Find Magnitude-frequency relation (MFR, as known as Gutenberg-Richter relation, a- and b-values) from synthetic or real data by determining magnitude of completeness (Mc) and goodness of fit.
### GMPE.py
* Ground motion Equations (GMPE) to predict peak ground accelerations (PGA) for active shallow crust (ASC, Zhao et al., 2006, Boore and Atkinson, 2008, and Chiou and Youngs, 2008) and for subduction interface (SUB, Atkinson and Boore, 2003, Abrahamson et al., 2016, and Youngs et al., 1997).
