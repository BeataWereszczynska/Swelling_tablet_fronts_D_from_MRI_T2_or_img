# Swelling_tablet_fronts_D_from_MRI_T2_or_img
Tool for characterizing the swelling of tablets immersed in a solution. Creates time plots and calculates swelling tablet front's diffusion rate D from time series of T2 maps (or properly contrasted, e.g. T2-weighted, MRI images) in FDF (file format native for Agilent MRI scanners) or Text Image format. This software is suitable for swelling tablets forming a membrane-like structure in contact with the solution in which they are immersed.

## The repository contains:
1. Python script D_from_T2maps.py - the main version of the software.
2. (COMMING SOON) Jupyter notebook file presenting:
- short introduction to the topic in which the software can be usefull,
- input parameters and their meanings,
- how the code works step by step,
- presentation of sample results with commentary.
3. EXE file for non-coders D_from_T2maps_vB_win10.exe (compiled for Windows 10) that can be used simply by double-clicking, taking input parameters from an input file. This program only accepts Text Images as an input (I had troubles compyling a script utilizing itk - python library enabling FDF import). HINT: Most of image files can be converted to Text Image format using ImageJ (https://imagej.nih.gov/ij/download.html). To import FDF files by ImageJ you'll need Multi FDF Opener plugin (https://imagej.nih.gov/ij/plugins/multi-opener.html).
4. Python script used for creating the EXE file: D_from_T2maps_vB.py.
5. Sample input file for D_from_T2maps_vB: INPUT-D_from_T2maps_vB.txt.
6. Sample MRI-derived T2-maps in FDF format in the MRI_FDF folder.
7. Sample MRI-derived T2-maps in Image File format in the MRI_TXTimages folder.
8. Sample results in D_results folder.
