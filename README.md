# Swelling_tablet_fronts_D_k_from_MRI_T2_or_img
Tool for characterizing the swelling of tablets immersed in a solution. Creates time plots and calculates eroding front's diffusion rate D and the rate of the swelling k from time series of either T<sub>2</sub>-maps or (properly contrasted, e.g. T<sub>2</sub>-weighted) MRI images in FDF (file format native for Agilent MRI scanners) or Text Image format. This software is suitable for swelling matrix tablets forming a membrane-like structure in contact with the solution in which they are immersed.

## The repository contains:
1. Python script D_k_from_T2maps.py - the main version of the software.
2. Jupyter notebook file notebook_D_k_from_T2maps.ipynb presenting:
- short introduction to the topic in which the software can be usefull,
- input parameters and their meanings,
- how the code works step by step,
- sample results with commentary.
3. EXE file for non-coders D_k_from_T2maps_vB_win10.exe (compiled for Windows 10) that can be used simply by double-clicking, taking input parameters from an input file. This program only accepts Text Images as an input (I had troubles compyling a script utilizing itk - python library enabling FDF import). HINT: Most of image files can be converted to Text Image format using ImageJ (https://imagej.nih.gov/ij/download.html). To import FDF files by ImageJ you'll need Multi FDF Opener plugin (https://imagej.nih.gov/ij/plugins/multi-opener.html).
4. Python script used for creating the EXE file: D_k_from_T2maps_vB.py.
5. Sample input file for D_k_from_T2maps_vB: INPUT-D_k_from_T2maps_vB.txt.
6. Sample MRI-derived T<sub>2</sub>-maps in FDF format in MRI_FDF folder.
7. Sample MRI-derived T<sub>2</sub>-maps in Image File format in MRI_TXTimages folder.
8. Sample results in D_results folder.

## License
The software is licensed under the MIT license. The non-software content of this project is licensed under the Creative Commons Attribution 4.0 International license. See the LICENSE file for license rights and limitations.
