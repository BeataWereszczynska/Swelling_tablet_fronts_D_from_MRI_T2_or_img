# -*- coding: utf-8 -*-
"""
Calculates swelling tablet front's diffusion rate D from time series of T2 maps (or MRI images) in Text Image format.
Version B: taking input parameters from the input file (INPUT-D_from_T2maps_vB.txt).

Created on Thu Oct 6 2022
Last modified on Mon Oct 10 2022
@author: Beata Wereszczyńska
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.optimize import curve_fit
import pymsgbox

# reading input parameters from the input file
exec(open('INPUT-D_from_T2maps_vB.txt').read())

def D_from_T2maps():
    """
    Calculates swelling tablet front's diffusion rate D from time series of T2 maps (or MRI images) in Text Image format.
    T2 map's file name has to be the time elapsed since putting the tablet in the solution in minutes.
    Input: 
        T2 maps (or MRI images) location folder: maps_path [str],
        size of a pixel in mm: pixel_size [float]
        region of interest: roi [tuple, ((x1,y1),(x2,y2))],
        averaging direction: aver_axis [0 - average rows, 1 - average collumns of ROI],
        range of data for D function fitting: fit_range [tuple (start_point, stop_point)]
        output folder: out_folder [str].
    """
    # message box
    pymsgbox.alert(text=f'Starting calculations with: \n \n \
     T2 maps (or MRI images) location folder: {maps_path} \n \
     size of a pixel in mm: {pixel_size} \n \
     region of interest: {roi} \n \
     output folder: {out_folder} \n \
     averaging direction: {aver_axis} \n \
     range of data for D function fitting: {fit_range}', \
     title='D_from_T2maps_vB', button='Close this window to continue')
    
    
    shutil.rmtree(out_folder, ignore_errors=True)            # removing residual output folder with content
    os.makedirs(out_folder)                                  # creating new output folder
    files = os.listdir(maps_path)                            # list of T2 map (MRI images) files
    df = pd.DataFrame()                                      # dataframe for the average profiles
    
    for file in files:
        
        # import images/maps as np arrays
        locals()[os.path.splitext(file)[0]] = np.loadtxt(f'{maps_path}/{file}', dtype=float)
        
    
        # T2 maps and ROI: visual veryfication
        img = locals()[os.path.splitext(file)[0]]
        img = img/(img.max()/255.0)
        plt.rcParams['figure.dpi'] = 200
        figure, image = plt.subplots()
        image.imshow(img, cmap=plt.get_cmap('gray'))
        w = roi[1][0] - roi[0][0]
        h = roi[1][1] - roi[0][1]
        roi_rect = Rectangle(roi[0], w, h, linewidth=0.5, edgecolor='cyan', facecolor='none')
        image.add_patch(roi_rect)
        plt.axis('off')
        plt.title(f'{int(os.path.splitext(file)[0])} minutes')
        # saving T2 map with ROI as png image
        plt.savefig(f'{out_folder}/{os.path.splitext(file)[0]}.png', bbox_inches='tight')
        # showing T2 map with ROI
        plt.close()
        
        # extracting ROIs from images
        locals()[os.path.splitext(file)[0]] = locals()[os.path.splitext(file)[0]][roi[0][1] : roi[1][1]+1, roi[0][0] : roi[1][0]+1]
        
        # creating average profiles
        locals()[os.path.splitext(file)[0]] = np.mean(locals()[os.path.splitext(file)[0]], axis=aver_axis)
    
        # filling dataframe with the profiles as columns (column names are time in minutes)
        df[int(os.path.splitext(file)[0])] = locals()[os.path.splitext(file)[0]]
        
        
    # dataframe indexes as distance in mm
    df['distances'] = (pd.Series(range(len(df.index)-1, -1, -1))) * pixel_size
    df = df.set_index(['distances'])
    
    # find min T2 (tablet front location) in every profile
    minT2distances = df.idxmin()
    minT2distances = minT2distances.rename({'0': 'distance_mm'})
    # save datapoints as csv
    minT2distances.to_csv(f'{out_folder}/datapoints.csv', index_label='time_minutes', header=['distance_mm'])
    
    # the T2-profile time dependence with overlayed scatter plot
    # (1) colormesh
    x = df.columns
    y = df.index
    z = np.array(df)
    x, y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    plt.grid(False)
    ax.pcolormesh(x.astype(float), y, z, cmap = 'gray')
    plt.colorbar(ax.pcolormesh(x.astype(float), y, z, cmap = 'gray'), label = r'$T_{2}$'+' (s)')
    # (2) scatter plot
    plt.scatter(minT2distances.index, minT2distances, marker='x', color='cyan', s=10)
    plt.xlabel('soaking time (minutes)')
    plt.ylabel('distance (mm)')
    plt.savefig(f'{out_folder}/plot1.png', bbox_inches='tight')
    plt.close()
    
    # data for function fitting (recomended range: from t-min to first plateau)
    t = list(minT2distances.index[fit_range[0]:fit_range[1]] * 60)             # t * 60 (min -> sec)
    dist = list(minT2distances[fit_range[0]:fit_range[1]] / 10)                # dist / 10 (mm -> cm)
    
    # fitting function definition
    def Dfunct(x, D, c):
        # diffusion coefficient approximation: dist = (2Dt)^(1/2) ; x is t
    	    return np.sqrt(2*D*x) + c
    
    # fit curve
    params, covariance = curve_fit(Dfunct, t, dist, bounds=([0,-np.inf], np.inf))
    D, c = params
    # get the standard deviations of the parameters
    st_devs = np.sqrt(np.diag(covariance))
    D_st_dev = st_devs[0]
    c_st_dev = st_devs[1]
    
    # save parameters with standard deviations in txt file
    with open(f'{out_folder}/fit_parameters.txt', 'w') as f:
        f.write('Fitting results for    <x> = (2Dt) ^ (1/2) + c')
        f.write('\n\n')
        f.write(f'D = {D} '+u'\u00B1'+f' {D_st_dev}')
        f.write('\n')
        f.write(f'c = {c} '+u'\u00B1'+f' {c_st_dev}')
        
    # plot of fitted curve
    x_curve = range(min(t), max(t), 1)
    y_curve = Dfunct(x_curve, D, c)
    plt.plot(x_curve, y_curve, color='black')
    plt.scatter(t, dist, color='black')
    plt.xlabel('soaking time (minutes)')
    plt.ylabel('distance (mm)')
    plt.savefig(f'{out_folder}/plot2.png', bbox_inches='tight')
    plt.close()
    

def main():
    D_from_T2maps()
    
    pymsgbox.alert(text=f'Results saved in the output folder ({out_folder}).', \
                   title='D_from_T2maps_vB', button='Close this window')

if __name__ == "__main__":
    main()

# label = ttk.Label(window, text = f'Results saved in output folder {out_folder}.\n You can now close this window.')
# label.pack()
# root.mainloop()

