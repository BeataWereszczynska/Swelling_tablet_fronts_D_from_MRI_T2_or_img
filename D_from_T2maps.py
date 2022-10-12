# -*- coding: utf-8 -*-
"""
Calculates swelling tablet front's diffusion rate D and rate of the swelling k 
from time series of T2 maps (or MRI images) in FDF or Text Image format.
Version A: taking input parameters simply from main() function (not using input file).

Created on Thu Oct 6 2022
Last modified on Wed Oct 12 2022
@author: Beata Wereszczyńska
"""
import os
import shutil
import itk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.optimize import curve_fit

def D_from_T2maps(maps_path, file_type, pixel_size, roi, aver_axis, fit_range_D, fit_range_k, out_folder):
    """
    Calculates swelling tablet front's diffusion rate D and rate of the swelling k 
    from time series of T2 maps (or MRI images) in FDF or Text Image format.
    T2 map's file name has to be the time elapsed since putting the tablet in the solution in minutes.
    Input: 
        T2 maps (or MRI images) location folder: maps_path [str],
        type of image file: file_type [bool: 0 - Text Image, 1 - FDF],
        size of a pixel in mm: pixel_size [float],
        region of interest: roi [tuple, ((x1,y1),(x2,y2))],
        averaging direction: aver_axis [0 - average rows, 1 - average collumns of ROI],
        range of data for D function fitting: fit_range_D [tuple (start_point, stop_point)],
        range of data for k function fitting: fit_range_k [tuple (start_point, stop_point)],
        output folder: out_folder [str].
    """
    shutil.rmtree(out_folder, ignore_errors=True)            # removing residual output folder with content
    os.makedirs(out_folder)                                  # creating new output folder
    files = os.listdir(maps_path)                            # list of T2 map (MRI images) files
    df = pd.DataFrame()                                      # dataframe for the average profiles
    
    for file in files:
        
        # import images/maps as np arrays
        if file_type:
            imageio = itk.FDFImageIO.New()
            locals()[os.path.splitext(file)[0]] = np.array(itk.imread(f'{maps_path}/{file}', imageio=imageio))
            
        else:
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
        plt.show()
        
        # extracting ROIs from images
        locals()[os.path.splitext(file)[0]] = locals()[os.path.splitext(file)[0]][roi[0][1] : roi[1][1]+1, \
                                                                                  roi[0][0] : roi[1][0]+1]
        
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
    plt.show()
    
    # data for function fitting (recomended range: from t-min to the point before the start of the first plateau)
    t = list(minT2distances.index[fit_range_D[0]:fit_range_D[1]] * 60)             # t * 60 (min -> sec)
    dist = list(minT2distances[fit_range_D[0]:fit_range_D[1]] / 10)                # dist / 10 (mm -> cm)
    
    # fitting function definition
    def Dfunct(x, D, c):
        # diffusion coefficient D approximation: dist = (2Dt)^(1/2) ; x is time
        return np.sqrt(2*D*x) + c
    
    # fit curve
    params, covariance = curve_fit(Dfunct, t, dist, bounds=((0,-np.inf), np.inf))
    D, c = params
    # get the standard deviations of the parameters
    st_devs = np.sqrt(np.diag(covariance))
    D_st_dev, c_st_dev = st_devs
    
    # plot of fitted curve
    x_curve = range(min(t), max(t), 1)
    y_curve = Dfunct(x_curve, D, c)
    plt.plot(x_curve, y_curve, color='black')
    plt.scatter(t, dist, color='black', s=30)
    plt.xlabel('soaking time (seconds)')
    plt.ylabel('distance (cm)')
    plt.savefig(f'{out_folder}/plot2.png', bbox_inches='tight')
    plt.show()
    
    # data for 2nd function fitting (recomended range: from t-min to the end of the first plateau)
    t = list(minT2distances.index[fit_range_k[0]:fit_range_k[1]])
    dist = list(minT2distances[fit_range_k[0]:fit_range_k[1]])
    
    # 2nd fitting function definition
    def kfunct(x, max_dist, a, k):
        # rate of swelling k approximation: dist = max_dist - a * exp(-k * t) ; x is time
        return max_dist - a * (np.exp(-k * x))
    
    # fit 2nd curve
    params, covariance = curve_fit(kfunct, t, dist, p0 = (dist[-1], 3, 0.0036), bounds=((0,-np.infty, 0), np.inf))
    max_dist, a, k  = params
    st_devs = np.sqrt(np.diag(covariance))
    max_dist_st_dev, a_st_dev, k_st_dev = st_devs
    
    # plot of 2nd fitted curve
    x_curve = range(min(t), max(t), 1)
    y_curve = kfunct(x_curve, max_dist, a, k)
    plt.plot(x_curve, y_curve, color='black')
    plt.scatter(t, dist, color='black', s=30)
    plt.xlabel('soaking time (minutes)')
    plt.ylabel('distance (mm)')
    plt.savefig(f'{out_folder}/plot3.png', bbox_inches='tight')
    plt.show()

    # save parameters with standard deviations in txt file
    with open(f'{out_folder}/fit_parameters.txt', 'w') as f:
        f.write('Fitting results for    <x> = (2Dt) ^ (1/2) + c \n\n')
        f.write(f'D = {D} '+u'\u00B1'+f' {D_st_dev} (cm2s-1) \n')
        f.write(f'c = {c} '+u'\u00B1'+f' {c_st_dev} (cm) \n\n')
        f.write('Fitting results for    dist = max_dist - a * exp(-k * t) \n\n')
        f.write(f'max_dist = {max_dist} '+u'\u00B1'+f' {max_dist_st_dev} (mm) \n')
        f.write(f'a = {a} '+u'\u00B1'+f' {a_st_dev} (mm) \n')
        f.write(f'k = {k} '+u'\u00B1'+f' {k_st_dev} (min-1)\n')
        

def main():
    maps_path = "MRI_FDF"             # Path to T2 maps (or MRI images): maps_path [str]
    file_type = 1                     # Type of image file: file_type [bool: 0 - Text Image, 1 - FDF]
    pixel_size = 0.3125               # Size of a pixel in mm: pixel_size [float]
    roi = ((66, 55), (69, 76))        # Region of interest: roi [tuple, ((x1,y2),(x2,y1))]
    out_folder = "D_results"          # Output folder: out_folder [str]
    aver_axis = 1                     # Averaging direction: aver_axis (0 - average rows, 1 - average collumns of ROI)
    fit_range_D = (0, 13)             # Range of data for D function fitting: fit_range_D [tuple (start_point, stop_point)]
    fit_range_k = (0, 16)             # Range of data for k function fitting: fit_range_k [tuple (start_point, stop_point)]
    
    # maps_path = "MRI_TXTimages"
    # file_type = 0
    
    D_from_T2maps(maps_path, file_type, pixel_size, roi, aver_axis, fit_range_D, fit_range_k, out_folder)


if __name__ == "__main__":
    main()
