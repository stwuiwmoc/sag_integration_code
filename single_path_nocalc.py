# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:04:43 2021

@author: swimc
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import single_path as sgp

def mkfolder(suffix = ""):
    import os
    """    
    Parameters
    ----------
    suffix : str, optional
        The default is "".

    Returns
    -------
    str ( script name + suffix )
    """
    filename = os.path.basename(__file__)
    filename = filename.replace(".py", "") + suffix
    folder = "mkfolder/" + filename + "/"
    os.makedirs(folder, exist_ok=True)
    return folder

if __name__ == '__main__':
    pitch_s = 20 #[mm]
    y_min, y_max = -750,750
    b_max_pitch = 50
    
    a = 1 #縦倍率
    ## measurement data reading
    read_fname = "raw_data/0922xm130_3deg/0922xm130_3deg_000.txt"
    y_m_raw, sag_m_raw = sgp.read_raw_measurement(read_fname)
    y_samp_cut = sgp.y_limb_cut(y_m_raw, y_m_raw, y_min-b_max_pitch, y_max+b_max_pitch)

    sag_m_cut = sgp.y_limb_cut(sag_m_raw, y_m_raw, y_min-b_max_pitch, y_max+b_max_pitch)
    
    f_sag_c_fit = np.polynomial.polynomial.polyfit(y_samp_cut, sag_m_cut, 4)
    sag_c_fit = sgp.poly_calc(f_sag_c_fit, y_samp_cut)
    
    sag_diff = 2*(sag_m_cut - sag_c_fit)
    
    ## =======================================================================
    ## y方向のデータ幅を20mmにして逐次積分
    polyfit_order = 6
    
    fig1 = plt.figure(figsize=(14,14))
    gs1 = fig1.add_gridspec(4,2)
    
    
    ax13 = fig1.add_subplot(gs1[0,1])
    ax13.set_ylabel("sag ( 20mm pitch)")
    ax13.grid()
    
    ax14 = fig1.add_subplot(gs1[1,1])
    ax14.set_ylabel("tilt")
    ax14.grid()
    
    ax15 = fig1.add_subplot(gs1[2,1])
    ax15.set_ylabel("height")
    ax15.grid()
    
    ax16 = fig1.add_subplot(gs1[3,1])
    ax16.set_ylabel("height - " + str(polyfit_order)+" order polyfit")
    ax16.grid()
    
    
    y_start_num = 4
    y_start_pitch = 5
    y_min_s = y_min + y_start_num * y_start_pitch
    
    color=["blue", "orange", "green", "red", "lightblue", "yellow", "lightgreen", "pink"]
    for j in range(y_start_num):
        
        y_samp_s = np.arange(y_max - j*y_start_pitch, y_min + j*y_start_pitch, -pitch_s, dtype="float")
        f_interp_diff = sp.interpolate.interp1d(y_samp_cut, sag_diff, kind="cubic")
        sag_m_interp_diff = f_interp_diff(y_samp_s)
        
        tilt = np.zeros(len(y_samp_s))
        for i in range( len(y_samp_s)-1 ): 
            tilt[i+1] = tilt[i] + sag_m_interp_diff[i]    
        
        height = np.zeros(len(y_samp_s))
        for i in range( len(y_samp_s)-1 ):
            height[i+1] = height[i] + tilt[i]
        
        f_height_fit = np.polynomial.polynomial.polyfit(y_samp_s, height, polyfit_order, w=np.ones(len(y_samp_s)))
        height_fit = sgp.poly_calc(f_height_fit, y_samp_s)
        height_diff = height - height_fit
        
        ## plot--------------------------------------------------------------------
        ax13.plot(y_samp_s, sag_m_interp_diff, color=color[j], label="start = "+str(y_samp_s.max()))
        ax14.plot(y_samp_s, tilt, color=color[j])
       
        ax15.plot(y_samp_s, height, color=color[j])
        ax15.plot(y_samp_s, height_fit, marker=".", color=color[j+4], label="fit for "+str(y_samp_s.max()))
        
        ax16.plot(y_samp_s, height_diff, color=color[j])
    
    ax13.legend()
    ax15.legend()
    
    ax17 = fig1.add_subplot(gs1[0,0])
    ax17.plot(y_m_raw, sag_m_raw)
    ax17.set_ylabel("measurement sag [nm]")
    ax17.set_title("raw_data")
    ax17.grid()
    
    
    ax11 = fig1.add_subplot(gs1[1,0])
    ax11.plot(y_samp_cut, sag_m_cut, label="measurement")
    ax11.plot(y_samp_cut, sag_c_fit, label="calculated")
    ax11.set_title("fittig calculated")
    ax11.grid()
    
    ax12 = fig1.add_subplot(gs1[2,0])
    ax12.plot(y_samp_cut, sag_diff)
    ax12.set_title("measurement - calculate")
    ax12.grid()

    fig1.tight_layout()
    