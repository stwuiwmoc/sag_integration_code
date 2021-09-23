# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:45:52 2021

@author: swimc
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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

def read_raw_measurement(fname):
    # x, y, z : [mm]
    # s : [nm]
    raw = np.loadtxt(fname)
    num, x, y, z, s1, s2, s3 = raw.T
    sag = (s2*2 - s1 - s3)/2
    return y, sag
   
def read_raw_calc(fname):
    # x, y, z, rho, R : [mm]
    # s : [mm] -> [nm]
    raw = np.loadtxt("raw_data/sample.csv", delimiter=",", skiprows=1, encoding="utf-8")
    x, y, z, rho, R, sag = raw.T
    sag_nm = sag * 1e6
    return y, sag_nm

def size_adjust(X, sag):
    #a, c = X
    #sag_ac = a * sag + c
    c = X
    sag_c = sag + c
    return sag_c
    
def fitting_func(X, Param):
    sag_m, sag_c = Param
    
    sag_c_adjust = size_adjust(X, sag_c)
    sigma = sum( (sag_m - sag_c_adjust)**2 )
    return sigma

if __name__ == '__main__':
    pitch_s = 20 #[mm]
    pitch_fit = 5
    y_min, y_max = -800, 800
    
    ## measurement data reading
    y_m_raw, sag_m_raw = read_raw_measurement("raw_data/0921_xm100_0deg.txt")
    
    ## caluculated data reading
    y_c_raw, sag_c_raw = read_raw_calc("raw_data/sample.csv")
    
    ## plot -------------------------------------------------------------------
    fig1 = plt.figure(figsize=(7,5))
    gs1 = fig1.add_gridspec(2,1)
    fig1.suptitle("raw data")

    ax11 = fig1.add_subplot(gs1[0,0])
    ax11.plot(y_m_raw, sag_m_raw)
    ax11.set_ylabel("measurement sag [nm]")
    
    ax12 = fig1.add_subplot(gs1[1,0])
    ax12.plot(y_c_raw, sag_c_raw)
    ax12.set_ylabel("calculated sag [nm]")
    
    ## =======================================================================
    ## m, cのy方向のデータ数を合わせる
    
    y_samp_fit = np.arange(y_max, y_min, -pitch_fit)
    
    f_interp_m_fit = sp.interpolate.interp1d(y_m_raw, sag_m_raw, kind="linear")
    sag_m_intrep_fit = f_interp_m_fit(y_samp_fit)
    f_interp_c_fit = sp.interpolate.interp1d(y_c_raw, sag_c_raw, kind="linear")
    sag_c_intrep_fit = f_interp_c_fit(y_samp_fit)
    
    ## plot--------------------------------------------------------------------
    fig2 = plt.figure(figsize=(7,5))
    fig2.suptitle("Interpolattion and Limb cut")
    gs2 = fig2.add_gridspec(2,1)
    
    ax21 = fig2.add_subplot(gs2[0,0])
    ax21.plot(y_samp_fit, sag_m_intrep_fit)
    ax21.set_ylabel("measurement sag [nm]")
    
    ax22 = fig2.add_subplot(gs2[1,0])
    ax22.plot(y_samp_fit, sag_c_intrep_fit)
    ax22.set_ylabel("calculated sag [nm]")
    
    ## =======================================================================
    ## cをmに合わせてfitting
    
    #a_init = (sag_m_intrep_fit.max() - sag_m_intrep_fit.min()) / (sag_c_intrep_fit.max() - sag_c_intrep_fit.min())
    #c_init = sag_m_intrep_fit.mean() - a_init * sag_c_intrep_fit.mean()
    c_init = sag_m_intrep_fit.mean() - sag_c_intrep_fit.mean()
    param = [sag_m_intrep_fit, sag_c_intrep_fit]
    #result1 = sp.optimize.minimize(fitting_func, x0=(a_init, c_init), 
    #                               args=param, method="Powell")
    result1 = sp.optimize.minimize(fitting_func, x0=(c_init), 
                                   args=param, method="Powell")
    
    sag_c_fit = size_adjust(result1.x, sag_c_intrep_fit)
    
    sag_diff = sag_m_intrep_fit - sag_c_fit
    
    ## plot--------------------------------------------------------------------
    fig3 = plt.figure(figsize=(7,5))
    fig3.suptitle("adjust calc")
    gs3 = fig3.add_gridspec(2,1)
    
    ax31 = fig3.add_subplot(gs3[0,0])
    ax31.plot(y_samp_fit, sag_m_intrep_fit, label="measurement")
    ax31.plot(y_samp_fit, sag_c_fit, label="calculated")
    ax31.legend()
    
    ax32 = fig3.add_subplot(gs3[1,0])
    ax32.plot(y_samp_fit, sag_diff)
    ax32.set_title("measurement - calculate")

    fig3.tight_layout()
    
    ## =======================================================================
    ## y方向のデータ幅を20mmにして逐次積分
    
    fig4 = plt.figure(figsize=(7,5))
    gs4 = fig4.add_gridspec(2,1)
    
    ax41 = fig4.add_subplot(gs4[0,0])
    ax41.set_ylabel("tilt")
    ax42 = fig4.add_subplot(gs4[1,0])
    ax42.set_ylabel("height")

    y_start_num = 4
    y_start_pitch = 5
    y_min_s = y_min + y_start_num * y_start_pitch
    for j in range(y_start_num):
        
        y_samp_s = np.arange(y_max - j*y_start_pitch, y_min_s + j*y_start_pitch, -pitch_s)
        
        f_interp_diff = sp.interpolate.interp1d(y_samp_fit, sag_diff, kind="linear")
        sag_m_interp_diff = f_interp_diff(y_samp_s)
        
        tilt = np.zeros(len(y_samp_s))
        for i in range( len(y_samp_s)-1 ):
            tilt[i+1] = tilt[i] + sag_m_interp_diff[i]
        
        height = np.zeros(len(y_samp_s))
        for i in range( len(y_samp_s)-1 ):
            height[i+1] = height[i] + tilt[i]
        
        ## plot--------------------------------------------------------------------
        ax41.plot(y_samp_s, tilt)
        
        ax42.plot(y_samp_s, height)
        
    fig4.tight_layout()
    