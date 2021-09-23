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

def y_limb_cut(arr_cut, y_idx, y_min, y_max):
    """
    

    Parameters
    ----------
    arr_cut : array
        端点を切り落とす対象のarray
    y_idx : array
        どこで切り落とすかを決定するためのy
    y_min : float
        y_minに最も近い -側の値が y_new の最後の値になる
    y_max : float
        y_maxに最も近い +側の値が y_new の最初の値になる

    Returns
    -------
    y_new : TYPE
        DESCRIPTION.

    """
    y_min = y_min - 1
    y_max = y_max + 1
    idxmin = abs(y_idx-y_min).argmin()
    idxmax = abs(y_idx-y_max).argmin()
    y_new = arr_cut[idxmax:idxmin]
    return y_new

def size_adjust(X, sag, y, y_min, y_max):
    
    
    b, c = X
    b = int(b)
    sag_b = np.roll(sag, b)
    sag_bc = 1 *sag_b + c
    # ずらした分が入らないように端を切り落とし
    sag_cut = y_limb_cut(sag_bc, y, y_min, y_max)
    return sag_cut
    
def fitting_func(X, Param):
    sag_m, sag_c, y, y_min, y_max = Param
    
    sag_c_adjust = size_adjust(X, sag_c, y, y_min, y_max)
    sag_m_cut = y_limb_cut(sag_m, y, y_min, y_max)
    sigma = sum( (sag_m_cut - sag_c_adjust)**2 )
    return sigma

if __name__ == '__main__':
    pitch_s = 20 #[mm]
    y_min, y_max = -800, 800
    b_max_pitch = 50
    
    
    ## measurement data reading
    y_m_raw, sag_m_raw = read_raw_measurement("raw_data/0921_xm100_0deg.txt")
    
    ## caluculated data reading
    y_c_raw, sag_c_raw = read_raw_calc("raw_data/sample.csv")
    
    ## =======================================================================
    ## m, cのy方向のデータ数を合わせる
    
    y_samp_cut = y_limb_cut(y_m_raw, y_m_raw, y_min-b_max_pitch, y_max+b_max_pitch)
    
    sag_m_interp_cut = y_limb_cut(sag_m_raw, y_m_raw, y_min-b_max_pitch, y_max+b_max_pitch)
    f_interp_c_cut = sp.interpolate.interp1d(y_c_raw, sag_c_raw, kind="linear")
    sag_c_interp_cut = f_interp_c_cut(y_samp_cut)
    
    
    ## =======================================================================
    ## cをmに合わせてfitting
    
    b_init = 0
    c_init = sag_m_interp_cut.mean() - sag_c_interp_cut.mean()
    param = [sag_m_interp_cut, sag_c_interp_cut, y_samp_cut, y_min, y_max]
    
    result1 = sp.optimize.minimize(fitting_func, x0=(b_init, c_init), 
                                   args=param, method="Powell")
    y_min_f = y_min + b_max_pitch
    y_max_f = y_max - b_max_pitch
    y_samp_fit = y_limb_cut(y_samp_cut, y_samp_cut, y_min_f, y_max_f)
    sag_c_fit = size_adjust(result1.x, sag_c_interp_cut, y_samp_cut, y_min_f, y_max_f)
    sag_m_fit = y_limb_cut(sag_m_interp_cut, y_samp_cut, y_min_f, y_max_f)
    sag_diff = sag_m_fit - sag_c_fit
    
    
    ## =======================================================================
    ## y方向のデータ幅を20mmにして逐次積分
    
    fig4 = plt.figure(figsize=(7,7))
    gs4 = fig4.add_gridspec(3,1)
    
    ax43 = fig4.add_subplot(gs4[0,0])
    ax43.set_ylabel("sag ( 20mm pitch)")
    ax43.grid()
    ax41 = fig4.add_subplot(gs4[1,0])
    ax41.set_ylabel("tilt")
    ax41.grid()
    ax42 = fig4.add_subplot(gs4[2,0])
    ax42.set_ylabel("height")
    ax42.grid()
    
    y_start_num = 4
    y_start_pitch = 5
    y_min_s = y_min_f + y_start_num * y_start_pitch
    for j in range(y_start_num):
        
        y_samp_s = np.arange(y_max_f - j*y_start_pitch, y_min_s + j*y_start_pitch, -pitch_s)
        
        f_interp_diff = sp.interpolate.interp1d(y_samp_fit, 2*sag_diff, kind="linear")
        sag_m_interp_diff = f_interp_diff(y_samp_s)
        
        tilt = np.zeros(len(y_samp_s))
        for i in range( len(y_samp_s)-1 ):
            tilt[i+1] = tilt[i] + sag_m_interp_diff[i]
        
        height = np.zeros(len(y_samp_s))
        for i in range( len(y_samp_s)-1 ):
            height[i+1] = height[i] + tilt[i]
        
        ## plot--------------------------------------------------------------------
        ax43.plot(y_samp_s, sag_m_interp_diff)
        ax41.plot(y_samp_s, tilt)
        
        ax42.plot(y_samp_s, height)
        
    fig4.tight_layout()
    
    ## plot--------------------------------------------------------------------
    fig1 = plt.figure(figsize=(7,12))
    gs1 = fig1.add_gridspec(6,1)
    
    ax11 = fig1.add_subplot(gs1[0,0])
    ax11.plot(y_m_raw, sag_m_raw)
    ax11.set_ylabel("measurement sag [nm]")
    ax11.set_title("raw_data")
    
    ax12 = fig1.add_subplot(gs1[1,0])
    ax12.plot(y_c_raw, sag_c_raw)
    ax12.set_ylabel("calculated sag [nm]")
    
    ax13 = fig1.add_subplot(gs1[2,0])
    ax13.plot(y_samp_cut, sag_m_interp_cut)
    ax13.set_ylabel("measurement sag [nm]")
    ax13.set_title("preprocessing")
    
    ax14 = fig1.add_subplot(gs1[3,0])
    ax14.plot(y_samp_cut, sag_c_interp_cut)
    ax14.set_ylabel("calculated sag [nm]")

    
    ax15 = fig1.add_subplot(gs1[4,0])
    ax15.plot(y_samp_fit, sag_m_fit, label="measurement")
    ax15.plot(y_samp_fit, sag_c_fit, label="calculated")
    ax15.set_title("fittig calculated sag")
    ax15.grid()
    
    ax16 = fig1.add_subplot(gs1[5,0])
    ax16.plot(y_samp_fit, sag_diff)
    ax16.set_title("measurement - calculate")
    ax16.grid()

    fig1.tight_layout()