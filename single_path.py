 # -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:45:52 2021

@author: swimc
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle

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

def read_raw_measurement(fname, full=False):
    # x, y : [mm]
    # azimuth : [degree]
    # s : [nm]
    raw = np.loadtxt(fname)
    num, x, y, azimuth, s1, s2, s3 = raw.T
    sag = (s2*2 - s1 - s3)/2
    if full==False:
        return y, sag
    else:
        return y, sag, x, azimuth
   
def read_raw_calc(fname, full=False):
    # x, y, z, rho, R : [mm]
    # s : [mm] -> [nm]
    raw = np.loadtxt("raw_data/sample.csv", delimiter=",", skiprows=1, encoding="utf-8")
    x, y, z, rho, R, sag = raw.T
    sag_nm = sag * 1e6
    if full == False:
        return y, sag_nm
    else:
        return y, sag_nm, z

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

def poly_calc(fanc, x):
    order = len(fanc)
    y = 0
    
    for i in range(order):
        y = y + fanc[i]*x**i    
    return y
"""
def size_adjust(X, sag_func, y):    
    a, b, c = X
    sag_abc =  a*poly_calc(sag_func, y-b) + c
    return sag_abc
"""  
def size_adjust(X, sag_func, y):    
    b, c = X
    sag_bc =  a*poly_calc(sag_func, y-b) + c
    return sag_bc
    

def fitting_func(X, Param):
    sag_m, sag_c_func, y = Param
    
    sag_c_adjust = size_adjust(X, sag_c_func, y)
    sigma = sum( (sag_m - sag_c_adjust)**2 )
    return sigma

def rotate_matrix(theta_deg):
    theta = np.deg2rad(theta_deg)
    matrix = np.array([[ np.cos(theta), np.sin(theta)],
                       [-np.sin(theta), np.cos(theta)]])
    return matrix
    

if __name__ == '__main__':
    pitch_s = 20 #[mm]
    y_min, y_max = -750,750
    b_max_pitch = 50
    
    a = 1 #縦倍率
    ## measurement data reading
    y_m_raw, sag_m_raw, x_m_raw, azimuth_m_raw = read_raw_measurement("raw_data/0921_xm100_0deg.txt", True)
    azimuth = int(azimuth_m_raw[0] + 9.893)
    
    ## caluculated data reading
    y_c_raw, sag_c_raw = read_raw_calc("raw_data/sample.csv")
    
    ## =======================================================================
    ## 前処理
    sag_m_gaussian = sp.ndimage.filters.gaussian_filter(sag_m_raw, 2)
    
    ## m, cのy方向のデータ数を合わせる
    
    y_samp_cut = y_limb_cut(y_m_raw, y_m_raw, y_min-b_max_pitch, y_max+b_max_pitch)
    
    sag_m_interp_cut = y_limb_cut(sag_m_gaussian, y_m_raw, y_min-b_max_pitch, y_max+b_max_pitch)
    
    f_interp_c_cut = np.polynomial.polynomial.polyfit(y_c_raw, sag_c_raw, 4)
    sag_c_interp_cut = poly_calc(f_interp_c_cut, y_samp_cut)
    
    
    ## =======================================================================
    ## cをmに合わせてfitting
    
    b_init = 0
    c_init = sag_m_interp_cut.mean() - sag_c_interp_cut.mean()
    param = [sag_m_interp_cut, f_interp_c_cut, y_samp_cut]
    
    #result1 = sp.optimize.minimize(fitting_func, x0=(1, b_init, c_init), 
     #                             args=param, method="Powell")
    result1 = sp.optimize.minimize(fitting_func, x0=(b_init, c_init), 
                                  args=param, method="Powell")
    
    sag_c_fit = size_adjust(result1.x, f_interp_c_cut, y_samp_cut)
    
    sag_diff = 2*(sag_m_interp_cut - sag_c_fit)
    
    
    ## =======================================================================
    ## y方向のデータ幅を20mmにして逐次積分
    polyfit_order = 6
    
    fig4 = plt.figure(figsize=(14,14))
    gs4 = fig4.add_gridspec(4,2)
    
    ax43 = fig4.add_subplot(gs4[0,1])
    ax43.set_ylabel("sag ( 20mm pitch)")
    ax43.grid()
    
    ax41 = fig4.add_subplot(gs4[1,1])
    ax41.set_ylabel("tilt")
    ax41.grid()
    
    ax42 = fig4.add_subplot(gs4[2,1])
    ax42.set_ylabel("height")
    ax42.grid()
    
    ax44 = fig4.add_subplot(gs4[3,1])
    ax44.set_ylabel("height - " + str(polyfit_order)+" order polyfit")
    ax44.grid()
    
    
    y_start_num = 1
    y_start_pitch = 5
    save_num = 1
    y_min_s = y_min + y_start_num * y_start_pitch
    color=["blue", "orange", "green", "red", "lightblue", "yellow", "lightgreen", "pink"]
    
    write_fname = mkfolder() + "test.txt"
    
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
        height_fit = poly_calc(f_height_fit, y_samp_s)
        height_diff = height - height_fit
        
        if j == save_num:
            ## savetxt ---------------------------------------------------------------
            ## xのデータ数をy_samp_sに合わせる
            f_interp_x_m = sp.interpolate.interp1d(y_m_raw, x_m_raw, kind="linear")
            x_m_interp = f_interp_x_m(y_samp_s)
            
            ## ロボ座標系からOAP座標系に回転
            x_rotate, y_rotate = np.dot(rotate_matrix(-azimuth), np.array([x_m_interp, y_samp_s]))
            
            data_id = np.arange(1, len(y_samp_s)+1)
            beam_id = 1 * np.ones(len(y_samp_s))
            arr_save = np.array([data_id, x_rotate, y_rotate, height_diff, beam_id]).T
            np.savetxt(write_fname, arr_save, fmt=["%.0f", "%.4f","%.4f","%.9f","%.0f"])
            
        ## plot--------------------------------------------------------------------
        ax43.plot(y_samp_s, sag_m_interp_diff, color=color[j], label="start = "+str(y_samp_s.max()))
        ax41.plot(y_samp_s, tilt, color=color[j])

        
        ax42.plot(y_samp_s, height, color=color[j])
        ax42.plot(y_samp_s, height_fit, marker=".", color=color[j+4], label="fit for "+str(y_samp_s.max()))
        ax44.plot(y_samp_s, height_diff, color=color[j])
    
    ax43.legend()
    ax42.legend()
    
    
    ax45 = fig4.add_subplot(gs4[0,0])
    ax45.plot(y_m_raw, sag_m_raw)
    ax45.set_ylabel("measurement sag [nm]")
    ax45.set_title("raw_data")
    
    ax46 = fig4.add_subplot(gs4[1,0])
    ax46.plot(y_samp_cut, sag_m_interp_cut, label="measurement")
    ax46.plot(y_samp_cut, sag_c_fit, label="calculated")
    ax46.set_title("fittig calculated")
    ax46.grid()
    
    ax47 = fig4.add_subplot(gs4[2,0])
    ax47.plot(y_samp_cut, sag_diff)
    ax47.set_title("measurement - calculate")
    ax47.grid()

    fig4.tight_layout()
    fig4.savefig(mkfolder() + "test.png")
    
    
    ## plot--------------------------------------------------------------------
    """
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
    ax15.plot(y_samp_cut, sag_m_interp_cut, label="measurement")
    ax15.plot(y_samp_cut, sag_c_fit, label="calculated")
    ax15.set_title("fittig calculated")
    ax15.grid()
    
    ax16 = fig1.add_subplot(gs1[5,0])
    ax16.plot(y_samp_cut, sag_diff)
    ax16.set_title("measurement - calculate")
    ax16.grid()

    fig1.tight_layout()
    """