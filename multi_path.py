# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:59:35 2021

@author: swimc
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import pandas as pd

import scipy.ndimage
import scipy.optimize
import scipy.interpolate


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
    y_min_fit, y_max_fit = -750,750
    b_max_pitch = 50
    
    a = 0.99809 #縦倍率
    
    fname_txt_m = "0922xm130_3deg"
    fname_txt_c = "x-130deg3sag^M0921"
    
    data_id_sum = 0
    write_fname = mkfolder("/"+fname_txt_m) + fname_txt_m + "_height.txt"
    
    start = time.time()
    y_samp_s_list = []
    azimuth_list = []
    b_list = []
    
    for k in range(1):
        print(k)
        print(time.time() - start)
        fnum = str(k).zfill(3)
        
        ## measurement data reading
        read_fname_m = "raw_data/" + fname_txt_m + "/" + fname_txt_m + "_" + fnum + ".txt"
        y_m_raw, sag_m_raw, x_m_raw, azimuth_raw = sgp.read_raw_measurement(read_fname_m, full=True)
        azimuth = round(azimuth_raw[0] + 10.814, 3)
        azimuth_list.append(azimuth)
        
       
        ## caluculated data reading
        read_fname_c = "raw_data/" + fname_txt_c + "/" + fname_txt_c + "_" + fnum + ".csv"
        y_c_raw, sag_c_raw, z_c_raw = sgp.read_raw_calc(read_fname_c, full=True)
        y_min_c, y_max_c = y_c_raw.min(), y_c_raw.max()
        
        y_c_cut = sgp.y_limb_cut(y_c_raw, y_c_raw, -799, 797)
        z_c_cut = sgp.y_limb_cut(z_c_raw, y_c_raw, -799, 797)
        
        ## 前処理
        sag_m_filter = sp.ndimage.filters.median_filter(sag_m_raw, 15)
        
        ## m, cのy方向のデータ数を合わせる
        y_samp_cut = sgp.y_limb_cut(y_m_raw, y_m_raw, y_min_fit-b_max_pitch, y_max_fit+b_max_pitch)    
        sag_m_cut = sgp.y_limb_cut(sag_m_filter, y_m_raw, y_min_fit-b_max_pitch, y_max_fit+b_max_pitch)
        
        f_interp_c_cut = np.polynomial.polynomial.polyfit(y_c_raw, sag_c_raw, 4)
        sag_c_interp_cut = sgp.poly_calc(f_interp_c_cut, y_samp_cut)
    
        ## =======================================================================
        ## cをmに合わせてfitting
        
        b_init = 0
        c_init = sag_m_cut.mean() - sag_c_interp_cut.mean()
        param = [sag_m_cut, f_interp_c_cut, y_samp_cut, a]
        
        result1 = sp.optimize.minimize(sgp.fitting_func, x0=(b_init, c_init), 
                                       args=param, method="Powell")
        
        sag_c_fit = sgp.size_adjust(result1.x, f_interp_c_cut, y_m_raw, a)
        b_list.append(result1.x[0])

        sag_diff = 2*(sag_m_filter - sag_c_fit)
        
        ## =======================================================================
        ## y方向のデータ幅を20mmにして逐次積分
        polyfit_order = 2
        
        fig1 = plt.figure(figsize=(14,14))
        gs1 = fig1.add_gridspec(4,2)
        fig1.suptitle(fname_txt_m+", calclation fitting\nnum = "+fnum+" , azimuth = " + str(azimuth) + " deg")
        
        ax13 = fig1.add_subplot(gs1[0,1])
        ax13.set_ylabel("sag ( 20mm pitch)")
        ax13.grid()
        ax13.set_ylim(-2000,4000)
        
        ax14 = fig1.add_subplot(gs1[1,1])
        ax14.set_ylabel("tilt")
        ax14.grid()
        ax14.set_ylim(-3000,5000)
        
        ax15 = fig1.add_subplot(gs1[2,1])
        ax15.set_ylabel("height")
        ax15.set_ylim(-100000, 300000)
        
        ax16 = fig1.add_subplot(gs1[3,1])
        ax16.set_ylabel("height - " + str(polyfit_order)+" order polyfit")
        ax16.grid()
        ax16.set_ylim(-4000,5000)
         
        f_z_interp_c = np.polynomial.polynomial.polyfit(y_c_cut, z_c_cut, 4)
        
        y_start_num = 4
        y_start_pitch = 5
        save_num = 0
        y_min_s = y_min_c + y_start_num * y_start_pitch
        
        color=["blue", "orange", "green", "red", "lightblue", "yellow", "lightgreen", "pink"]
        
        y_samp_s_list = []
        
        for j in range(y_start_num):
            
            ## 20mmピッチの計算
            y0 = y_max_c - j * y_start_pitch
            y_samp_s = [y0]
            
            for i in range(100):
                sol = sp.optimize.newton(sgp.pythagoras, 20, args=((f_z_interp_c, y_samp_s[i]),))
                y0 = y0 - sol
                y_samp_s.append(y0)
                
                if y0 <= y_min_c:
                    break
            
            y_samp_s = y_samp_s[:-1]
            y_samp_s_list.append(y_samp_s)
            y_samp_s = np.array(y_samp_s)
            
            ## サグの誤差を20mmピッチで補間
            f_interp_diff = sp.interpolate.interp1d(y_m_raw, sag_diff, kind="linear")
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
            
            if j == save_num:
                ## savetxt ---------------------------------------------------------------
                ## xのデータ数をy_samp_sに合わせる
                f_interp_x_m = sp.interpolate.interp1d(y_m_raw, x_m_raw, kind="linear")
                x_m_interp = f_interp_x_m(y_samp_s)
                
                ## ロボ座標系からOAP座標系に回転
                x_rotate, y_rotate = np.dot(sgp.rotate_matrix(-azimuth), np.array([x_m_interp, y_samp_s]))
                
                data_id = np.arange(1, len(y_samp_s)+1) + data_id_sum
                beam_id = (k+1) * np.ones(len(y_samp_s))
                arr_save = np.array([data_id, x_rotate, y_rotate, height_diff, beam_id]).T
                with open(write_fname, "a") as wf:
                    np.savetxt(wf, arr_save, fmt=["%.0f", "%.4f","%.4f","%.9f","%.0f"])
                    np.savetxt(wf, np.array([[0], [k+1]]).T, fmt="%.0f", newline="\n\n")
                    
                
                data_id_sum = data_id_sum + len(y_samp_s)
                    
                
            
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
        ax11.plot(y_m_raw, sag_c_fit, label="calculated")
        ax11.set_title("fittig calculated")
        ax11.grid()
        
        ax12 = fig1.add_subplot(gs1[2,0])
        ax12.plot(y_m_raw, sag_diff)
        ax12.set_title("measurement - calculate")
        ax12.grid()
        #ax12.set_ylim(-600, 800)
    
        fig1.tight_layout()
        fig1.savefig(mkfolder("/"+fname_txt_m) + fname_txt_m +"_" + fnum + ".png")
        #fig1.clear()
        #fig1.clf()