# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:04:43 2021

@author: swimc
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

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
    
    fname_txt = "0922xm130_3deg"
    data_id_sum = 0
    write_fname = mkfolder("/"+fname_txt) + fname_txt + "_height.txt"
    
    start = time.time()
    for k in range(120):
        print(k)
        print(time.time() - start)
        fnum = str(k).zfill(3)
        ## measurement data reading
        read_fname = "raw_data/"+fname_txt+"/"+fname_txt+"_" + fnum + ".txt"
        y_m_raw, sag_m_raw, x_m_raw, azimuth_raw = sgp.read_raw_measurement(read_fname, full=True)
        azimuth = int(azimuth_raw[0] + 9.893)
       
        ## =======================================================================
        ## 前処理
        sag_m_filter = sp.ndimage.filters.median_filter(sag_m_raw, 15)
        #sag_m_filter = sag_m_raw
        
        ## m, cのy方向のデータ数を合わせる
        y_samp_cut = sgp.y_limb_cut(y_m_raw, y_m_raw, y_min-b_max_pitch, y_max+b_max_pitch)    
        sag_m_cut = sgp.y_limb_cut(sag_m_filter, y_m_raw, y_min-b_max_pitch, y_max+b_max_pitch)
        
        f_sag_c_fit = np.polynomial.polynomial.polyfit(y_samp_cut, sag_m_cut, 4)
        sag_c_fit = sgp.poly_calc(f_sag_c_fit, y_samp_cut)
        
        sag_diff = 2*(sag_m_cut - sag_c_fit)
        
        ## =======================================================================
        ## y方向のデータ幅を20mmにして逐次積分
        polyfit_order = 6
        
        fig1 = plt.figure(figsize=(14,14))
        gs1 = fig1.add_gridspec(4,2)
        fig1.suptitle(fname_txt+"\nno calc, num = "+fnum+" , azimuth = " + str(azimuth) + " deg")
        
        ax13 = fig1.add_subplot(gs1[0,1])
        ax13.set_ylabel("sag ( 20mm pitch)")
        ax13.grid()
        ax13.set_ylim(-800,800)
        
        ax14 = fig1.add_subplot(gs1[1,1])
        ax14.set_ylabel("tilt")
        ax14.grid()
        ax14.set_ylim(-1200,800)
        
        ax15 = fig1.add_subplot(gs1[2,1])
        ax15.set_ylabel("height")
        ax15.set_ylim(-40000, 15000)
        
        ax16 = fig1.add_subplot(gs1[3,1])
        ax16.set_ylabel("height - " + str(polyfit_order)+" order polyfit")
        ax16.grid()
        ax16.set_ylim(-1200,1200)
         
        y_start_num = 4
        y_start_pitch = 5
        save_num = 0
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
        ax11.plot(y_samp_cut, sag_c_fit, label="calculated")
        ax11.set_title("fittig calculated")
        ax11.grid()
        
        ax12 = fig1.add_subplot(gs1[2,0])
        ax12.plot(y_samp_cut, sag_diff)
        ax12.set_title("measurement - calculate")
        ax12.grid()
        ax12.set_ylim(-600, 800)
    
        fig1.tight_layout()
        fig1.savefig(mkfolder("/"+fname_txt) + fname_txt +"_" + fnum + ".png")
        fig1.clear()
        fig1.clf()