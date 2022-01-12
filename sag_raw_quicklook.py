# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:08:55 2021

@author: swimc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def raw_data_divide(fname):
    df_raw = pd.read_csv(fname, sep=" ", skiprows=2)
    df_list = []
    j = 0
    for i in range(len(df_raw)):
        if df_raw[i:i+1]["Idx."].values < df_raw[i+1:i+2]["Idx."].values:
            pass
        else:
            print(j)
            df_temp = df_raw[j+1:i+1]
            df_temp["sag"] =2* df_temp["Out2"] - (df_temp["Out1"] +df_temp["Out3"])
            
            df_list.append(df_temp)
            j = i
    
    return df_list

def make_df_diff_list(df_list):
    df_diff_list = []
    for i in range(len(df_list)):
        df_temp = df_list[i]
        df_diff_temp = df_temp.diff()
        df_diff_temp = df_diff_temp.rename(columns={"Out1":"Out1_diff",
                                                    "Out2":"Out2_diff",
                                                    "Out3":"Out3_diff"})
        
        df_diff_temp = df_temp.join([df_diff_temp["Out1_diff"],
                                     df_diff_temp["Out2_diff"],
                                     df_diff_temp["Out3_diff"],])
        df_diff_list.append(df_diff_temp)
    return df_diff_list

def make_df_list_plot(df1_list, df2_list, ax, out_name, magn):
    linewidth = 1
    
    ax.set_title(out_name + " ( red : before cleaning / black : after cleaning )")
    ax.set_ylabel("theta")
    ax.set_ylim(-75,50)
    
    for i in range(len(df1_list)):
        df1_temp = df1_list[i]
        df2_temp = df2_list[i]
        ax.plot(df1_temp["Idx."], df1_temp[out_name]*magn + df1_temp["theta"], 
                linewidth=linewidth, color="red")
        ax.plot(df2_temp["Idx."], df2_temp[out_name]*magn + df2_temp["theta"], 
                linewidth=linewidth, color="black")
        
    return ax
            

if __name__ == '__main__':
    # 標準モードのデータ
    fname_data1 = "raw_data/sag_rawdata/1214/RT_202112140919_FTP.txt"
    
    # 干渉モードのデータ
    fname_data2 = "raw_data/sag_rawdata/1214/RT_202112140947_FTP.txt"


    df1_list = raw_data_divide(fname_data1)
    df2_list = raw_data_divide(fname_data2)
    
    df1_diff_list = make_df_diff_list(df1_list)
    df2_diff_list = make_df_diff_list(df2_list)
    
    fig1 = plt.figure(figsize=(20,10))
    gs1 = fig1.add_gridspec(2,4)
    ax12 = fig1.add_subplot(gs1[1,0])
    ax12 = make_df_list_plot(df1_diff_list, df2_diff_list, ax12, "Out1_diff", magn=1e-1)
    ax11 = fig1.add_subplot(gs1[0,0])
    ax11 = make_df_list_plot(df1_diff_list, df2_diff_list, ax11, "Out1", magn=1e-2)
    
    ax22 = fig1.add_subplot(gs1[1,1])
    ax22 = make_df_list_plot(df1_diff_list, df2_diff_list, ax22, "Out2_diff", magn=1e-1)
    ax21 = fig1.add_subplot(gs1[0,1])
    ax21 = make_df_list_plot(df1_diff_list, df2_diff_list, ax21, "Out2", magn=1e-2)
    
    ax32 = fig1.add_subplot(gs1[1,2])
    ax32 = make_df_list_plot(df1_diff_list, df2_diff_list, ax32, "Out3_diff", magn=1e-1)
    ax31 = fig1.add_subplot(gs1[0,2])
    ax31 = make_df_list_plot(df1_diff_list, df2_diff_list, ax31, "Out3", magn=1e-2)
    
    ax41 = fig1.add_subplot(gs1[0,3])
    ax41 = make_df_list_plot(df1_diff_list, df2_diff_list, ax41, "sag", magn=1e-2)
    
    fig1.tight_layout()
    