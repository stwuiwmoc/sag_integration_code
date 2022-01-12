# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:22:36 2021

@author: swimc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits import axes_grid1


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

def raw_height_divide(fname):
    df_raw = pd.read_csv(fname, sep=" ", names=["idx", "x", "y", "z", "num"])
    nan_idx_list = []
    for i in range(len(df_raw)):
        if np.isnan(df_raw["num"][i]):
            nan_idx_list.append(i)
    df_list = []
    for i in range(int(len(nan_idx_list)/2)):
        start_idx = nan_idx_list[i*2]
        end_idx = nan_idx_list[i*2+1]
        df_temp = df_raw[start_idx+1:end_idx]
        df_list.append(df_temp)
    return df_list
    
def make_df_diff_list(df_before_list, df_after_list):
    df_difference_list = []
    
    for i in range(len(df_before_list)): 
        df_temp = df_before_list[i].drop(columns="z")
        df_temp["z"] = df_after_list[i]["z"] - df_before_list[i]["z"]
        df_difference_list.append(df_temp)
    return df_difference_list

def make_scatter_plot(figure, position, df_list):
    fs = 15
    cmap = cm.jet
    df  = pd.concat(df_list)
    
    ax = figure.add_subplot(position)
    ax.scatter(df["x"], df["y"], c=df["z"], cmap=cmap)
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    norm = Normalize(vmin=df["z"].min(), vmax=df["z"].max())
    mappable = cm.ScalarMappable(norm = norm, cmap = cm.jet)
    cbar = figure.colorbar(mappable, ax=ax, cax=cax)
    cbar.set_label("hoge", fontsize=fs)
    
        
        
def make_circle_path_plot(figure, position, df_list):
    ax = figure.add_subplot(position)
    df_temp = df_list[0]
    df_temp["radius"] = np.sqrt(df_temp["x"]**2 + df_temp["y"]**2)
    df_temp["theta"] = np.rad2deg(np.arccos(df_temp["x"]/df_temp["radius"]))
    ax.plot(df_temp["theta"], df_temp["z"])

if __name__ == "__main__":
    fname14 = "raw_data/1214ym870-510cir.v4.22.hei.txt"
    fname16 = "raw_data/1216ym870-510cir.v4.2.hei.txt"
    
    df_list14 = raw_height_divide(fname14)
    df_list16 = raw_height_divide(fname16)
    
    df_diff_list = make_df_diff_list(df_list14, df_list16)
    
    
    fig1 = plt.figure(figsize=(5,10))
    gs1 = fig1.add_gridspec(2,1)
    make_scatter_plot(fig1, gs1[0,0], df_list14)
    make_scatter_plot(fig1, gs1[1,0], df_list16)
    
    fig2 = plt.figure(figsize=(8,12))
    gs2 = fig2.add_gridspec(3,1)
    make_scatter_plot(fig2, gs2[0:2,0:2], df_diff_list)
    make_circle_path_plot(fig2, gs2[2,0:2], df_diff_list)
    