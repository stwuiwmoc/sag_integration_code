#%%
from typing import List
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import csv
import pandas as pd

def mkhelp(instance):
    import inspect
    attr_list = list(instance.__dict__.keys())
    for attr in attr_list:
        if attr.startswith("_"): continue
        print(attr)
    for method in inspect.getmembers(instance, inspect.ismethod):
        if method[0].startswith("_"): continue
        print(method[0]+"()")

class Constants:
    def __init__(self, pitch_length) -> None:
        """class : Constants
        all physical length is [mm] in psm

        Parameters
        ----------
        pitch_length : float
            picth length in iterated integral（逐次積分）
        """
        self.pitch_length=pitch_length
        
    def h(self):
        mkhelp(self)

class MeasurementDataDivide:
    def __init__(self, filepath:str, skiprows:int=3) -> None:
        self.filepath=filepath
        self.raw=pd.read_csv(self.filepath, skiprows=skiprows, delimiter=" ")
        
        self.raw_df_list=self.__raw_data_divide()
        
        return
    
    def h(self):
        mkhelp(self)
    
    def __raw_data_divide(self) -> List:
        df_raw = self.raw
        df_list = []
        j=0
        for i in range(len(df_raw)):
            if df_raw.iloc[i:i+1]["Idx."].values < df_raw.iloc[i+1:i+2]["Idx."].values:
                pass
            else:
                print(j)
                df_temp = df_raw.iloc[j+1:i+1]
                df_temp["sag"] =( 2* df_temp["Out2"] - (df_temp["Out1"] +df_temp["Out3"]))/2
                
                df_list.append(df_temp)
                j = i
                    
        return df_list