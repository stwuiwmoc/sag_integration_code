# %%
from typing import List
import numpy as np
from pandas.core.frame import DataFrame
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import csv
import pandas as pd


def mkhelp(instance):
    import inspect
    attr_list = list(instance.__dict__.keys())
    for attr in attr_list:
        if attr.startswith("_"):
            continue
        print(attr)
    for method in inspect.getmembers(instance, inspect.ismethod):
        if method[0].startswith("_"):
            continue
        print(method[0] + "()")


class Constants:
    def __init__(self, pitch_length) -> None:
        """class : Constants
        all physical length is [mm] in psm

        Parameters
        ----------
        pitch_length : float
            picth length in iterated integral（逐次積分）
        """
        self.pitch_length = pitch_length

    def h(self) -> None:
        mkhelp(self)


class MeasurementDataDivide:
    def __init__(self, filepath: str, skiprows: int = 3) -> None:
        """class : MeasurementDataDivede

        Parameters
        ----------
        filepath : str
            filepath of measurement raw data
        skiprows : int, optional
            skip row number in pd.read_csv()
            by default 3
        """
        self.filepath = filepath
        self.raw = pd.read_csv(self.filepath, skiprows=skiprows, delimiter=" ")

        self.raw_df_list = self.__raw_data_divide()

        return

    def h(self) -> None:
        mkhelp(self)

    def __raw_data_divide(self) -> List:
        df_raw = self.raw
        df_list = []
        j = 0
        for i in range(len(df_raw)):
            if df_raw.iloc[i:i + 1]["Idx."].values < df_raw.iloc[i + 1:i + 2]["Idx."].values:
                pass
            else:
                print(j)
                df_temp = df_raw.iloc[j + 1:i + 1]
                df_temp["sag"] = (2 * df_temp["Out2"] - (df_temp["Out1"] + df_temp["Out3"])) / 2

                df_list.append(df_temp)
                j = i

        return df_list


class CirclePathIntegration:
    def __init__(self, DataFrame: DataFrame) -> None:
        self.df_raw = DataFrame
        self.df = self.__remove_theta_duplication(theta_end_specifying_value=-19)
        return

    def h(self) -> None:
        mkhelp(self)

    def __remove_theta_duplication(self, theta_end_specifying_value: float) -> DataFrame:
        """測定序盤と終盤のthetaの重複分を除去

        Parameters
        ----------
        theta_end_specifying_value : float
            終盤で同じthetaが出力される部分があるので、どのthetaまでで切るか
            -15~-19くらいが良いと思われる

        Returns
        -------
        df_remove_duplication : DataFrame
            df_rawからthetaの重複部分を除去したもの
        """
        df_theta = self.df_raw["theta"]

        for i in range(len(df_theta)):
            j = -(i + 1)
            if df_theta.iloc[j] < theta_end_specifying_value:
                pass
            else:
                end_duplicate_last_idx = j + 1
                theta_end = df_theta.iloc[end_duplicate_last_idx]
                break

        for i in range(len(df_theta)):
            if df_theta.iloc[i] > theta_end:
                pass
            else:
                head_duplicate_last_idx = i - 1
                break

        df_remove_duplication = self.df_raw.iloc[head_duplicate_last_idx:end_duplicate_last_idx]
        return df_remove_duplication
