# %%
from typing import List
import numpy as np
from pandas.core.frame import DataFrame
import scipy as sp
from scipy import ndimage
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
    def __init__(self, Constants, DataFrame: DataFrame) -> None:
        self.consts = Constants
        self.df_raw = DataFrame

        self.radius = np.mean(np.sqrt(self.df_raw["x"] ** 2 + self.df_raw["y"] ** 2))
        self.delta_theta_per_20mm_pitch = 2 * np.rad2deg(
            np.arcsin(((self.consts.pitch_length / 2) / self.radius)))

        self.df = self.__remove_theta_duplication(theta_end_specifying_value=-19)
        self.df["sag_smooth"] = ndimage.filters.gaussian_filter(self.df["sag"], 3)

        self.idx_before_pitch, self.theta_pitch, self.sag_pitch = self.__pitch_calculation()

        self.tilt = self.__integration(self.sag_pitch)
        self.height = self.__integration(self.tilt)
        return

    def h(self) -> None:
        mkhelp(self)

    def __remove_theta_duplication(self, theta_end_specifying_value: float) -> DataFrame:
        """測定序盤と終盤のthetaの重複分を除去
        ただし、逐次で高さに直した時に2つ分使えないデータが出るので、
        その分を考慮して20mmピッチでのdelta theta 3つ分は重複させておく

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
                theta_end = df_theta.iloc[end_duplicate_last_idx] + 3 * self.delta_theta_per_20mm_pitch
                break

        for i in range(len(df_theta)):
            if df_theta.iloc[i] > theta_end:
                pass
            else:
                head_duplicate_last_idx = i - 1
                break

        df_remove_duplication = self.df_raw.iloc[head_duplicate_last_idx:end_duplicate_last_idx]
        return df_remove_duplication

    def __pitch_calculation(self):
        """20mmピッチの計算
        """
        def rate_calculation(target, before_target, after_target):
            rate = (target - before_target) / (after_target - before_target)
            return rate

        def target_calculation(rate, before_target, after_target):
            temp = rate * (after_target - before_target)
            target = temp + before_target
            return target

        df_theta = self.df["theta"]
        df_sag = self.df["sag_smooth"]

        # thetaの切り替わり位置
        theta_min_idx = self.df["theta"].idxmin()

        before_target_idx_list = [0]
        theta_pitch_list = [df_theta.iloc[0]]
        sag_pitch_list = [df_sag.iloc[0]]

        theta_target_value = df_theta.iloc[0] - self.delta_theta_per_20mm_pitch

        for i in range(len(df_theta)):
            theta_temp = df_theta.iloc[i]
            if i == theta_min_idx:
                theta_target_value += 360
                continue

            elif theta_target_value > theta_temp:

                theta_before_target_value = df_theta.iloc[i - 1]
                theta_after_target_value = theta_temp

                sag_before_target_value = df_sag.iloc[i - 1]
                sag_after_target_value = df_sag.iloc[i]

                rate_from_before_target_value = rate_calculation(theta_target_value,
                                                                 theta_before_target_value,
                                                                 theta_after_target_value)
                sag_target_value = target_calculation(rate_from_before_target_value,
                                                      sag_before_target_value,
                                                      sag_after_target_value)

                before_target_idx_list.append(i)
                theta_pitch_list.append(theta_target_value)
                sag_pitch_list.append(sag_target_value)

                theta_target_value += -self.delta_theta_per_20mm_pitch

            else:
                pass
        result = [np.array(before_target_idx_list),
                  np.array(theta_pitch_list),
                  np.array(sag_pitch_list)]
        return result

    def __integration(self, array):
        result_list = [0]
        for i in range(len(array) - 1):
            result_temp = result_list[i] + array[i]
            result_list.append(result_temp)

        return np.array(result_list)
