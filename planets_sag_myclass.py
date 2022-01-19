# %%
from typing import List
import numpy as np
from pandas.core.frame import DataFrame
from scipy import ndimage
from scipy import optimize
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


class IdealSagReading:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.df_raw = self.__csv_reading()
        self.interpolated_function = self.__make_interpolated_function()

    def h(self) -> None:
        mkhelp(self)

    def __csv_reading(self):
        raw = pd.read_csv(self.filepath,
                          names=["x", "y", "theta", "sag"])
        return raw

    def __make_interpolated_function(self):
        theta = self.df_raw["theta"]
        sag = self.df_raw["sag"]
        interpolated_function = interpolate.interp1d(x=theta,
                                                     y=sag,
                                                     kind="cubic",
                                                     fill_value="extrapolate")
        return interpolated_function


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
    def __init__(self, Constants, DataFrame: DataFrame, integration_optimize_init: float) -> None:
        self.consts = Constants
        self.df_raw = DataFrame
        self.integration_optimize_init = integration_optimize_init

        self.radius = np.mean(np.sqrt(self.df_raw["x"] ** 2 + self.df_raw["y"] ** 2))
        self.delta_theta_per_20mm_pitch = 2 * np.rad2deg(
            np.arcsin(((self.consts.pitch_length / 2) / self.radius)))

        self.df = self.__remove_theta_duplication(theta_end_specifying_value=-19)
        self.df["sag_smooth"] = ndimage.filters.gaussian_filter(self.df["sag"], 3)

        self.theta_pitch, self.sag_pitch, self.circumference_pitch = self.__pitch_calculation()

        self.optimize_result, self.tilt, self.height = self.__integration_limb_optimize(self.sag_pitch)
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
        """20mmピッチの計算と、末尾ではみ出る部分の処理
        """

        def linear_interpolation(x_target: float, x_A: float, x_B: float, y_A: float, y_B: float) -> float:
            """点A(x_A, y_A) と点B(x_B, y_B) の2点間の単純な線形補間
                y - y_A = n * (x - x_A)
                但し n = (y_B - y_A) / (x_B - x_A)

            Parameters
            ----------
            x_target : float
                線形補間したいx座標
            x_A : float
                targetの一つ前の点のx座標
            x_B : float
                targetの一つ後の点のx座標
            y_A : float
                targetの一つ前の点のy座標
            y_B : float
                targetの一つ後の点のy座標

            Returns
            -------
            y: float
                線形補間されたy座標
            """
            n_tilt = (y_B - y_A) / (x_B - x_A)
            y = n_tilt * (x_target - x_A) + y_A
            return y

        df_theta = self.df["theta"]
        df_sag = self.df["sag_smooth"]

        # theta測定出力の -180 -> +180 への切り替わり位置idx
        theta_min_idx = self.df["theta"].idxmin()

        theta_pitch_list = [df_theta.iloc[0]]
        angle_from_head_pitch_list = [0]
        sag_pitch_list = [df_sag.iloc[0]]

        theta_target_value = df_theta.iloc[0] - self.delta_theta_per_20mm_pitch
        angle_from_head_value = self.delta_theta_per_20mm_pitch

        for i in range(len(df_theta)):
            # 測定出力のthetaを1つずつ取り出す
            theta_temp = df_theta.iloc[i]

            if i == theta_min_idx:
                # theta測定出力の -180 -> +180への 切り替わりへの対応
                theta_target_value += 360
                continue

            elif theta_target_value > theta_temp:
                # 前回の20mmピッチの値をtheta_tempが超えたら
                # 20mmピッチの値を線形補間する
                theta_before_target_value = df_theta.iloc[i - 1]
                theta_after_target_value = theta_temp

                sag_before_target_value = df_sag.iloc[i - 1]
                sag_after_target_value = df_sag.iloc[i]

                sag_target_value = linear_interpolation(x_target=theta_target_value,
                                                        x_A=theta_before_target_value,
                                                        x_B=theta_after_target_value,
                                                        y_A=sag_before_target_value,
                                                        y_B=sag_after_target_value)

                if angle_from_head_value < 360:
                    # pitchの開始点から1周回転するまでは、
                    # 20mmピッチの計算結果をlistに追加して次の20mmピッチの計算へ

                    theta_pitch_list.append(theta_target_value)
                    angle_from_head_pitch_list.append(angle_from_head_value)
                    sag_pitch_list.append(sag_target_value)

                    theta_target_value += -self.delta_theta_per_20mm_pitch
                    angle_from_head_value += self.delta_theta_per_20mm_pitch
                    continue

                else:
                    # 20mmピッチの最後はピッタリ360degにならないので、
                    # 末尾だけ360degで終わるように線形補間してループ終了
                    angle_end_value = 360

                    angle_before_end_value = angle_from_head_pitch_list[-1]
                    angle_after_end_value = angle_from_head_value

                    theta_before_end_value = theta_pitch_list[-1]
                    theta_after_end_value = theta_target_value

                    sag_before_end_value = sag_pitch_list[-1]
                    sag_after_end_value = sag_target_value

                    theta_end_value = linear_interpolation(x_target=angle_end_value,
                                                           x_A=angle_before_end_value,
                                                           x_B=angle_after_end_value,
                                                           y_A=theta_before_end_value,
                                                           y_B=theta_after_end_value)

                    sag_end_value = linear_interpolation(x_target=angle_end_value,
                                                         x_A=angle_before_end_value,
                                                         x_B=angle_after_end_value,
                                                         y_A=sag_before_end_value,
                                                         y_B=sag_after_end_value)

                    theta_pitch_list.append(theta_end_value)
                    angle_from_head_pitch_list.append(angle_end_value)
                    sag_pitch_list.append(sag_end_value)
                    break

            else:
                pass

        theta_pitch_array = np.array(theta_pitch_list)
        sag_pitch_array = np.array(sag_pitch_list)
        circumference_from_head_pitch_array = self.radius * np.deg2rad(np.array(angle_from_head_pitch_list))

        result = [theta_pitch_array,
                  sag_pitch_array,
                  circumference_from_head_pitch_array]

        return result

    def __integration_limb_optimize(self, sag: float) -> list:
        """heightの1番目と最後の値が等しくなるように最適化して逐次積分

        Parameters
        ----------
        sag : float
            逐次の元にするsag

        Returns
        -------
        list
            [OptimizeResult,
            tilt,
            height]
        """
        def integration(array: float, result_head_value: float) -> float:
            """逐次積分

            Parameters
            ----------
            array : float
                逐次元の1d-array
            result_head_value : float
                逐次結果の1番目に入れる値

            Returns
            -------
            float
                逐次結果
            """
            result_list = [result_head_value]

            for i in range(len(array) - 1):
                result_temp = result_list[i] + array[i]
                result_list.append(result_temp)

            result_array = np.array(result_list)

            return result_array

        def minimize_function(x: list, params: list) -> float:
            """optimize.minimizeの引数として渡す関数

            Parameters
            ----------
            x : list
                フィッティングパラメータ（tiltでの逐次積分の1番目の値）
            params : list
                逐次の元のsag

            Returns
            -------
            float
                heightの1番目の値と最後の値の差分を0（最小）にする
            """
            sag_in_optimize = params[0]

            tilt_in_optimize = integration(sag_in_optimize, x)
            height_in_optimize = integration(tilt_in_optimize, 0)

            sigma = (height_in_optimize[0] - height_in_optimize[-1]) ** 2
            return sigma

        params = [sag]

        optimize_result = optimize.minimize(fun=minimize_function,
                                            x0=[self.integration_optimize_init],
                                            args=(params, ),
                                            method="Powell")

        tilt_optimized = integration(sag, optimize_result["x"][0])
        height_optimized = integration(tilt_optimized, 0)

        result = [optimize_result,
                  tilt_optimized,
                  height_optimized]

        return result
