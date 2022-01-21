# %%
from typing import List
import numpy as np
from pandas.core.frame import DataFrame
from scipy import ndimage
from scipy import optimize
from scipy import interpolate
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
    def __init__(self, pitch_length: float, vertical_magnification: float = 0.99809) -> None:
        """class : Constants
            all physical length is [mm] in psm

        Parameters
        ----------
        pitch_length : float
            picth length in iterated integral（逐次積分）
        vertical_magnification : float, optional
            縦倍率, by default 0.99809
        """
        self.pitch_length = pitch_length
        self.vertical_magnification = vertical_magnification

    def h(self) -> None:
        mkhelp(self)


class IdealSagReading:
    def __init__(self, filepath_ideal_sag: str) -> None:
        """Class : IdealSagReading
        鍵谷先生の円環パスでの理想サグを読み込み、補間用の関数を作る

        Parameters
        ----------
        filepath_ideal_sag : str
            理想サグのpath
        """
        self.filepath = filepath_ideal_sag
        self.df = self.__csv_reading()
        self.interpolated_function = self.__make_interpolated_function(theta=self.df["theta_signed"])

    def h(self) -> None:
        mkhelp(self)

    def __csv_reading(self):
        df_raw = pd.read_csv(self.filepath,
                             names=["x", "y", "theta", "sag"])

        # 測定出力に合わせた符号付きthetaを追加
        theta_array = df_raw["theta"].values
        theta_signed_array = np.where(theta_array <= 180,
                                      theta_array,
                                      theta_array - 360)

        df_raw["theta_signed"] = theta_signed_array
        return df_raw

    def __make_interpolated_function(self, theta: float):
        """CirclePathIntegrationでsag補間に使うための関数作成

        Parameters
        ----------
        theta : float
            補間でx軸として使うtheta

        Returns
        -------
        function
            使用時にはfunction(theta) -> 理想sag
        """
        sag = self.df["sag"]
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
                df_temp = df_temp.assign(sag=((2 * df_temp["Out2"] - (df_temp["Out1"] + df_temp["Out3"])) / 2))

                df_list.append(df_temp)
                j = i

        return df_list


class ConnectedSagReading:
    def __init__(self, filepath: str):
        self.filepath = filepath
        columns = ["Idx.", "x", "y", "theta", "Out1", "Out2", "Out3", "sag"]
        self.df_raw = pd.read_csv(self.filepath,
                                  names=columns)
        self.df_float = self.df_raw.astype(float)

    def h(self) -> None:
        mkhelp(self)


class CirclePathPitch:
    def __init__(self, Constants, df_measurement: DataFrame) -> None:
        self.consts = Constants
        self.df_raw = df_measurement

        self.radius = np.mean(np.sqrt(self.df_raw["x"] ** 2 + self.df_raw["y"] ** 2))
        self.delta_theta_per_20mm_pitch = 2 * np.rad2deg(
            np.arcsin(((self.consts.pitch_length / 2) / self.radius)))

        df_temp = self.__remove_theta_duplication(theta_end_specifying_value=-19)
        self.df_removed = df_temp.assign(sag_smooth=ndimage.filters.gaussian_filter(df_temp["sag"], 3))
        del df_temp

        pitch_calculation_result = self.__pitch_calculation(dataframe=self.df_removed)
        self.theta_pitch = pitch_calculation_result[0]
        self.sag_pitch = pitch_calculation_result[1]
        self.circumference_pitch = pitch_calculation_result[2]
        self.df_pitch = pd.DataFrame({"theta": self.theta_pitch,
                                      "sag": self.sag_pitch,
                                      "circumference": self.circumference_pitch})
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

    def __pitch_calculation(self, dataframe: DataFrame):
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

        df_theta = dataframe["theta"]
        df_sag = dataframe["sag_smooth"]

        # theta測定出力の -180 -> +180 への切り替わり位置idx
        theta_min_idx = dataframe["theta"].idxmin()

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


class CirclePathIntegration:
    def __init__(self, Constants, IdealSagReading, df_pitch: DataFrame, integration_optimize_init: float) -> None:
        self.consts = Constants
        self.ideal_sag = IdealSagReading
        self.integration_optimize_init = integration_optimize_init

        self.theta = df_pitch["theta"].values
        self.circumference = df_pitch["circumference"].values
        self.sag = df_pitch["sag"].values

        self.sag_optimize_result = self.__sag_fitting()[0]
        self.sag_optimize_removing = self.__sag_fitting()[1]
        self.sag_diff = self.__sag_fitting()[2]

        self.integration_optimize_result = self.__integration_limb_optimize(self.sag_diff)[0]
        self.tilt = self.__integration_limb_optimize(self.sag_diff)[1]
        self.height = self.__integration_limb_optimize(self.sag_diff)[2]

        self.res, self.removed, self.sin = self.__height_fitting()
        return

    def h(self) -> None:
        mkhelp(self)

    def __sag_fitting(self):
        def make_sag_difference(measured: float, ideal: float, vertical_magn: float, vertical_shift: float) -> float:
            """理想サグと測定サグの差分をとる

            Parameters
            ----------
            measured : float
                測定値
            ideal : float
                理想値
            vertical_magn : float
                縦倍率
            vertical_shift : float
                縦ずれ

            Returns
            -------
            float
                （測定sag）-（縦ずれ、縦倍率を処理した理想sag）
            """
            ideal_shifted = vertical_magn * ideal + vertical_shift
            difference = measured - ideal_shifted
            return difference

        def minimize_function(x, params):
            measured_sag_, ideal_sag_, vertical_magnification_ = params
            sag_difference = make_sag_difference(measured=measured_sag_,
                                                 ideal=ideal_sag_,
                                                 vertical_magn=vertical_magnification_,
                                                 vertical_shift=x)

            sigma = np.sum(sag_difference ** 2)

            return sigma

        measured_theta = self.theta
        measured_sag = self.sag
        ideal_sag = self.ideal_sag.interpolated_function(measured_theta)

        params = [measured_sag, ideal_sag, self.consts.vertical_magnification]

        optimize_result = optimize.minimize(fun=minimize_function,
                                            x0=0,
                                            args=(params,),
                                            method="Powell")

        ideal_sag_optimized = - make_sag_difference(measured=np.zeros(len(measured_sag)),
                                                    ideal=ideal_sag,
                                                    vertical_magn=self.consts.vertical_magnification,
                                                    vertical_shift=optimize_result["x"][0])

        sag_difference = make_sag_difference(measured=measured_sag,
                                             ideal=ideal_sag,
                                             vertical_magn=self.consts.vertical_magnification,
                                             vertical_shift=optimize_result["x"][0])

        return optimize_result, ideal_sag_optimized, sag_difference

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

            result_array = np.array(result_list, dtype=float)

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

    def __height_fitting(self):
        def remove_sin_function(x_array, y_array, y_magn, x_shift, y_shift):
            sin_array = y_magn * np.sin(np.deg2rad(x_array - x_shift)) + y_shift
            y_diff = y_array - sin_array
            return y_diff

        def minimize_function(x, params_):
            theta_, height_ = params_
            difference = remove_sin_function(x_array=theta_,
                                             y_array=height_,
                                             y_magn=x[0],
                                             x_shift=x[1],
                                             y_shift=x[2])
            sigma = np.sum(difference ** 2)
            return sigma

        def constraints_function(x):
            theta_, height_ = params
            difference = remove_sin_function(x_array=theta_,
                                             y_array=height_,
                                             y_magn=x[0],
                                             x_shift=x[1],
                                             y_shift=x[2])
            difference_of_head_and_end = difference[0] - difference[-1]
            formula = 1e-20 - abs(difference_of_head_and_end)

            return formula

        theta = self.theta
        height = self.height

        params = [theta, height]

        cons = ({"type": "ineq", "fun": constraints_function})

        optimize_result = optimize.minimize(fun=minimize_function,
                                            x0=[2e5, 0, 1e4],
                                            args=(params,),
                                            constraints=cons,
                                            method="COBYLA")

        sin_removing = - remove_sin_function(theta,
                                             np.zeros(len(theta)),
                                             optimize_result["x"][0],
                                             optimize_result["x"][1],
                                             optimize_result["x"][2])

        height_optimized = remove_sin_function(theta,
                                               height,
                                               optimize_result["x"][0],
                                               optimize_result["x"][1],
                                               optimize_result["x"][0])

        return optimize_result, sin_removing, height_optimized
