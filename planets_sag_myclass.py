# %%
from typing import List, Iterable
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
        self.df_raw = pd.read_csv(self.filepath,
                                  names=["x", "y", "theta", "sag"])

        self.df = self.__theta_add_sign(df_raw=self.df_raw)
        self.interpolated_function = self.__make_interpolated_function(theta=self.df["theta_signed"],
                                                                       sag=self.df["sag"])

    def h(self) -> None:
        mkhelp(self)

    def __theta_add_sign(self, df_raw: DataFrame) -> DataFrame:
        """ロボと同じ符号付の値theta_signedを追加し、外挿用にtheta_signedの両端を90deg分複製

        Parameters
        ----------
        df_raw : DataFrame
            [description]

        Returns
        -------
        DataFrame
            theta_signedを追加し外挿用に複製したもの
        """
        df_forward = df_raw[df_raw["theta"] < 270]
        df_backward = df_raw[df_raw["theta"] > 90]

        df_forward_signed = df_forward.assign(theta_signed=df_forward["theta"])
        df_backward_signed = df_backward.assign(theta_signed=(df_backward["theta"].values - 360))
        df_concat = pd.concat([df_forward_signed, df_backward_signed], axis=0)
        df_new = df_concat.reset_index(drop=True)
        return df_new

    def __make_interpolated_function(self, theta: Iterable[float], sag: Iterable[float]):
        """CirclePathIntegrationでsag補間に使うための関数作成

        Parameters
        ----------
        theta : Iterable[float]
            補間でx軸として使うtheta
        sag : Iterable[float]
            補間でy軸として使うsag

        Returns
        -------
        function
            使用時にはfunction(theta) -> 理想sag
        """

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

    def __raw_data_divide(self) -> List[DataFrame]:
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
    def __init__(self, Constants, IdealSagReading, df_measurement: DataFrame) -> None:
        self.consts = Constants
        self.df_raw = df_measurement
        self.ideal_sag_const = IdealSagReading

        self.radius = np.mean(np.sqrt(self.df_raw["x"] ** 2 + self.df_raw["y"] ** 2))
        self.delta_theta_per_20mm_pitch = 2 * np.rad2deg(
            np.arcsin(((self.consts.pitch_length / 2) / self.radius)))

        df_temp = self.__remove_theta_duplication(theta_end_specifying_value=-19)
        self.df_removed = df_temp.assign(sag_smooth=ndimage.filters.gaussian_filter(df_temp["sag"], 3))
        del df_temp

        sag_fitting_return = self.__sag_fitting(measured_theta=self.df_removed["theta"],
                                                measured_sag=self.df_removed["sag_smooth"])

        self.sag_optimize_result = sag_fitting_return["optimize_result"]
        self.sag_optimize_removing = sag_fitting_return["ideal_sag_optimized"]
        self.sag_diff = sag_fitting_return["sag_diff"]
        self.df_removed = self.df_removed.assign(sag_diff=self.sag_diff)

        pitch_calculation_result = self.__pitch_calculation(dataframe=self.df_removed)

        self.theta_pitch = pitch_calculation_result["theta"]
        self.sag_diff_pitch = pitch_calculation_result["sag_diff"]
        self.angle_from_head_pitch = pitch_calculation_result["angle_from_head"]
        self.circumference_pitch = pitch_calculation_result["circumference"]

        self.df_pitch = pd.DataFrame({"theta": self.theta_pitch,
                                      "sag_diff": self.sag_diff_pitch,
                                      "angle_from_head": self.angle_from_head_pitch,
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
        theta_array = self.df_raw["theta"].values

        for i in range(len(theta_array)):
            j = -(i + 1)
            if theta_array[j] < theta_end_specifying_value:
                pass
            else:
                end_duplicate_last_idx = j + 1
                theta_end = theta_array[end_duplicate_last_idx] + 3 * self.delta_theta_per_20mm_pitch
                break

        for i in range(len(theta_array)):
            if theta_array[i] > theta_end:
                pass
            else:
                head_duplicate_last_idx = i - 1
                break

        df_remove_duplication = self.df_raw.iloc[head_duplicate_last_idx:end_duplicate_last_idx]
        return df_remove_duplication

    def __sag_fitting(self,
                      measured_theta: Iterable[float],
                      measured_sag: Iterable[float]) -> dict:

        def make_sag_difference(theta_array: Iterable[float],
                                measured_array: Iterable[float],
                                ideal_func,
                                vertical_magn: float,
                                vertical_shift: float,
                                horizontal_shift: float) -> Iterable[float]:
            """理想サグと測定サグの差分を取る

            Parameters
            ----------
            theta_array : Iterable[float]
                theta（横軸）
            measured_array : Iterable[float]
                測定値
            ideal_func : function
                理想値を作る関数
            vertical_magn : float
                縦倍率
            vertical_shift : float
                縦ずれ
            horizontal_shift : float
                横ずれ

            Returns
            -------
            Iterable[float]
                （測定sag）-（縦ずれ、横ずれ、縦倍率を処理した理想sag）
            """

            ideal_shifted = vertical_magn * ideal_func(theta_array - horizontal_shift) + vertical_shift
            difference = measured_array - ideal_shifted
            return difference

        def minimize_function(x: List[float], params: list):
            measured_theta_, measured_sag_, ideal_sag_func_, vertical_magnification_ = params
            sag_difference = make_sag_difference(theta_array=measured_theta_,
                                                 measured_array=measured_sag_,
                                                 ideal_func=ideal_sag_func_,
                                                 vertical_magn=vertical_magnification_,
                                                 vertical_shift=x[0],
                                                 horizontal_shift=x[1])

            sigma = np.sum(sag_difference ** 2)

            return sigma

        ideal_sag_func = self.ideal_sag_const.interpolated_function

        params = [measured_theta, measured_sag, ideal_sag_func, self.consts.vertical_magnification]

        optimize_result = optimize.minimize(fun=minimize_function,
                                            x0=(0, 0),
                                            args=(params,),
                                            method="Powell")

        ideal_sag_optimized = - make_sag_difference(theta_array=measured_theta,
                                                    measured_array=np.zeros(len(measured_sag)),
                                                    ideal_func=ideal_sag_func,
                                                    vertical_magn=self.consts.vertical_magnification,
                                                    vertical_shift=optimize_result["x"][0],
                                                    horizontal_shift=optimize_result["x"][1])

        sag_difference = make_sag_difference(theta_array=measured_theta,
                                             measured_array=measured_sag,
                                             ideal_func=ideal_sag_func,
                                             vertical_magn=self.consts.vertical_magnification,
                                             vertical_shift=optimize_result["x"][0],
                                             horizontal_shift=optimize_result["x"][1])

        result = {"optimize_result": optimize_result,
                  "ideal_sag_optimized": ideal_sag_optimized,
                  "sag_diff": sag_difference}

        return result

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

        theta_array = dataframe["theta"].values
        sag_array = dataframe["sag_diff"].values

        # theta測定出力の -180 -> +180 への切り替わり位置idx
        theta_min_idx = theta_array.argmin()

        theta_pitch_list = [theta_array[0]]
        angle_from_head_pitch_list = [0]
        sag_pitch_list = [sag_array[0]]

        theta_target_value = theta_array[0] - self.delta_theta_per_20mm_pitch
        angle_from_head_value = self.delta_theta_per_20mm_pitch

        for i in range(len(theta_array)):
            # 測定出力のthetaを1つずつ取り出す
            theta_temp = theta_array[i]

            if i == theta_min_idx:
                # theta測定出力の -180 -> +180への 切り替わりへの対応
                theta_target_value += 360
                continue

            if theta_target_value > theta_temp:
                # 前回の20mmピッチの値をtheta_tempが超えたら
                # 20mmピッチの値を線形補間する
                theta_before_target_value = theta_array[i - 1]
                theta_after_target_value = theta_temp

                sag_before_target_value = sag_array[i - 1]
                sag_after_target_value = sag_array[i]

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
        angle_from_head_pitch_array = np.array(angle_from_head_pitch_list)
        circumference_from_head_pitch_array = self.radius * np.deg2rad(angle_from_head_pitch_array)

        result = {"theta": theta_pitch_array,
                  "sag_diff": sag_pitch_array,
                  "angle_from_head": angle_from_head_pitch_array,
                  "circumference": circumference_from_head_pitch_array}

        return result


class CirclePathIntegration:
    def __init__(self,
                 Constants,
                 IdealSagReading,
                 df_pitch: DataFrame,
                 height_optimize_init: list[float]) -> None:

        self.consts = Constants
        self.ideal_sag_const = IdealSagReading
        self.height_optimize_init = height_optimize_init

        self.theta = df_pitch["theta"].values
        self.circumference = df_pitch["circumference"].values
        self.sag_diff = df_pitch["sag_diff"].values

        self.integration_tilt_optimize_result = self.__integration_limb_optimize(self.sag_diff)[0]
        self.integration_height_optimize_result = self.__integration_limb_optimize(self.sag_diff)[1]
        self.tilt = self.__integration_limb_optimize(self.sag_diff)[2]
        self.height = self.__integration_limb_optimize(self.sag_diff)[3]

        self.height_optimize_result = self.__height_fitting()["optimize_result"]
        self.height_removing = self.__height_fitting()["sin_removing"]
        self.height_removed = self.__height_fitting()["height_optimized"]

        self.df_save = pd.DataFrame({"angle": df_pitch["angle_from_head"].values,
                                     "height": self.height_removed * 1e-9})

    def h(self) -> None:
        mkhelp(self)

    def __integration_limb_optimize(self, sag: Iterable[float]) -> List:
        """heightの1番目と最後の値が等しくなるように最適化して逐次積分

        Parameters
        ----------
        sag : Iterable[float]
            逐次の元にするsag

        Returns
        -------
        List
            [OptimizeResult,
            tilt,
            height]
        """
        def integration(array: Iterable[float],
                        vertical_shift: float,
                        result_head_value: float) -> Iterable[float]:
            """逐次積分

            Parameters
            ----------
            array : Iterable[float]
                逐次元の1d-array
            vertical_shift : float
                縦ずれ
            result_head_value : float
                逐次結果の1番目に入れる値

            Returns
            -------
            Iterable[float]
                逐次結果
            """
            result_list = [result_head_value]
            array_shifted = array + vertical_shift

            for i in range(len(array) - 1):
                result_temp = result_list[i] + array_shifted[i]
                result_list.append(result_temp)

            result_array = np.array(result_list, dtype=float)

            return result_array

        def minimize_function(x: List[float], params: list) -> float:
            """optimize.minimizeの引数として渡す関数

            Parameters
            ----------
            x : List[float]
                フィッティングパラメータ（tiltでの逐次積分の1番目の値）
            params : list
                逐次の元のsag

            Returns
            -------
            float
                heightの1番目の値と最後の値の差分を0（最小）にする
            """
            input_array = params[0]

            output_array = integration(array=input_array,
                                       vertical_shift=x[0],
                                       result_head_value=x[1])

            sigma = (output_array[0] - output_array[-1]) ** 2
            return sigma

        params_tilt = [sag]

        tilt_optimize_result = optimize.minimize(fun=minimize_function,
                                                 x0=(0, 0),
                                                 args=(params_tilt, ),
                                                 method="Powell")

        tilt_optimized = integration(array=sag,
                                     vertical_shift=tilt_optimize_result["x"][0],
                                     result_head_value=tilt_optimize_result["x"][1])

        params_height = [tilt_optimized]
        height_optimize_result = optimize.minimize(fun=minimize_function,
                                                   x0=(0, 0),
                                                   args=(params_height, ),
                                                   method="Powell")

        height_optimized = integration(array=tilt_optimized,
                                       vertical_shift=height_optimize_result["x"][0],
                                       result_head_value=height_optimize_result["x"][1])

        result = [tilt_optimize_result,
                  height_optimize_result,
                  tilt_optimized,
                  height_optimized]

        return result

    def __height_fitting(self):
        def remove_sin_function(x_array: Iterable[float],
                                y_array: Iterable[float],
                                y_magn: float,
                                x_shift: float,
                                y_shift: float) -> Iterable[float]:
            """縦ずれ、横ずれ、縦倍率を加えたsinを除去

            Parameters
            ----------
            x_array : Iterable[float]
                x軸（theta）
            y_array : Iterable[float]
                y軸（height）
            y_magn : float
                縦倍率
            x_shift : float
                横ずれ
            y_shift : float
                縦ずれ

            Returns
            -------
            Iterable[float]
                sinを除去したheight
            """

            sin_array = y_magn * np.sin(np.deg2rad(x_array - x_shift)) + y_shift
            y_diff = y_array - sin_array
            return y_diff

        def minimize_function(x: List[float], params_: list) -> float:
            theta_, height_ = params_
            difference = remove_sin_function(x_array=theta_,
                                             y_array=height_,
                                             y_magn=x[0],
                                             x_shift=x[1],
                                             y_shift=x[2])
            sigma = np.sum(difference ** 2)
            return sigma

        theta = self.theta
        height = self.height
        init = self.height_optimize_init

        params = [theta, height]

        optimize_result = optimize.minimize(fun=minimize_function,
                                            x0=init,
                                            args=(params,),
                                            method="Nelder-Mead")

        sin_removing = - remove_sin_function(theta,
                                             np.zeros(len(theta)),
                                             optimize_result["x"][0],
                                             optimize_result["x"][1],
                                             optimize_result["x"][2])

        height_optimized = remove_sin_function(theta,
                                               height,
                                               optimize_result["x"][0],
                                               optimize_result["x"][1],
                                               optimize_result["x"][2])

        result_dict = {"optimize_result": optimize_result,
                       "sin_removing": sin_removing,
                       "height_optimized": height_optimized}

        return result_dict
