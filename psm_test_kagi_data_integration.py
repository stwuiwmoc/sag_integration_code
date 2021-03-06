# %%
import planets_sag_myclass as psm
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd


def mkfolder(suffix=""):
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


if __name__ == "__main__":
    importlib.reload(psm)

    CONSTS = psm.Constants(pitch_length=20)  # [mm]
    IDEAL_SAG = psm.IdealSagReading(filepath_ideal_sag="raw_data/calcCirSagDist01.csv")

# %%
    serial = "0117e"
    input_fname = "raw_data/sag_rawdata/0117/" + serial + ".jc_i000.txt"
    mes = psm.ConnectedSagReading(filepath=input_fname)

    importlib.reload(psm)

    pitch = psm.CirclePathPitch(Constants=CONSTS,
                                IdealSagReading=IDEAL_SAG,
                                df_measurement=mes.df_float)

    itg = psm.CirclePathIntegration(Constants=CONSTS,
                                    IdealSagReading=IDEAL_SAG,
                                    df_pitch=pitch.df_pitch,
                                    height_optimize_init=[-4e4, 70, -4e4])

    fig1 = plt.figure(figsize=(12, 8))
    gs1 = fig1.add_gridspec(3, 1)
    fig1.suptitle(serial)

    ax11 = fig1.add_subplot(gs1[0, 0])
    ax11.scatter(mes.df_float["theta"], mes.df_float["Out1"], s=1, label="s1")
    ax11.scatter(mes.df_float["theta"], mes.df_float["Out2"], s=1, label="s2")
    ax11.scatter(mes.df_float["theta"], mes.df_float["Out3"], s=1, label="s3")
    ax11.legend()

    ax12 = fig1.add_subplot(gs1[1, 0])
    ax12.plot(pitch.df_removed["theta"], pitch.df_removed["sag_smooth"], label="measured_sag_smoothed")
    ax12.plot(pitch.df_removed["theta"], pitch.sag_optimize_removing, label="ideal_sag_fitted")
    ax12.set_ylabel("sag [nm]")
    ax12.legend()

    ax13 = fig1.add_subplot(gs1[2, 0])
    ax13.plot(pitch.df_removed["theta"], pitch.df_removed["sag_diff"])
    ax13.scatter(pitch.theta_pitch, pitch.sag_diff_pitch, s=10, color="red", label="20mm pitch points")
    ax13.grid()
    ax13.set_ylabel("sag deviation [nm]")
    ax13.set_xlabel("robot-coordinate theta[deg]")
    ax13.legend()
    fig1.tight_layout()
    fig1.savefig(mkfolder() + serial + "_fig1.png")

    fig2 = plt.figure(figsize=(12, 12))
    gs2 = fig2.add_gridspec(4, 1)
    fig2.suptitle(serial)

    ax21 = fig2.add_subplot(gs2[0, 0])
    ax21.plot(-itg.circumference, itg.sag_diff)
    ax21.scatter(-pitch.circumference_pitch, pitch.sag_diff_pitch, s=10, color="red", label="20mm pitch points")
    ax21.grid()
    ax21.legend()
    ax21.set_ylabel("sag deviation [nm]")

    ax23 = fig2.add_subplot(gs2[1, 0])
    ax23.plot(-itg.circumference, itg.tilt)
    ax23.grid()
    ax23.set_ylim(-2000, 2000)
    ax23.set_ylabel("gradient [nm / pitch]")

    ax24 = fig2.add_subplot(gs2[2, 0])
    ax24.plot(-itg.circumference, itg.height, label="height")
    ax24.plot(-itg.circumference, itg.height_removing, label="sin_fitting")
    ax24.grid()
    ax24.legend()
    ax24.set_ylabel("height [nm]")

    ax25 = fig2.add_subplot(gs2[3, 0])
    ax25.plot(-itg.circumference, itg.height_removed)
    ax25.grid()
    ax25.set_ylabel("height_deviation [nm]")
    ax25.set_xlabel("robot-arm scanning coordinate [mm]")

    fig2.tight_layout()
    fig2.savefig(mkfolder() + serial + "_fig2.png")

    save_fname = mkfolder() + serial + "_height.csv"
    itg.df_save.to_csv(save_fname, index=False)
