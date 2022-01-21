# %%
import planets_sag_myclass as psm
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd

if __name__ == "__main__":
    importlib.reload(psm)

    CONSTS = psm.Constants(pitch_length=20)  # [mm]
    IDEAL_SAG = psm.IdealSagReading(filepath_ideal_sag="raw_data/calcCirSagDist01.csv")
    mes = psm.ConnectedSagReading(filepath="raw_data/sag_rawdata/0117/0117e.jc_i000.txt")

# %%
    importlib.reload(psm)

    pitch = psm.CirclePathPitch(Constants=CONSTS,
                                df_measurement=mes.df_float)

    itg = psm.CirclePathIntegration(Constants=CONSTS,
                                    IdealSagReading=IDEAL_SAG,
                                    df_pitch=pitch.df_pitch,
                                    integration_optimize_init=-5e4,
                                    height_optimize_init=[-3.5e4, 70, -3.5e4])

# %%
    fig1 = plt.figure(figsize=(12, 4))

    gs1 = fig1.add_gridspec(2, 1)
    ax11 = fig1.add_subplot(gs1[0, 0])
    ax11.scatter(mes.df_float["theta"], mes.df_float["Out1"], s=1, label="s1")
    ax11.scatter(mes.df_float["theta"], mes.df_float["Out2"], s=1, label="s2")
    ax11.scatter(mes.df_float["theta"], mes.df_float["Out3"], s=1, label="s2")
    ax11.legend()

    ax12 = fig1.add_subplot(gs1[1, 0])
    ax12.plot(itg.theta, itg.sag)
    ax12.plot(itg.theta, itg.sag_optimize_removing)

    fig2 = plt.figure(figsize=(12, 12))
    gs2 = fig2.add_gridspec(5, 1)

    ax21 = fig2.add_subplot(gs2[0, 0])
    ax21.plot(-itg.circumference, itg.sag)
    ax21.plot(-itg.circumference, itg.sag_optimize_removing)

    ax22 = fig2.add_subplot(gs2[1, 0])
    ax22.plot(-itg.circumference, itg.sag_diff)

    ax23 = fig2.add_subplot(gs2[2, 0])
    ax23.plot(-itg.circumference, itg.tilt)

    ax24 = fig2.add_subplot(gs2[3, 0])
    ax24.plot(-itg.circumference, itg.height)

    ax25 = fig2.add_subplot(gs2[4, 0])
