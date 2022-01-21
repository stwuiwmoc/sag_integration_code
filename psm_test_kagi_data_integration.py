# %%
import planets_sag_myclass as psm
import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd

if __name__ == "__main__":
    importlib.reload(psm)

    CONSTS = psm.Constants(pitch_length=20)  # [mm]
    mes = psm.ConnectedSagReading(filepath="raw_data/sag_rawdata/0117/0117e.jc_i000.txt")

# %%
    importlib.reload(psm)
    IDEAL_SAG = psm.IdealSagReading(filepath_ideal_sag="raw_data/calcCirSagDist01.csv")

    pitch = psm.CirclePathPitch(Constants=CONSTS,
                                df_measurement=mes.df_float)

    itg = psm.CirclePathIntegration(Constants=CONSTS,
                                    IdealSagReading=IDEAL_SAG,
                                    df_pitch=pitch.df_pitch,
                                    integration_optimize_init=-5e4)
