# %%
import planets_sag_myclass as psm
import numpy as np
import matplotlib.pyplot as plt
import importlib

if __name__ == "__main__":
    importlib.reload(psm)

    CONSTS = psm.Constants(pitch_length=20)  # [mm]
    measurement = psm.MeasurementDataDivide(filepath="raw_data/sag_rawdata/0117/RT_202201171652_FTP.txt")

# %%
    importlib.reload(psm)
    IDEAL_SAG = psm.IdealSagReading(filepath_ideal_sag="raw_data/calcCirSagDist01.csv")

    c = psm.CirclePathIntegration(Constants=CONSTS,
                                  IdealSagReading=IDEAL_SAG,
                                  df_measurement=measurement.raw_df_list[0],
                                  integration_optimize_init=-5e4)
