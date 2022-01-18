# %%
import planets_sag_myclass as psm
import numpy as np
import matplotlib.pyplot as plt
import importlib

if __name__ == "__main__":
    importlib.reload(psm)

    CONSTS = psm.Constants(pitch_length=20)  # [mm]
    measurement = psm.MeasurementDataDivide(filepath="raw_data/1216/RT_202112161123_FTP.txt")

# %%
    importlib.reload(psm)
    c = psm.CirclePathIntegration(CONSTS, measurement.raw_df_list[0])
