#%%
import planets_sag_myclass as psm
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    CONSTS = psm.Constants(pitch_length=20e-3)
    measurement = psm.MeasurementDataDivide(filepath="raw_data/1216/RT_202112161123_FTP.txt")
    
