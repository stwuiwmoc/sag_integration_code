# -*- coding: utf-8 -*-
""" 
Created on Fri Sep 24 21:06:56 2021
@author: swimc
""" 

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time

import single_path as sgp

def pythagoras(x, args):
    fz, y_init = args
    vert = sgp.poly_calc(fz, y_init+x) - sgp.poly_calc(fz, y_init)
    hori = x
    return vert**2 + hori**2 - 400

if __name__ == '__main__':
    y_c_raw, sag_c_raw, z_c_raw = sgp.read_raw_calc("raw_data/sample.csv", full=True)
    fz = np.polynomial.polynomial.polyfit(y_c_raw, z_c_raw, 4)
    z_fit = sgp.poly_calc(fz, y_c_raw)
    y0 = -750
    
    y_samp_s = [y0]
    
    root_guess = 19
    
    args = fz, y_samp_s
    start = time.time()
    
    for i in range(100):
        sol = sp.optimize.newton(pythagoras, root_guess, args=((fz,y_samp_s[i] ),))
        y0 = y0 + sol
        y_samp_s.append(y0)
        if y0 >= 730:
            break
    
    print(time.time() - start)
    