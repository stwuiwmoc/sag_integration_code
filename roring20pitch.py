# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:25:23 2021

@author: swimc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def mkfolder(suffix = ""):
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

def oap_z_height(x_from_mirror_center, y_from_mirror_center, 
                 p_curvature_radius=8667, q_off_axis_distance=1800, phi_rad_clocking=0):
    """
    

    Parameters
    ----------
    x_from_mirror_center : TYPE
        DESCRIPTION.
    y_from_mirror_center : TYPE
        DESCRIPTION.
    p_curvature_radius : TYPE, optional
        DESCRIPTION. The default is 8667.
    q_off_axis_distance : TYPE, optional
        DESCRIPTION. The default is 1800.
    phi_rad_clocking : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    cz1 : TYPE
        DESCRIPTION.

    """
    phi = phi_rad_clocking
    aqx = q_off_axis_distance
    cx = x_from_mirror_center
    cy = y_from_mirror_center
    
    a = 1/(2*p_curvature_radius)
    aqz = a * ( aqx**2 + 0**2 )
    theta = np.arctan(2*a*aqx)
    D = -4*a**2*cx**2*np.sin(phi)**2*np.sin(theta)**2 - 8*a**2*cx*cy*np.sin(phi)*np.sin(theta)**2*np.cos(phi) + 4*a**2*cy**2*np.sin(phi)**2*np.sin(theta)**2 - 4*a**2*cy**2*np.sin(theta)**2 + 2*a*aqx*np.sin(2*theta) + 4*a*aqz*np.sin(theta)**2 + 4*a*cx*np.sin(theta)*np.cos(phi) - 4*a*cy*np.sin(phi)*np.sin(theta) - np.sin(theta)**2 + 1

    cz1 = (4*a*aqx*np.sin(theta) - a*cx*(np.sin(phi - 2*theta) - np.sin(phi + 2*theta)) - a*cy*(np.cos(phi - 2*theta) - np.cos(phi + 2*theta)) - 2*np.sqrt(D) + 2*np.cos(theta))/(4*a*np.sin(theta)**2)
    return cz1

class OapFunctions:
    
    def __init__(self, x_from_mirror_center, y_from_mirror_center,
                 p_curvature_radius=8667, q_off_axis_distance=1800, phi_clocking_rad=0):
        self.x = x_from_mirror_center
        self.y = y_from_mirror_center
        self.z = oap_z_height(x_from_mirror_center, y_from_mirror_center, p_curvature_radius, q_off_axis_distance)
        self.phi = phi_clocking_rad
        self.p = p_curvature_radius
        self.q = q_off_axis_distance    
        
        self.a = 1/(2*p_curvature_radius)
        

    def half_off_axis_angle(self, output_radian: bool =True):  
        aqx = self.q
        
        aqz = a * ( aqx**2 + 0**2 )
        half_off_axis_angle_radian = np.arctan(2*a*aqx)
        if output_radian:    
            return half_off_axis_angle_radian
        else:
            half_off_axis_angle_degree = np.rad2deg(half_off_axis_angle_radian)
            return half_off_axis_angle_degree

    def gradient(self):
        aqx = self.q
        
        a = 1/(2*self.p)
        aqz = a * ( aqx**2 + 0**2 )
        theta = np.arctan(2*a*aqx)
        
        dfdx = (-a*(np.sin(phi - 2*theta) - np.sin(phi + 2*theta)) - 2*(-4*a**2*cx*np.sin(phi)**2*np.sin(theta)**2 - 4*a**2*cy*np.sin(phi)*np.sin(theta)**2*np.cos(phi) + 2*a*np.sin(theta)*np.cos(phi))/np.sqrt(-4*a**2*cx**2*np.sin(phi)**2*np.sin(theta)**2 - 8*a**2*cx*cy*np.sin(phi)*np.sin(theta)**2*np.cos(phi) + 4*a**2*cy**2*np.sin(phi)**2*np.sin(theta)**2 - 4*a**2*cy**2*np.sin(theta)**2 + 2*a*aqx*np.sin(2*theta) + 4*a*aqz*np.sin(theta)**2 + 4*a*cx*np.sin(theta)*np.cos(phi) - 4*a*cy*np.sin(phi)*np.sin(theta) - np.sin(theta)**2 + 1))/(4*a*np.sin(theta)**2)
        dfdy = (-a*(np.cos(phi - 2*theta) - np.cos(phi + 2*theta)) - 2*(-4*a**2*cx*np.sin(phi)*np.sin(theta)**2*np.cos(phi) + 4*a**2*cy*np.sin(phi)**2*np.sin(theta)**2 - 4*a**2*cy*np.sin(theta)**2 - 2*a*np.sin(phi)*np.sin(theta))/np.sqrt(-4*a**2*cx**2*np.sin(phi)**2*np.sin(theta)**2 - 8*a**2*cx*cy*np.sin(phi)*np.sin(theta)**2*np.cos(phi) + 4*a**2*cy**2*np.sin(phi)**2*np.sin(theta)**2 - 4*a**2*cy**2*np.sin(theta)**2 + 2*a*aqx*np.sin(2*theta) + 4*a*aqz*np.sin(theta)**2 + 4*a*cx*np.sin(theta)*np.cos(phi) - 4*a*cy*np.sin(phi)*np.sin(theta) - np.sin(theta)**2 + 1))/(4*a*np.sin(theta)**2)
        dfdz = -1
        gradient_vector = np.array([dfdx, dfdy, dfdz])
        return gradient_vector
        """
        """