#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import csv

def mkhelp(instance):
    import inspect
    attr_list = list(instance.__dict__.keys())
    for attr in attr_list:
        if attr.startswith("_"): continue
        print(attr)
    for method in inspect.getmembers(instance, inspect.ismethod):
        if method[0].startswith("_"): continue
        print(method[0]+"()")

class Constants:
    def __init__(self, pitch_length, ):
        self.pitch_length=pitch_length
        
    def h(self):
        mkhelp(self)
        