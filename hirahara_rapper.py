# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:40:59 2021

@author: swimc
"""

import subprocess

def oap_exe(A=False, B=False, L=False, S=False, N=False, h=False):
    import subprocess
    command = ["OAP.exe", "-x"]
    
    command.append(str(A))
    command.append(str(B))
    
    if L != False:
        command.extend(["-L", str(L)])
    
    if S != False:
        command.extend(["-S", str(S)])
    
    if N != False:
        command.extend(["-N", str(N)])
    
    if h != False: command.append("-h")

    res = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="cp932")

    return res.stdout
    
def oap_out(A, B, L, S, N):
    import pandas as pd
    res = oap_exe(A, B, L, S, N)
    out = res[34:-24]
    return out
    
if __name__ == '__main__':
    print(oap_exe(h=True))
    out = oap_out(0,0,False, False, False)