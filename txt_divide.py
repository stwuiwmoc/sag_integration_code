# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 18:14:25 2021

@author: swimc
"""
import time

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

if __name__ == "__main__":
    
    read_fname = "0922xm130_3deg.txt"
    nametxt = read_fname[:-4] + "_"
    
    with open("raw_data/"+read_fname, "r") as readfile:
        i = 0
        write_fname = nametxt + str(i).zfill(3) + ".txt"
        foldername = mkfolder()+write_fname
        start = time.time()
    
        for data in readfile:
            with open(foldername, "a") as writefile:
                writefile.write(data)
                writefile.close()
            
            if data=="\n":
                print(str(i))
                i = i+1
                write_fname = nametxt + str(i).zfill(3) + ".txt"
                foldername = mkfolder()+write_fname
                print(time.time() - start)
                
        readfile.close()
