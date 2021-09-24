# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:08:56 2021

@author: swimc
"""

import time
import csv

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
    read_fname = "raw_data/x-130deg3sag.csv"
    nametxt = read_fname[9:-4] + "_"

    with open(read_fname, encoding="utf-8") as readfile:
        i = 0
        write_fname = nametxt + str(i).zfill(3) + ".csv"
        foldername = mkfolder() + write_fname
        start = time.time()

        reader = csv.reader(readfile)
    
        for row in reader:
            with open(foldername, "a", encoding="utf-8", newline="") as writefile:
                if len(row[0]) != 0:
                    writer = csv.writer(writefile)
                    writer.writerow(row)
                    writefile.close()
    
            if len(row[0]) == 0:
                i = i + 1
                write_fname = nametxt + str(i).zfill(3) + ".csv"
                foldername = mkfolder()+write_fname
                print(time.time() - start)
    
                with open(foldername, "a", encoding="utf-8", newline="") as writefile:
                    writer = csv.writer(writefile)
                    writer.writerow(["X","Y","Z","theta","rho","1/rho","sag"])
                    writefile.close()
