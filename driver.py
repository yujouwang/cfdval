import os
from pathlib import Path
import glob
import re
import pandas as pd 
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set_theme()
from importlib import reload  


from valcode.manchester import ManchesterLineData
from valcode.mydata import *

D_m = 108E-3 # main pipe diameter (m)
D_b = 21E-3 # Branch pipe diameter (m)
T_h_les = 303
T_c_les = 288
T_c_stc = 36.5 + 273
T_h_stc = 95 + 273


def line():
    x_locs = [-7, -4.5, -2, 1, 2, 4.5]
    les_line_folder = r'E:\LES_Manchester\UoM_TJ_LES-20211013T183109Z-001\UoM_TJ_LES\lines'
    struct_line_folder = r'E:\project1_Manchester\CFD_Manchester\1_FullModel\Struct\1p5mm_v2_ManchesterInlet\Lines'
    cubic_folder = r'E:\project1_Manchester\CFD_Manchester\1_FullModel\Cubic'
    loc = x_locs[0]

    cubic = MyLineData(
        cubic_folder, 
        loc, 
        f_read_Umean=read_cubic_Umean, 
        f_read_tke=read_cubic_tke, 
        f_read_Tmean=read_cubic_Tmean)

    coord = cubic.vertical.Umean.coord
    Umean = cubic.vertical.Umean.U
    print(coord)



if __name__ == '__main__':
    line()