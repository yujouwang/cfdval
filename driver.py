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


from IPython.display import clear_output


# from code.config import *
# import sys
# sys.path.append(r'D:\Dropbox (MIT)\ResearchProjects\2020_CFD\DataProcessing\cfd_1_ManchesterLESVal\code')
# from valcode.datastructure import SimFileDataPath
# from valcode.csvtables import TableProbes
# from valcode.probe_data import probe_locations


D_m = 108E-3 # main pipe diameter (m)
D_b = 21E-3 # Branch pipe diameter (m)
T_h_les = 303
T_c_les = 288
T_c_stc = 36.5 + 273
T_h_stc = 95 + 273


def pre_proc_struct():
    # Initiate 
    structs = TableProbes(
        data_dir = data_dir, 
        save_to = save_to,
        table_name = 'probes_table', 
        probe_locs=probe_locations,
        N_files=3000
        )
    to_dict = structs.parse_all_columns(overwrite=False)
    

def read_les_line_data():
    from valcode.linedata import ManchesterLineData
    data_dir = r'E:\LES_Manchester\UoM_TJ_LES-20211013T183109Z-001\UoM_TJ_LES\lines'
    manchester = ManchesterLineData(data_dir)
    data = manchester.get_data(data_type='T', dir='V', x_loc=2)
    print(data)



def main():
    # Preprocessing 
    # probe data
    struct_data_dir = r'F:\project1_Manchester\CFD_Manchester\1_FullModel\Struct\1p5mm_v2_ManchesterInlet'
    struct_pp_save_to = r'./data/parsed_CFD_tables_v2'
    pre_proc_struct(struct_data_dir, struct_pp_save_to)

    les_data_dir = r'F:\LES_Manchester\UoM_TJ_LES-20211013T183109Z-001\UoM_TJ_LES'






if __name__ == '__main__':
    from valcode.manchester import ReynoldsStress
    data_folder = r'E:\LES_Manchester\UoM_TJ_LES-20211013T183109Z-001\UoM_TJ_LES\lines'
    dir = 'V'
    loc = 1
    mean = ReynoldsStress(data_folder, dir, loc)
    print(mean.uu)