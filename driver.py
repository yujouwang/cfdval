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
from valcode.csvtables import TableProbes
from valcode.probe_data import probe_locations





D_m = 108E-3 # main pipe diameter (m)
D_b = 21E-3 # Branch pipe diameter (m)
T_h_les = 303
T_c_les = 288
T_c_stc = 36.5 + 273
T_h_stc = 95 + 273


def main():
    data_dir = r'F:\project1_Manchester\CFD_Manchester\1_FullModel\Struct\1p5mm_v2_ManchesterInlet\Probes'
    save_to = r'./data/parsed_CFD_tables_v2'

    # Initiate 
    structs = TableProbes(
        data_dir = data_dir, 
        save_to = save_to,
        table_name = 'probes_table', 
        probe_locs=probe_locations,
        N_files=3000
        )

    to_dict = structs.parse_all_columns(overwrite=False)

if __name__ == '__main__':
    main()