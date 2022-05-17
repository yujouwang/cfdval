from pathlib import Path
import glob
import re
import pandas as pd 
import numpy as np


def extract_line(df: pd.DataFrame, loc: float, dir: str):
    if dir == 'V':
        output = df[(df['X (m)'] == loc) & (abs(df['Y (m)'])==0)]
        assert len(output) <= 200
    elif dir =='H':
        output = df[(df['X (m)'] == loc) & (abs(df['Z (m)'])==0)]
        assert len(output) <= 200
    else:
        raise ValueError(f"Check dir: {dir}, should be 'V' or 'H' ")
    return output


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main():
    # find all files
    folder = Path(r'E:\project1_Manchester\CFD_Manchester\1_FullModel\Struct\1p5mm_v2_ManchesterInlet\Lines') 
    files = natural_sort(glob.glob1(folder, '*.csv'))
    columns = ['Temperature (K)', 'Turbulent Kinetic Energy (J/kg)', 'TKE_Resolved',
        'TKE_Total', 'Velocity[i] (m/s)', 'Velocity[j] (m/s)',
        'Velocity[k] (m/s)', 'X (m)', 'Y (m)', 'Z (m)']

    # Get mean
    df = pd.read_csv(folder / files[-1000])[columns]
    for file in files[-999:]:
        df_2 = pd.read_csv(folder/file)[columns]
        df = df + df_2
    df_mean = df/1000
    df_mean.to_csv(folder/'mean.csv', index=None)

    # Get std
    df = pd.read_csv(folder / files[-1000])[columns]
    df = (df - df_mean)**2
    for file in files[-999:]:
        df_2 = pd.read_csv(folder/file)[columns]
        df = df + (df_2 - df_mean)**2
    df_std = np.sqrt(df/1000)
    df_std.to_csv(folder/'std.csv', index=None)
    


    # Get data 
    # For every file in files
    # Parse the csv into line loc dict: key=loc, value: pd.DataFrame
    # for every loc in locs
    # Iterat
    
