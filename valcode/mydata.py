
"""
The classes and functions for reading my CFD (Star-ccm+) datastructure
My data: 
    star-ccm+ outputs are a bunch of snapshots; should be parsed first

"""
from pathlib import Path
from grpc import Call
import numpy as np
import pandas as pd
from typing import Callable

from valcode.datastructure import MeanVelocity, MeanTemperature, ReynoldsStress, LineData, HorizontalLine, VerticalLine


ReadDataFunction =  Callable


# =================================

T_c_stc = 36.5 + 273
T_h_stc = 95 + 273

# Customize these functions to read from disk
def extract_line(df: pd.DataFrame, direction: str, loc: float):
    if direction == 'V':
        output = df[(abs(df['X (m)'] - loc) < 1E-5) & (abs(df['Y (m)'])==0)]
        # assert len(output) <= 200, f"debug: dir: {direction}, loc:{loc}"
        assert len(output) <= 205, f"{output}"
    elif direction =='H':
        output = df[(abs(df['X (m)'] - loc) <1E-5) & (abs(df['Z (m)'])==0)]
        assert len(output) <= 205, f"debug: dir: {direction}, loc:{loc}"
    else:
        raise ValueError(f"Check direction: {direction}, should be 'V' or 'H' ")
    return output

def load_my_mean_file(data_folder: str, direction: str, loc: float):
    df = pd.read_csv(Path(data_folder) / 'mean.csv')
    x = df['X (m)']
    loc_in_m = 0.021 * loc
    data = extract_line(df, direction, loc_in_m)
    assert len(data) >=0, "No line is found"
    return data

def read_struct_Umean(data_folder, direction, loc):
    data = load_my_mean_file(data_folder, direction, loc)

    if direction == 'H':
        coord = data['Y (m)'].values
    else:
        coord = data['Z (m)'].values

    U = data['Velocity[i] (m/s)'].values
    V = data['Velocity[j] (m/s)'].values
    W = data['Velocity[k] (m/s)'].values
    return coord, U, V, W

def read_cubic_Umean(data_folder, direction, loc):
    data = load_my_mean_file(data_folder, direction, loc)
    if direction == 'H':
        coord = data['Y (m)'].values
    else:
        coord = data['Z (m)'].values

    U = data['Mean of Velocity[i] (m/s)'].values
    V = data['Mean of Velocity[j] (m/s)'].values
    W = data['Mean of Velocity[k] (m/s)'].values
    return coord, U, V, W


def read_struct_tke(data_folder: str, direction: str, loc: float):
    data = load_my_mean_file(data_folder, direction, loc)
    if direction == 'H':
        coord = data['Y (m)'].values
    else:
        coord = data['Z (m)'].values
    tke = data['TKE_Total'].values
    return coord, tke

def read_cubic_tke(data_folder, direction, loc):
    data = load_my_mean_file(data_folder, direction, loc)
    if direction == 'H':
        coord = data['Y (m)'].values
    else:
        coord = data['Z (m)'].values

    tke = data['Mean of Turbulent Kinetic Energy (J/kg)'].values
    return coord, tke

def read_struct_Tmean(data_folder: str, direction:str, loc: float):
    data = load_my_mean_file(data_folder, direction, loc)
    if direction == 'H':
        coord = data['Y (m)'].values
    else:
        coord = data['Z (m)'].values
    temperature = data['Temperature (K)'].values 
    temperature = (temperature - T_c_stc) / (T_h_stc - T_c_stc)
    return coord, temperature

def read_cubic_Tmean(data_folder, direction, loc):
    data = load_my_mean_file(data_folder, direction, loc)
    if direction == 'H':
        coord = data['Y (m)'].values
    else:
        coord = data['Z (m)'].values

    temperature = data['Mean of Temperature (K)'].values
    temperature = (temperature - T_c_stc) / (T_h_stc - T_c_stc)
    return coord, temperature

# =================================
class MyMeanVelocity(MeanVelocity):
    def __init__(self, data_folder: str, direction: str, loc: float, f_read_Umean: ReadDataFunction):
        self.data_folder = data_folder
        self.direction = direction
        self.loc = loc
        self._coord, self._U, self._V, self._W = f_read_Umean(data_folder, direction, loc)

    @property
    def coord(self):
        return self._coord

    @property
    def U(self):
        return self._U

    @property
    def V(self):
        return self._V

    @property
    def W(self):
        return self._W


class MyMeanTemperature(MeanTemperature):
    def __init__(self, data_folder: str, direction: str, loc: float, f_read_Tmean: ReadDataFunction):
        self.data_folder = data_folder
        self.direction = direction
        self.loc = loc
        self._coord, self._T = f_read_Tmean(data_folder, direction, loc)

    @property 
    def coord(self):
        return self._coord
    
    @property
    def T(self):
        return self._T -273.15


class MyReynoldsStress(ReynoldsStress):
    def __init__(self, data_folder: str, direction:str, loc: float, f_read_tke: ReadDataFunction):
        self.data_folder = data_folder
        self.direction = direction
        self.loc = loc
        self._coord, self._tke = f_read_tke(data_folder, direction, loc)
        
    def _file_path(self):
        loc_map = {
            -7: 'line1_y',
            1: 'line2_y',
            2: 'line3_y',
            -0.5: 'line4_y',
            -2: 'line5_y',
            -4.5:'line6_y'
        }
        return Path(self.data_folder) / loc_map[self.loc] /'22.56'/ 'line_UPrime2Mean.xy'

    @property
    def coord(self):
        return self._coord

    @property
    def tke(self):
        return self._tke



class MyVerticalLine(VerticalLine):
    def __init__(
        self, 
        data_folder: str, 
        loc: float, 
        f_read_Umean: ReadDataFunction, 
        f_read_tke: ReadDataFunction,
        f_read_Tmean: ReadDataFunction
        ):
        self.data_folder = data_folder
        self.loc = loc
        self.direction = 'V'
        self._Umean =  MyMeanVelocity(self.data_folder, self.direction, self.loc, f_read_Umean)
        self._tke = MyReynoldsStress(self.data_folder, self.direction, self.loc, f_read_tke)
        self._Tmean = MyMeanTemperature(self.data_folder, self.direction, self.loc, f_read_Tmean)
    
    @property
    def Umean(self):
        return self._Umean
        
    @property
    def tke(self):
        return self._tke

    @property
    def Tmean(self):
        return self._Tmean


class MyHorizontalLine(HorizontalLine):
    def __init__(self, data_folder: str, loc: float, f_read_Umean: ReadDataFunction):
        self.data_folder = data_folder
        self.loc = loc
        self.direction = 'H'
        self._Umean =  MyMeanVelocity(self.data_folder, self.direction, self.loc, f_read_Umean)
    
    @property
    def Umean(self):
        return self._Umean
    

class MyLineData(LineData):
    def __init__(
        self, 
        data_folder: str, 
        loc: float, 
        f_read_Umean: ReadDataFunction, 
        f_read_tke: ReadDataFunction, 
        f_read_Tmean: ReadDataFunction
        ):
        self._vertical = MyVerticalLine(data_folder, loc, f_read_Umean, f_read_tke, f_read_Tmean)
        self._horizontal = MyHorizontalLine(data_folder, loc, f_read_Umean)
    
    @property
    def vertical(self):
        return self._vertical
    
    @property
    def horizontal(self):
        return self._horizontal