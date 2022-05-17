
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
# Customize these functions to read from disk
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

def read_struct_Umean(data_folder, dir, loc):
    df = pd.read_csv(data_folder / 'mean.csv')
    x = df['X (m)']
    loc_in_m = 0.021 * loc
    data = extract_line(df, loc_in_m, dir)
    
    if dir == 'H':
        coord = data['Y (m)'].values
    else:
        coord = data['Z (m)'].values

    U = data['Velocity[i] (m/s)']
    V = data['Velocity[j] (m/s)']
    W = data['Velocity[k] (m/s)']
    return coord, U, V, W

def read_cubic_Umean(data_folder, dir, loc):
    # Specify direction 
    direction_name = 'horizontal' if dir == 'H' else 'vertical'
    data = pd.read_csv(Path(data_folder) / f'SS_Velocity_{direction_name}.csv')

    # Read 
    probe_name = f'line_{loc}Db_H: Velocity[i] (m/s)' if dir=='H' else f'line_{loc}Db_V: Velocity[i] (m/s)'
    coord_name = f'line_{loc}Db_H: Position[Y] (m)' if dir=='H' else f'line_{loc}Db_V: Position[Z] (m)'

    coord = data[coord_name].values
    U = data[probe_name].values
    V = np.ones(len(U)) * np.nan 
    W = np.ones(len(U)) * np.nan 

    return coord, U, V, W





# =================================
class MyMeanVelocity(MeanVelocity):
    def __init__(self, data_folder: str, dir: str, loc: float, f_read_Umean: ReadDataFunction):
        self.data_folder = data_folder
        self.dir = dir
        self.loc = loc
        self._coord, self._U, self._V, self._W = f_read_Umean(data_folder, dir, loc)

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
    def __init__(self, data_folder: str, dir: str, loc: float, f_read_Tmean: ReadDataFunction):
        self.data_folder = data_folder
        self.dir = dir
        self.loc = loc
        self._coord, self._T = f_read_Tmean(data_folder, dir, loc)

    @property 
    def coord(self):
        return self._coord
    
    @property
    def T(self):
        return self._T


class MyReynoldsStress(ReynoldsStress):
    def __init__(self, data_folder: str, dir:str, loc: float, f_read_reynoldStress: ReadDataFunction):
        self.data_folder = data_folder
        self.dir = dir
        self.loc = loc
        self._coord, self._uu, self._vv, self._ww, self._uv, self._uw, self._vw = f_read_reynoldStress(data_folder, dir, loc)
        
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
    def uu(self):
        return self._uu

    @property
    def vv(self):
        return self._vv

    @property
    def ww(self):
        return self._ww

    @property
    def uv(self):
        return self._uv

    @property
    def uw(self):
        return self._uw

    @property
    def vw(self):
        return self._vw


class MyVerticalLine(VerticalLine):
    def __init__(
        self, 
        data_folder: str, 
        loc: float, 
        f_read_Umean: ReadDataFunction, 
        f_read_Uprime: ReadDataFunction,
        f_read_Tmean: ReadDataFunction
        ):
        self.data_folder = data_folder
        self.loc = loc
        self.dir = 'V'
        self._Umean =  MyMeanVelocity(self.data_folder, self.dir, self.loc, f_read_Umean)
        self._Uprime = MyReynoldsStress(self.data_folder, self.dir, self.loc, f_read_Uprime)
        self._Tmean = MyMeanTemperature(self.data_folder, self.dir, self.loc, f_read_Tmean)
    
    @property
    def Umean(self):
        return self._Umean
        
    @property
    def Uprime(self):
        return self._Uprime

    @property
    def Tmean(self):
        return self._Tmean


class MyHorizontalLine(HorizontalLine):
    def __init__(self, data_folder: str, loc: float, f_read_Umean: ReadDataFunction):
        self.data_folder = data_folder
        self.loc = loc
        self.dir = 'H'
        self._Umean =  MyMeanVelocity(self.data_folder, self.dir, self.loc, f_read_Umean)
    

class MyLineData(LineData):
    def __init__(self, data_folder: str, loc: float, f_read_Umean: ReadDataFunction, f_read_Uprime: ReadDataFunction, f_read_Tmean: ReadDataFunction):
        self._vertical = MyVerticalLine(data_folder, loc, f_read_Umean, f_read_Uprime, f_read_Tmean)
        self._horizontal = MyHorizontalLine(data_folder, loc, f_read_Umean)
    
    @property
    def vertical(self):
        return self._vertical
    
    @property
    def horizontal(self):
        return self._horizontal