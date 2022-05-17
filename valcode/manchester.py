"""
The classes and functions for reading Manchester datastructure
"""
from pathlib import Path
import numpy as np
import pandas as pd

from valcode.datastructure import MeanVelocity, MeanTemperature, ReynoldsStress, LineData, HorizontalLine, VerticalLine


class MeanVelocityManchester(MeanVelocity):
    def __init__(self, data_folder: str, dir: str, loc: float):
        self.data_folder = data_folder
        self.dir = dir
        self.loc = loc
        self._coord, self._U, self._V, self._W = self._read_data()
    
    def _read_data(self):
        file_path = self._file_path()
        data = pd.read_csv(file_path)
        coord = data['Points:2'].values if self.dir =='H' else data['Points:1'].values
        U = data['UMean:0'].values
        V = data['UMean:1'].values
        W= data['UMean:2'].values
        return coord, U, V, W

    def _file_path(self):
        loc_in_m = str(round(0.021 * self.loc, 4)).split('.')[1]
        sign = '' if self.loc >=0 else '-'
        assert self.dir in ['H', 'V'], "Check direction"
        dir_in_coord = 'Z' if self.dir == 'H' else 'Y'
        return Path(self.data_folder) / 'csv_archer' / f'2256_profile_x_{sign}{loc_in_m}_{dir_in_coord}.csv'

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


class MeanTemperatureManchester(MeanTemperature):
    def __init__(self, data_folder: str, dir: str, loc: float):
        self.data_folder = data_folder
        self.dir = dir
        self.loc = loc
        self._coord, self._T = self._read_data()

    def _read_data(self):
        file_path = self._file_path()
        data = pd.read_csv(file_path)
        coord: np.ndarray = data['Points:2'].values if self.dir =='H' else data['Points:1'].values
        T = data['TMean'].values
        return coord, T

    def _file_path(self):
        loc_in_m = str(round(0.021 * self.loc, 4)).split('.')[1]
        sign = '' if self.loc >=0 else '-'
        assert self.dir in ['H', 'V'], "Check direction"
        dir_in_coord = 'Z' if self.dir == 'H' else 'Y'
        return Path(self.data_folder) / 'csv_archer' / f'2256_profileT_x_{sign}{loc_in_m}_{dir_in_coord}.csv'

    @property 
    def coord(self):
        return self._coord
    
    @property
    def T(self):
        return self._T


class ReynoldsStressManchester(ReynoldsStress):
    def __init__(self, data_folder: str, dir:str, loc: float):
        self._coord, self._uu, self._vv, self._ww, self._uv, self._uw, self._vw = self._read_data()
    
    def _read_data(self):
        file_path = self._file_path()
        data = pd.read_csv(file_path, header=None, delimiter=' ')
        coord: np.ndarray = data[0].values
        uu: np.ndarray = data[1].values
        vv: np.ndarray = data[2].values
        ww: np.ndarray = data[3].values
        uv: np.ndarray = data[4].values
        uw: np.ndarray = data[5].values
        vw: np.ndarray = data[6].values
        return coord, uu, vv, ww, uv, uw, vw
        
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


class VerticalLineManchester(VerticalLine):
    def __init__(self, data_folder: str, loc: float):
        self.data_folder = data_folder
        self.loc = loc
        self.dir = 'V'
        self._Umean =  MeanVelocityManchester(self.data_folder, self.dir, self.loc)
        self._Uprime = ReynoldsStressManchester(self.data_folder, self.dir, self.loc)
        self._Tmean = MeanTemperatureManchester(self.data_folder, self.dir, self.loc)
    
    @property
    def Umean(self):
        return self._Umean
        
    @property
    def Uprime(self):
        return self._Uprime

    @property
    def Tmean(self):
        return self._Tmean


class HorizontalLineManchester(HorizontalLine):
    def __init__(self, data_folder: str, loc: float):
        self.data_folder = data_folder
        self.loc = loc
        self.dir = 'H'
        self._Umean =  MeanVelocityManchester(self.data_folder, self.dir, self.loc)
    

class ManchesterLineData(LineData):
    def __init__(self, data_folder: str, loc: float):
        self._vertical = VerticalLineManchester(data_folder, loc)
        self._horizontal = HorizontalLineManchester(data_folder, loc)
    
    @property
    def vertical(self):
        return self._vertical
    
    @property
    def horizontal(self):
        return self._horizontal