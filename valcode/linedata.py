from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


from valcode.manchester import ManchesterLineData
from valcode.mydata import MyLineData


class LineDataReader(ABC):
    """ The abstract class of line data """
    @abstractmethod
    def get_data(self, data_type: str, dir: str, x_loc:float) -> np.ndarray:
        raise NotImplementedError


class ManchesterLineDataReader(LineDataReader):
    """ The data structure of the Manchester"""
    def __init__(self, data_dir: str):
        """
        INPUT:
            data_dir: str, the main folder which contains all the line data
        """
        # Data Directory
        self.data_dir = data_dir

    def get_vertical_data(self, loc: float):
        assert loc in [-7, 1, 2, -0.5, -2, -4.5], 'The location should be in [-7, 1, 2, -0.5, -2, -4.5]'
        data = ManchesterLineData(self.data_folder, loc)
        vert_data = data.vertical
        return vert_data

    def get_horizontal_data(self, loc: float):
        assert loc in [-7, 1, 2, -0.5, -2, -4.5], 'The location should be in [-7, 1, 2, -0.5, -2, -4.5]'
        data = ManchesterLineData(self.data_folder, loc)
        vert_data = data.vertical
        return vert_data
    
        

class MyLineDataReader(LineDataReader):
    """ My data structure"""
    def __init__(self, data_dir: str):
        """
        INPUT:
            data_dir: str, the main folder which contains all the line data
        """
        # Data Directory
        self.data_dir = data_dir

    def get_vertical_data(self, loc: float):
        assert loc in [-7, 1, 2, -0.5, -2, -4.5], 'The location should be in [-7, 1, 2, -0.5, -2, -4.5]'
        data = MyLineData(self.data_folder, loc)
        vert_data = data.vertical
        return vert_data

    def get_horizontal_data(self, loc: float):
        assert loc in [-7, 1, 2, -0.5, -2, -4.5], 'The location should be in [-7, 1, 2, -0.5, -2, -4.5]'
        data = MyLineData(self.data_folder, loc)
        vert_data = data.vertical
        return vert_data
    
