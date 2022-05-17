from enum import Enum, auto
from re import A

from valcode.linedata import LineDataReader

X_LOCS = [-7, -4.5, -2, 1, 2]

class DataType(Enum):
    T = auto()
    U = auto()
    V = auto()
    W = auto()
    TKE = auto()

manchester_to_csv_mapper = {
    DataType.T: "Temperature (K)",
    DataType.U: "Velocity[i] (m/s)",
    DataType.V: "Velocity[j] (m/s)",
    DataType.W: "Velocity[k] (m/s)",
    DataType.TKE: "TKE_Total"
}


def plot_vertical_line(data_type: Enum, xlim=None, scaling_fac=8, shift=288, title=''):
    column = manchester_to_csv_mapper[data_type]



class PlotLineData:
    def __init__(self, *line_objects: LineDataReader) -> None:
        self.line_objects = line_objects
        
    def plot(self):
        for line_obj in self.line_objects:
            # plot 
            data = line_obj.get_data()
            plot(data)