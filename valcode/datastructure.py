""" 
This script describe how data is stored
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path


class MeanVelocity(ABC):
    @property
    @abstractmethod
    def coord(self):
        pass

    @property
    @abstractmethod
    def U(self):
        pass

    @property
    @abstractmethod
    def V(self):
        pass

    @property
    @abstractmethod
    def W(self):
        pass


class MeanTemperature(ABC):
    @property
    @abstractmethod
    def coord(self):
        pass
    @property
    @abstractmethod
    def T(self):
        pass


class ReynoldsStress(ABC):
    @property
    @abstractmethod
    def coord(self):
        pass

    @property
    @abstractmethod
    def tke(self):
        pass


    
class HorizontalLine(ABC):
    @property
    @abstractmethod
    def Umean(self):
        pass

class VerticalLine(ABC):
    @property
    @abstractmethod
    def Umean(self):
        pass

    @property
    @abstractmethod
    def tke(self):
        pass

    @property
    @abstractmethod
    def Tmean(self):
        pass

class LineData(ABC):
    @property
    @abstractmethod
    def vertical(self):
        pass
    @property
    @abstractmethod
    def horizontal(self):
        pass


    

