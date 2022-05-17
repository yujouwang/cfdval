from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

from valcode.manchester import ManchesterHorizontalLine, ManchesterVerticalLine

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
    def uu(self):
        pass

    @property
    @abstractmethod
    def vv(self):
        pass

    @property
    @abstractmethod
    def ww(self):
        pass

    @property
    @abstractmethod
    def uv(self):
        pass

    @property
    @abstractmethod
    def uw(self):
        pass

    @property
    @abstractmethod
    def vw(self):
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
    def Uprime(self):
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


    

