"""
This module contains the comparison
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os 
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from manchester import *
from scipy.fft import fft, fftfreq
sns.set_theme()
sns.set_style("whitegrid")


from probe_data import probe_locations
from csvtables import TableProbes, TableLines


COLOR_MAP = ['orangered', 'seagreen', 'royalblue', 'darkviolet', 'darkgoldenrod', 'cyan', 'blue']
# COLOR_MAP = sns.color_palette()
D_b = 21E-3 

def plot_along_x(les_data, stc_data, function, probe_locations, title='', ylabel=''):
    
    assert len(les_data)==len(stc_data) and len(les_data)==len(probe_locations), "The lenths of inputs are not consistent"
    x_unique = np.unique(np.array(probe_locations)[:,0])
    
    for i_p, probe in enumerate(probe_locations):
        x, y, z = probe
        loc_index = np.where(x == x_unique)[0][0]

        difference = function(stc_data[i_p], les_data[i_p])
        plt.scatter(float(x), difference, color=COLOR_MAP[loc_index])
    # Plot settings
    plt.xticks([-0.216, 0, 0.021, 0.042, 0.063], ['-7Db', '0', '1Db','2Db','3Db'])
    plt.plot([-0.22, 0.1], [0,0], linestyle='--', color='k')
    plt.xlabel("X (Db)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return

def plot_along_angle(les_data, stc_data, function, probe_locations, title='', ylabel=''):
    COLOR_MAP = sns.color_palette("Dark2")
    assert len(les_data)==len(stc_data) and len(les_data)==len(probe_locations), "The lenths of inputs are not consistent"
    angles = [abs(np.arctan(float(probe[2])/ float(probe[1])) *180/np.pi ) for probe in probe_locations[1:]] 
    _, edges = np.histogram(angles, bins=5)
    
    for i_p, probe in enumerate(probe_locations):
        x, y, z = probe       
        angle = np.arctan(float(probe[2])/ float(probe[1])) *180/np.pi
        angle_index = find_group(edges, abs(angle))        
        difference = function(stc_data[i_p], les_data[i_p])
        plt.scatter(angle, difference, color=COLOR_MAP[angle_index])
        
    # Plot settings
#     plt.xticks([-0.216, 0, 0.021, 0.042, 0.063], ['-7Db', '0', '1Db','2Db','3Db'])
    plt.plot([-30, 30], [0,0], linestyle='--', color='k')
    plt.xlabel("Angle (deg)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return


def plot_xy_plot(les_data, stc_data, probe_locations, xlim=None, title=''):
    assert len(les_data)==len(stc_data) and len(les_data)==len(probe_locations), "The lenths of inputs are not consistent"
    x_unique = np.unique(np.array(probe_locations)[:,0])
    ax = plt.subplot(111)
    
    for i_p, probe in enumerate(probe_locations):
        x, y, z = probe
        loc_index = np.where(x == x_unique)[0][0]
        ax.scatter(les_data[i_p], stc_data[i_p], color=COLOR_MAP[loc_index])
    if xlim:
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
       
    # Plot setting 
    ax.plot([0,320], [0,320], 'k--')
    ax.set_xlabel("LES data")
    ax.set_ylabel("STRUCT data")
    plt.title(title)
    legend_elements = [Line2D([0],[0], marker='o',color=COLOR_MAP[i], label='%s Db'%round(float(x)/D_b, 1)) 
                       for i, x in enumerate(x_unique)]
    plt.legend(handles=legend_elements, title='x location:')
    plt.show()
    return



def find_group(edges, value):
    for i in range(len(edges)-1):
        if value>=edges[i] and value<=edges[i+1]:
            return i
    return -1

def plot_xy_plot_angle(les_data, stc_data, probe_locations, xlim=None, title=''):
    assert len(les_data)==len(stc_data) and len(les_data)==len(probe_locations), "The lenths of inputs are not consistent"
    angle = [abs(np.arctan(float(probe[2])/ float(probe[1])) *180/np.pi ) for probe in probe_locations[1:]] 
    _, edges = np.histogram(angle, bins=5)
    ax = plt.subplot(111)
    
    for i_p, probe in enumerate(probe_locations):
        x, y, z = probe        
        angle_index = find_group(edges, abs(np.arctan(float(probe[2])/ float(probe[1])) *180/np.pi))
        ax.scatter(les_data[i_p], stc_data[i_p], color=COLOR_MAP[angle_index])
    if xlim:
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
       
    # Plot setting 
    ax.plot([0,320], [0,320], 'k--')
    ax.set_xlabel("LES data")
    ax.set_ylabel("STRUCT data")
    plt.title(title)
    legend_elements = [Line2D([0],[0], marker='o',color=COLOR_MAP[i], label='+- % deg'%int(float(x))) 
                       for i, x in enumerate(edges)]
    plt.legend(handles=legend_elements, title='x location:')
    plt.show()
    return


    


class InstDataPlot:
    def __init__(self, probe, column):
        self.probe = probe
        self.column = column
        self._init_les_data()
        self._init_stc_data()
        
    def _init_les_data(self):
        if self.column == 'Temperature (K)':
            data_type = 'T' 
        if self.column == 'Velocity[i] (m/s)':
            data_type = 'U' 
        if self.column == 'Velocity[j] (m/s)':
            data_type = 'U' 
        if self.column == 'Velocity[k] (m/s)':
            data_type = 'U' 
            
        root = r'F:\LES_Manchester\UoM_TJ_LES-20211013T183109Z-001\UoM_TJ_LES'
        working_dir = os.path.join(root, 'probes')
        save_to = r'D:\Dropbox (MIT)\ResearchProjects\2020_CFD\DataProcessing\cfd_1_ManchesterLESVal/data'
        les = ManchesterProbes(working_dir, save_to, self.probe)
        time_les, data_les = les.get_probe_data(self.probe, data_type)
        logging.debug("Success in getting the les data")
        
        if self.column == 'Velocity[i] (m/s)':
            data_les = data_les[:, 0]
        elif self.column == 'Velocity[j] (m/s)':
            data_les = data_les[:, 1]
        elif self.column == 'Velocity[k] (m/s)':
            data_les = data_les[:, 2]

        self.time_les = time_les - 10
        self.data_les = data_les.flatten()
        self.dt_les = time_les[1] - time_les[0]
        return 
    
    def _init_stc_data(self):
        self.dt_stc = 0.00375
        data_dir = r'F:\project1_Manchester\CFD_Manchester\1_FullModel\Struct\1p5mm_v2_ManchesterInlet\Probes'
        table_name = 'probes_table'
        save_to = r'D:\Dropbox (MIT)\ResearchProjects\2020_CFD\DataProcessing\cfd_1_ManchesterLESVal/data/parsed_CFD_tables_v2'

        # Initiate 
        structs = TableProbes(
            data_dir = data_dir, 
            save_to = save_to,
            table_name =table_name, 
            probe_locs=probe_locations,
            N_files=3000
            )
        data_stc = structs.read_file(column=self.column, probe=self.probe)
        time_stc =self.dt_stc * np.arange(len(data_stc))
        
        self.data_stc = data_stc
        self.time_stc = time_stc
        return
    
    @staticmethod
    def _get_psd(dt_s, x):
        """
        Get fft of the samples
        :param dt_s: time step
        :param x: samples
        :return:
        """
        # 2-side FFT
        N = len(x)
        # xdft = scipy.fft.fft(x)
        # freq = scipy.fft.fftfreq(N, dt_s)
        xdft = fft(x)
        freq =fftfreq(N, dt_s)

        # convert 2-side to 1-side
        if N % 2 == 0:
            xdft_oneside = xdft[0:int(N / 2 )]
            freq_oneside = freq[0:int(N / 2 )]
        else:
            xdft_oneside = xdft[0:int((N - 1) / 2)+1]
            freq_oneside = freq[0:int((N - 1) / 2)+1]


        # Power spectrum
        Fs = 1 / dt_s
        psdx = 1 / (Fs * N) * abs(xdft_oneside)**2
        psdx[1:-1] = 2 * psdx[1:-1] # power for one-side
        return freq_oneside, psdx
    
    def fft(self, time_span=5):
        self.freq_les, self.psdx_les = self._get_psd(self.dt_les, self.data_les[:int(time_span/self.dt_les)])
        self.freq_stc, self.psdx_stc = self._get_psd(self.dt_stc, self.data_stc[:int(time_span/self.dt_stc)])
    
    def plot(self, probe_shortname, ylabel='', xlim=(0,5)):
        plot = plt.figure(figsize=(8,3))
        x, y ,z= self.probe
        probe_loc = int( float(x)/D_b)
        plt.suptitle("Probe loc: %s, %s Db"%(probe_shortname, probe_loc))
        
        # Inst data
        ax1 = plot.add_subplot(121)    
        ax1.plot(self.time_les, self.data_les, label='LES')
        ax1.plot(self.time_stc, self.data_stc, label='Struct')
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel('Time (s)')
        ax1.set_title('')
        ax1.set_xlim(xlim)
        ax1.legend()
        
        # Freq data
        ax2 = plot.add_subplot(122)
        ax2.plot(self.freq_les, self.psdx_les, label='LES')
        ax2.plot(self.freq_stc, self.psdx_stc, label='STRUCT')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_title("PSD")
        ax2.legend()
        return


def plot_vertical(x_loc, data_type, xlim=None):
    if data_type == 'T':
        column = "Temperature (K)"
    elif data_type == 'U':
        column = "Velocity[i] (m/s)"
    elif data_type == 'V':
        column = "Velocity[j] (m/s)"
    elif data_type == 'W':
        column = "Velocity[k] (m/s)"

    # LES data
    l_les = ManchesterLine(x_loc, 'V')
    mean_les = l_les.get_line(data_type)

    # Struct data
    working_dir = r'F:\CFD_Manchester\1_FullModel\Struct\1p5mm_v2_ManchesterInlet\Lines'
    table_name = 'Lines_table'
    l_stc = TableLines(working_dir, table_name)
    mean_stc, std = l_stc.get_lineprobe(column=column, x_loc=x_loc, direction='V')
    Z, _ = l_stc.get_lineprobe(column="Z (m)", x_loc=x_loc, direction='V')
    if data_type == 'T':
        mean_stc = mean_stc + 0.15

        # Plot
    f, ax = plt.subplots(1, 1, figsize=(4, 6))
    ax.plot(mean_les[column], mean_les['y'], 'r-', label='LES')
    ax.plot(mean_stc, Z, 'bo', label='Struct')

    ax.set_xlabel(column)
    ax.set_ylabel('Z (m)')
    ax.legend()
    plt.title('loc: %s Db' % x_loc)
    if xlim:
        ax.set_xlim(xlim)
    plt.show()


def plot_horizontal(x_loc, data_type, xlim=None):
    if data_type == 'T':
        column = "Temperature (K)"
    elif data_type == 'U':
        column = "Velocity[i] (m/s)"
    elif data_type == 'V':
        column = "Velocity[j] (m/s)"
    elif data_type == 'W':
        column = "Velocity[k] (m/s)"

    # LES data
    l_les = ManchesterLine(x_loc, 'H')
    mean_les = l_les.get_line(data_type)

    # Struct data
    working_dir = r'F:\CFD_Manchester\1_FullModel\Struct\1p5mm_v2_ManchesterInlet\Lines'
    table_name = 'Lines_table'
    l_stc = TableLines(working_dir, table_name)
    mean_stc, std = l_stc.get_lineprobe(column=column, x_loc=x_loc, direction='H')
    Z, _ = l_stc.get_lineprobe(column="Y (m)", x_loc=x_loc, direction='H')
    if data_type == 'T':
        mean_stc = mean_stc + 0.15

        # Plot
    f, ax = plt.subplots(1, 1, figsize=(4, 6))
    ax.plot(mean_les[column], mean_les['z'], 'r-', label='LES')
    ax.plot(mean_stc, Z, 'bo', label='Struct')

    ax.set_xlabel(column)
    ax.set_ylabel('Z (m)')
    ax.legend()
    plt.title('loc: %s Db' % x_loc)
    if xlim:
        ax.set_xlim(xlim)
    plt.show()

manchester_to_csv_mapper = {
    "T": "Temperature (K)",
    "U": "Velocity[i] (m/s)",
    "V": "Velocity[j] (m/s)",
    "W": "Velocity[k] (m/s)",
    "TKE": "TKE_Total"
}
from manchester import get_line_data

def plot_vertical_summary(data_type, xlim=None, scaling_fac=8, shift=288, title=''):
    logging.debug("Summary starts")
    sns.set_style("whitegrid")
    assert data_type in ["T", "U", "V", "W", "TKE"], "data_type not in the list"

    # Map the csv column with the Manchester
    column = manchester_to_csv_mapper[data_type]

    x_locs = [-7, -4.5, -2, 1, 2]
    f, ax = plt.subplots(1, 1, figsize=(9, 3))

    for x_loc in x_locs :
        # LES data
        Z_les, mean_les = get_line_data(x_loc, direction='V', data_type=data_type)

        # Struct data
        table_name = 'Lines_table'
        data_dir = r'F:\project1_Manchester\CFD_Manchester\1_FullModel\Struct\1p5mm_v2_ManchesterInlet\Lines'
        save_to = r'D:\Dropbox (MIT)\ResearchProjects\2020_CFD\DataProcessing\cfd_1_ManchesterLESVal/data'
        l_stc = TableLines(data_dir, save_to, table_name)
        mean_stc, std = l_stc.get_lineprobe(column=column, x_loc=x_loc, direction='V')
        Z, _ = l_stc.get_lineprobe(column="Z (m)", x_loc=x_loc, direction='V')

        # A bit modification
        if data_type == 'T':
            mean_stc = mean_stc + 0.15

        # Plot
        ax.plot(mean_les + x_loc * scaling_fac, Z_les, 'r-', label='LES')
        ax.plot(mean_stc + x_loc * scaling_fac, Z, 'bo', label='Struct', markersize=2)

    x_locs = [-7, -4.5, -2, 0, 1, 2, 4.5]
    ax.set_xticks(np.array(x_locs) * scaling_fac + shift)
    ax.set_xticklabels(['%sDb' % x for x in x_locs], fontsize=12)

    ax.set_xlabel(" Location")
    ax.set_ylabel('Z (m)')

    legend_elements = [Line2D([0], [0], color='r', linestyle='-', label="LES"),
                       Line2D([0], [0], color='b', marker='o', label="STRUCT"), ]

    if xlim:
        ax.set_xlim(xlim)
    ax.set_title(title, fontsize=16)
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.3, 0.5))
    plt.tight_layout()
    plt.show()
