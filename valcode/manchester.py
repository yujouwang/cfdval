"""
This module contains the tools to parse the LES data from Manchester
"""
import numpy as np
from pathlib import Path
import pandas as pd
import glob
import re
import pickle
from tqdm import tqdm
import logging
from numpy import genfromtxt




def _check_folder_name(folder_name):
    try:
        float(folder_name)
        return True
    except:
        # rint("Drop item in the folder: %s"%folder_name)
        return False


def _get_folders(folders):
    """ Search for all folders"""
    for folder in folders:
        check = _check_folder_name(folder)
        if check is True:
            pass
        else:
            folders.remove(folder)
    return folders


def _get_coordinates_of_probes(lines_from_text):
    x, y, z = [line.split() for line in lines_from_text[0:3]]
    x.remove('#')
    y.remove('#')
    z.remove('#')
    assert len(x) == len(y) and len(y) == len(z), "Check x, y, z keys"
    coord = list(zip(x, y, z))[1:]
    return coord


def probe_parser(file, dim=1):
    """ From the file, parse the time and probe data  """
    with open(file, 'r') as f:
        lines = f.readlines()

        # Initialize the data handles
    coords = _get_coordinates_of_probes(lines)
    N_probes = len(coords)
    time = []
    probe_data = {','.join(key): [[] for i in range(dim)] for key in coords}

    # Iterate
    N_line = len(lines[4:])
    data = np.zeros((N_line, N_probes, dim))
    for i_t, line in enumerate(lines[4:]):
        # Time
        t = line.split()[0]
        time.append(float(t))

        # Parse the data
        if dim == 1:
            keyword = '([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
            probes = re.findall('%s' % (keyword), line)
            assert len(probes) == (
                        N_probes + 1), "Check the parser. The number of the columns should be %s (including time)" % (
                        N_probes + 1)

        elif dim == 3:
            keyword = '([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)? [+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)? [+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
            probes = re.findall('%s' % (keyword), line)

            assert len(probes) == N_probes, "Check the parser. The number of the columns should be %s" % N_probes

        # Record each time step
        for i in range(N_probes):
            coord = ','.join(coords[i])
            if dim == 1:
                data[i_t][i][0] = float(probes[i + 1])

            elif dim == 3:
                U, V, W = probes[i].split()
                data[i_t][i][0] = float(U)
                data[i_t][i][1] = float(V)
                data[i_t][i][2] = float(W)
            else:
                raise ValueError("Check the dimension for the parser")

    for i in range(N_probes):
        coord = ','.join(coords[i])
        probe_data[coord] = data[:, i, :]

    return coords, np.array(time), probe_data


def process_Manchester_data(data_dir, save_to, file_name, dim, save_name=''):
    data_file_path = os.path.join(save_to, 'parsed_%s_%s.pkl' % (file_name, save_name))

    if os.path.exists(data_file_path):
        with open(data_file_path, "rb") as f:
            data = pickle.load(f)
    else:
        folders = _get_folders(glob.glob1(data_dir, '*'))
        probes, time, data = None, None, None
        for folder in folders:
            file = os.path.join(data_dir, folder, file_name)

            # Get probes
            p, t, d = probe_parser(file, dim)
            if time is None and data is None:
                time = t
                data = d
                probes = p

            else:
                assert (probes == p), "Different probes are detected in file: %s" % file
                time = np.concatenate((time, t))
                for probe in probes:
                    probe = ','.join(probe)
                    data[probe] = np.concatenate((data[probe], d[probe]), axis=0)

        # Save files
        data['time'] = time
        with open(data_file_path, 'wb') as f:
            pickle.dump(data, f)

    return data


class ManchesterProbes:
    """Get the Manchester"""

    def __init__(self, data_dir, save_to, probe_locs):
        self.data_dir = Path(data_dir)
        self.save_to = Path(save_to)
        self.probe_locs = probe_locs
        self.datatypes = ['T', 'k', 'U']

    def _get_data(self, data_type, overwrite, save_name=''):
        post_processed_data_path = self.save_to / "parsed_Mahchester" / f'parsed_{data_type}_{save_name}.pkl'

        if post_processed_data_path.exists() and overwrite == False:
            with open(post_processed_data_path, "rb") as f:
                data = pickle.load(f)
        else:
            print("LES file not found: start parsing")
            if data_type == 'U':
                dim = 3
            else:
                dim = 1
            save
            data = process_Manchester_data(self.data_dir, self.save_to, data_type, dim)
        return data

    def get_probe_data(self, probe, data_type, save_name='', overwrite=False):
        data = self._get_data(data_type, overwrite, save_name)
        time = data['time']
        probe_data = data[','.join(probe)]
        return time, probe_data


def _convert_array_to_df(column_names, data):
    return pd.DataFrame(data=data, columns=column_names)


def get_probe_table(file_path, column_names):
    data = _line_data_parser(file_path)
    data = np.array(data, dtype=float)
    data_shape = data.shape
    assert data_shape[1] == len(column_names), " The lenth of column names doesn't match the data shape "
    return _convert_array_to_df(column_names, data)



# ============================ Line data ===================================


def _find_line_csv_archer_filename(x_loc, direction, data_type):
    loc = str(round(0.021 * x_loc, 4)).split('.')[1]
    if x_loc > 0:
        sign = ''
    else:
        sign='-'
    if direction == 'H':
        dir_mcst = 'Z'
    else:
        dir_mcst = 'Y'

    if data_type == "T":
        file_name = '2256_profileT_x_%s%s_%s.csv' % (sign, loc, dir_mcst)
    else:
        file_name = '2256_profile_x_%s%s_%s.csv' % (sign, loc, dir_mcst)

    return file_name

# mapping the sub_dir name to the location, unit: Db
loc_map = {
    'line1_y': -7,
    'line2_y': 1,
    'line3_y': 2,
    'line4_y':-0.5,
    'line5_y':-2,
    'line6_y':-4.5
}
# reverse the key mapping
loc_map_rev = {value:key for key, value in loc_map.items()}


def _find_line_file_path(x_loc, direction, data_type):
    """ Find the file path based on the probe and data_type"""
    if data_type == 'TKE':
        folder_path = os.path.join(les_root, "lines", loc_map_rev[x_loc], '22.56')
        file_name = 'line_UPrime2Mean.xy'
    else:
        folder_path = os.path.join(les_root, "lines", "csv_archer")
        file_name = _find_line_csv_archer_filename(x_loc, direction, data_type)

    file_path = os.path.join(folder_path, file_name)
    return file_path


def _line_data_parser(file_path, start_from_line):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    content = []

    for line in lines[start_from_line:]:
        if 'nan' in line:
            pass
        else:
            for special_char in ['\n', '\t']:
                line = line.replace(special_char, '')
            line = line.split(" ")
            while " " in line:
                line.remove(" ")
            while  "" in line:
                line.remove('')
            content.append(line)

    return content

def _read_line_data(file_path, data_type):
    file_ext = file_path.split('.')[1]
    if file_ext == 'csv':
        data = genfromtxt(file_path, delimiter=',')
    else:
        data = _line_data_parser(file_path, start_from_line=1)
    return np.array(data, dtype=float)

table_column_id={
    'T':0,
    'U':0,
    'V':1,
    'W':2,
    'TKE':[0,1,2]
}

def _get_column_names(data_type):
    if data_type == 'T':
        column_names = ['T',
                         'TPrime2Mean',
                         'vtkValidPointMask',
                         'arc_length',
                         'X',
                         'Z',
                         'Y',
                         'vtkOriginalIndices',
                         '__vtkIsSelected__',
                         'vtkOriginalProcessIds']

    elif data_type in ['U', 'V', 'W']:
        column_names = ['U',
                         'V',
                         'W',
                         'vtkValidPointMask',
                         'arc_length',
                         'X',
                         'Z',
                         'Y',
                         'vtkOriginalIndices',
                         '__vtkIsSelected__',
                         'vtkOriginalProcessIds']
    elif data_type == 'TKE':
        column_names = ['Z', 'uu', 'vv', 'ww', 'uv', 'uw', 'vw']
    else:
        raise NotImplementedError("The data type is not supported")

    return column_names

def read_line_table(x_loc, direction, data_type):
    avail_types = ['T', 'U', 'V', 'W', 'TKE']
    assert data_type in avail_types, "The available data_types: %s" % avail_types
    file_path = _find_line_file_path(x_loc, direction, data_type)
    table = _read_line_data(file_path, data_type)
    column_names = _get_column_names(data_type)

    return pd.DataFrame(table, columns=column_names)



def get_line_data(x_loc, direction, data_type):
    """ Get the line data """
    df = read_line_table(x_loc, direction, data_type)
    if direction == 'V':
        axis = df['Z']
    elif direction == 'H':
        axis = df['Y']
    else:
        raise ValueError("Available direction: H or V")

    if data_type == 'TKE':
        value = df['uu'] + df['vv'] + df['ww']
    else:
        value = df[data_type]

    return axis, value

