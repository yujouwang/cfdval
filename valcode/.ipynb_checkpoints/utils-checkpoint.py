import numpy as np
import os 
import pandas as pd
import glob 
import re
import pickle
from tqdm import tqdm
from config import *


def _check_folder_name(folder_name):
    try:
        float(folder_name)
        return True
    except:
        #rint("Drop item in the folder: %s"%folder_name)
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
    coord = list(zip(x,y,z))[1:]
    return coord

def parser(file, dim=1):
    """ From the file, parse the time and probe data  """
    with open(file, 'r') as f:
        lines = f.readlines()    
    
    # Initialize the data handles
    coords = _get_coordinates_of_probes(lines)
    N_probes = len(coords)
    time = []
    probe_data = {','.join(key):[[] for i in range(dim)] for key in coords}

    # Iterate
    N_line = len(lines[4:])
    data = np.zeros((N_line, N_probes, dim))
    for i_t, line in enumerate(lines[4:]):
        # Time 
        t = line.split()[0]
        time.append(float(t))
        
        # Parse the data 
        if dim ==1:
            keyword = '([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
            probes = re.findall('%s'%(keyword), line)
            assert len(probes) == (N_probes+1), "Check the parser. The number of the columns should be %s (including time)"%(N_probes+1)

        elif dim ==3:
            keyword = '([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)? [+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)? [+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'    
            probes = re.findall('%s'%(keyword), line)

            assert len(probes) == N_probes, "Check the parser. The number of the columns should be %s"%N_probes

        # Record each time step
        for i in range(N_probes):
            coord = ','.join(coords[i])            
            if dim==1:
                data[i_t][i][0] = float(probes[i+1])
                
            elif dim==3:
                U,V,W = probes[i].split()
                data[i_t][i][0] = float(U)
                data[i_t][i][1] = float(V)
                data[i_t][i][2] = float(W)
            else:
                raise ValueError("Check the dimension for the parser")
    
    for i in range(N_probes):
        coord = ','.join(coords[i])
        probe_data[coord] = data[:, i,:]
        
    return coords, np.array(time), probe_data



def process_Manchester_data(working_dir, file_name, dim, save_name=''):

    save_to = os.path.join(DATA_DIR, 'parsed_Mahchester')
    data_file_path = os.path.join(save_to, 'parsed_%s_%s.pkl'%(file_name, save_name))
    
    if os.path.exists(data_file_path):
        with open(data_file_path, "rb") as f:
            data = pickle.load(f)                
    else:
        folders = _get_folders(glob.glob1(working_dir, '*'))
        probes, time, data = None, None, None
        for folder in folders:
            file = os.path.join(working_dir, folder, file_name)

            # Get probes
            p, t, d =  parser(file, dim)        
            if time is None and data is None:
                time = t
                data = d
                probes = p

            else:
                assert(probes==p), "Different probes are detected in file: %s"%file
                time = np.concatenate((time, t))        
                for probe in probes:
                    probe = ','.join(probe)
                    data[probe] = np.concatenate((data[probe], d[probe]), axis=0)

        # Save files 
        data['time'] = time
        with open(data_file_path, 'wb') as f:
            pickle.dump(data, f)
            
    return data
                
        
class LESProbes:
    def __init__(self, working_dir, probe_locs, save_to=None):
        self.probe_locs = probe_locs
        self.datatypes =  ['T', 'k', 'U' ]
        self.working_dir = working_dir        
    
    def _get_data(self, data_type, overwrite, save_name=''):
        data_file_path = os.path.join(DATA_DIR, 'parsed_Mahchester', 'parsed_%s_%s.pkl'%(data_type, save_name))
        
        if os.path.exists(data_file_path) and overwrite==False:
            with open(data_file_path, "rb") as f:
                data = pickle.load(f)
        else:
            print("LES file not found: start parsing")
            if data_type == 'U':
                dim =3 
            else:
                dim =1                 
            data = process_Manchester_data(self.working_dir, data_type, dim)
        return data
    
    def get_probe_data(self, probe, data_type, save_name='', overwrite=False):
        data = self._get_data(data_type, overwrite, save_name)
        time = data['time']
        probe_data = data[','.join(probe)]
        return time, probe_data

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

class Tables:
    def __init__(self, working_dir, table_name, file_start_from):
        self.working_dir = working_dir
        self.table_name = table_name 
        self._init_files(file_start_from)
        
    def _init_files(self, file_start_from):
        """ Search all the tables and sort """
        files = glob.glob1(self.working_dir, '%s*.csv'%self.table_name)
        self.files = natural_sort(files)[file_start_from:]        
        return 
        

class TableProbes(Tables):
    def __init__(self, working_dir, table_name, probe_locs, file_start_from=0):
        super(TableProbes, self).__init__(working_dir, table_name, file_start_from)        
        self.probe_locs = probe_locs
        self._init_probes()
        self.save_to = os.path.join(DATA_DIR, 'parsed_CFD_tables')
        
    def __find_probe_index(self, x, y, z, data, tolerance=1E-5):
        """"""
        data_f = data[abs(data['X (m)'] - x) < tolerance]
        data_f = data_f[abs(data_f['Y (m)'] - z) < tolerance]
        data_f = data_f[abs(data_f['Z (m)'] - y) < tolerance]
        index = data_f.index
        assert len(index)==1, "the lenth of the x,y,z=(%s, %s, %s) is %s"%(x,y,z, index)
        return index[0]
    
    def _init_probes(self):
        data = pd.read_csv(os.path.join(self.working_dir, self.files[0]))
        indices = []
        for probe in self.probe_locs:
            x, y, z = [float(i) for i in probe]
            index = self.__find_probe_index(x, y, z, data)
            indices.append(index)
        self.indices = indices 
        self.columns = data.keys()
        return
    
    def __convert_saved_name(self, name):
        assert type(name)==str, "The name to be converted must be str"
        name = name.replace(" (", "_")
        name = name.replace(")","")
        name = name.replace("[", "_")
        name = name.replace("]", "")
        name = name.replace("/","-")
        return name
    
    def parse_all_columns(self, overwrite=False):
        # Check if the file exists 
        save_name = self.__convert_saved_name(self.columns[0])
        save_to = os.path.join(self.save_to, '%s.pkl'%save_name)
        if os.path.exists(save_to) and overwrite==False:
            print(f"Files are already parsed, use 'read_file' to retrieve them")
            return 
        
        else:        
            N_cols = len(self.columns)
            data_holder = [ {','.join(key):[] for key in self.probe_locs} 
                           for i in range(N_cols)]

            # For each file, get the data 
            for file in tqdm(self.files):
                table = pd.read_csv(os.path.join(self.working_dir, file))
                for i, column in enumerate(self.columns):
                    to_dict = data_holder[i] 

                    for j, probe in enumerate(self.probe_locs):
                        dict_name = ','.join(probe)
                        index = self.indices[j]
                        value = table[column].iloc[index]
                        to_dict[dict_name].append(value)

            # save data 
            for i, column in enumerate(self.columns):
                save_name = self.__convert_saved_name(column)
                save_to = os.path.join(self.save_to, '%s.pkl'%save_name)
                with open(save_to, 'wb') as f:
                    pickle.dump(data_holder[i], f)            
            return 
    
    def read_file(self, column, probe):
        save_name = self.__convert_saved_name(column)
        save_to = os.path.join(self.save_to, '%s.pkl'%save_name)
        with open(save_to, "rb") as f:
            data = pickle.load(f) 
            
        probe_name = ','.join(probe)
        
        return np.array(data[probe_name])
  
        
        
        
        
    
    
    
        
    
