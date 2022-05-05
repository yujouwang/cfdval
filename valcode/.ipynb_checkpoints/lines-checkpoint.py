import pandas as pd
import numpy as np
from tqdm import tqdm
import os 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from utils import Tables
from config import *

D_b = 21E-3
x_locs_star = [1, 2, -2, -0.5, -7, 4.5, -4.5]
x_locs = [0.021, 0.042, -0.042, 0.084, -0.084, -0.0105, -0.147, 0.0945, -0.0945]
dirs =['H', 'V']
root = r'F:\LES_Manchester\UoM_TJ_LES-20211013T183109Z-001\UoM_TJ_LES'

class ManchesterLine:
    def __init__(self, x_loc, direction):
        self.working_dir = os.path.join(root, 'lines', 'csv_archer')
        self.x_loc = x_loc
        self.direction = direction
        
    def get_line(self, data_type='U'):
        # Specify file name
        loc = str(round(0.021*self.x_loc, 4)).split('.')[1]
        if self.x_loc > 0:
            sign =''
        else:
            sign='-'
        
        # Specify file 
        if self.direction == 'H':
            dir_mcst ='Z'
        else:
            dir_mcst = 'Y'
            
        #       
        if data_type== 'T':
            file_name= '2256_profileT_x_%s%s_%s.csv'%(sign, loc, dir_mcst)
            column_mapper = {
                         'TMean': 'Temperature (K)',
                         'Points:0':'x',
                         'Points:1':'y',
                         'Points:2':'z' 
                    }
            
        if data_type in ['U', 'V', 'W']:   
            file_name= '2256_profile_x_%s%s_%s.csv'%(sign, loc, dir_mcst)
            column_mapper = {'UMean:0':'Velocity[i] (m/s)',
                         'UMean:2':'Velocity[j] (m/s)',
                         'UMean:1':'Velocity[k] (m/s)',
                         'Points:0':'x',
                         'Points:1':'y',
                         'Points:2':'z' 
                    }

        file_path = os.path.join(self.working_dir, file_name)
        file = pd.read_csv(file_path)
        file = file.rename(columns=column_mapper)
        return file


class TableLines(Tables):
    def __init__(self, working_dir, table_name, file_start_from=1000):
        super(TableLines, self).__init__(working_dir, table_name, file_start_from)
        self.working_dir = working_dir
        self.table_name = table_name
        self._init_files(file_start_from)
        self._init_lines_indices()
        self.save_to = os.path.join(DATA_DIR, 'parsed_CFD_LineTables')

        
    def __find_indices(self, x_loc, direction):
        """ Given the x_loc and the direction, output the indices """
        file = pd.read_csv(os.path.join(self.working_dir, self.files[0]))
        self.columns = file.keys()
        self.N_pts = len(file)
        self.N_cols = len(self.columns )
        self.N_t = len(self.files)
        
        if direction =='H':
            fixed = 'Z (m)'
        elif direction == 'V':
            fixed = 'Y (m)'
        else:
            raise ValueError("Check direction")
        
        file_f = file[abs(file['X (m)']-x_loc) < 1E-5] 
        file_f = file_f[abs(file_f[fixed] - 0) <1E-5]
        assert file_f.index.size <= 200, "More than 200 points are found: %s, %s"%(x_loc, direction)
        return file_f.index.values
    
    def _init_lines_indices(self):
        """ output several lists that indicates """
        indices = {}
        print("Initiating line indices...")
        for x_loc in x_locs_star:
            for direction in dirs:
                indices[(x_loc, direction)] = self.__find_indices(x_loc*D_b, direction)
        self.indices = indices    
        print("Done!")
        return 
    
    def _init_stat(self): 
        # Check if file exist
        save_to = os.path.join(self.save_to, 'mean.pkl')
        if os.path.exists(save_to):
            with open(save_to, "rb") as f:
                mean = pickle.load(f) 
            
                
            save_to = os.path.join(self.save_to, 'std.pkl')
            with open(save_to, "rb") as f:
                std = pickle.load(f) 
            
        else:
            time_history = np.zeros((self.N_t, self.N_pts, self.N_cols))

            for i_t, file in enumerate(tqdm(self.files)):
                data = pd.read_csv(os.path.join(self.working_dir, file))
                time_history[i_t, :, :] = data

            mean = time_history.mean(axis=0)
            std = time_history.std(axis=0)

            # Save 
            save_to = os.path.join(self.save_to, 'mean.pkl')
            with open(save_to, 'wb') as f:
                pickle.dump(mean, f)     

            save_to = os.path.join(self.save_to, 'std.pkl')
            with open(save_to, 'wb') as f:
                pickle.dump(std, f)   
        return mean, std
    
    def get_lineprobe(self, column, x_loc, direction):
        mean, std = self._init_stat()
        
        col_index = list(self.columns).index(column)
        
        all_means = mean[:, col_index]
        all_stds = std[:, col_index]
        
        indices_for_line = self.indices[(x_loc, direction)]
        return all_means[indices_for_line], all_stds[indices_for_line]
        
        
    


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
    l_stc= TableLines(working_dir, table_name)
    mean_stc, std = l_stc.get_lineprobe(column=column, x_loc=x_loc, direction='V')
    Z, _ = l_stc.get_lineprobe(column="Z (m)", x_loc=x_loc, direction='V')
    if data_type == 'T':
        mean_stc = mean_stc+0.15 
    
    
    # Plot
    f, ax  = plt.subplots(1,1, figsize=(4,6))
    ax.plot(mean_les[column], mean_les['y'], 'r-', label='LES')
    ax.plot(mean_stc , Z,  'bo', label='Struct')
    
    ax.set_xlabel(column)
    ax.set_ylabel('Z (m)')
    ax.legend()
    plt.title('loc: %s Db'%x_loc)
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
    l_stc= TableLines(working_dir, table_name)
    mean_stc, std = l_stc.get_lineprobe(column=column, x_loc=x_loc, direction='H')
    Z, _ = l_stc.get_lineprobe(column="Y (m)", x_loc=x_loc, direction='H')
    if data_type == 'T':
        mean_stc = mean_stc+0.15 
    
    
    # Plot
    f, ax  = plt.subplots(1,1, figsize=(4,6))
    ax.plot(mean_les[column], mean_les['z'], 'r-', label='LES')
    ax.plot(mean_stc , Z,  'bo', label='Struct')
    
    ax.set_xlabel(column)
    ax.set_ylabel('Z (m)')
    ax.legend()
    plt.title('loc: %s Db'%x_loc)
    if xlim:
        ax.set_xlim(xlim)
    plt.show()