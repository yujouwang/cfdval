"""
This module contains tools for parsing the data from starccm+ table outputs
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import os
import glob

from utils import natural_sort
from config import*

class Tables:
    def __init__(self, working_dir, table_name, file_start_from):
        self.working_dir = working_dir
        self.table_name = table_name
        self._init_files(file_start_from)

    def _init_files(self, file_start_from):
        """ Search all the tables and sort """
        files = glob.glob1(self.working_dir, '%s*.csv' % self.table_name)
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
        assert len(index) == 1, "the lenth of the x,y,z=(%s, %s, %s) is %s" % (x, y, z, index)
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
        assert type(name) == str, "The name to be converted must be str"
        name = name.replace(" (", "_")
        name = name.replace(")", "")
        name = name.replace("[", "_")
        name = name.replace("]", "")
        name = name.replace("/", "-")
        return name

    def parse_all_columns(self, overwrite=False):
        # Check if the file exists
        save_name = self.__convert_saved_name(self.columns[0])
        save_to = os.path.join(self.save_to, '%s.pkl' % save_name)
        if os.path.exists(save_to) and overwrite == False:
            print(f"Files are already parsed, use 'read_file' to retrieve them")
            return

        else:
            N_cols = len(self.columns)
            data_holder = [{','.join(key): [] for key in self.probe_locs}
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
                save_to = os.path.join(self.save_to, '%s.pkl' % save_name)
                with open(save_to, 'wb') as f:
                    pickle.dump(data_holder[i], f)
            return

    def read_file(self, column, probe):
        save_name = self.__convert_saved_name(column)
        save_to = os.path.join(self.save_to, '%s.pkl' % save_name)
        with open(save_to, "rb") as f:
            data = pickle.load(f)

        probe_name = ','.join(probe)

        return np.array(data[probe_name])



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
        self.N_cols = len(self.columns)
        self.N_t = len(self.files)

        if direction == 'H':
            fixed = 'Z (m)'
        elif direction == 'V':
            fixed = 'Y (m)'
        else:
            raise ValueError("Check direction")

        file_f = file[abs(file['X (m)'] - x_loc) < 1E-5]
        file_f = file_f[abs(file_f[fixed] - 0) < 1E-5]
        assert file_f.index.size <= 200, "More than 200 points are found: %s, %s" % (x_loc, direction)
        return file_f.index.values

    def _init_lines_indices(self):
        """ output several lists that indicates """
        indices = {}
        for x_loc in x_locs_star:
            for direction in dirs:
                indices[(x_loc, direction)] = self.__find_indices(x_loc * D_b, direction)
        self.indices = indices
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

