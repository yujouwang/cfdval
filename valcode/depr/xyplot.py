import numpy as np 
import pandas as pd 


def _pp_temp(probe_name):
    file_name='T'
    # LES data
    time_les, temp_les = get_temp(working_dir, file_name, probe_name)
    temp_les = (temp_les- T_c_les)/(T_h_les-T_c_les)
    # Struct data
    temp_stc = df_struct_temp[','.join(probe_name)]
    temp_stc = (temp_stc- T_c_stc)/(T_h_stc-T_c_stc)
    return temp_les, temp_stc

def _pp_TKE(probe_name):
    file_name = 'k'
    time_les, les = get_temp(working_dir, file_name, probe_name)
    stc = df_struct_tke[','.join(probe_name)]
    return les, stc


def _pp_Velocity(probe_name):
    les_U, les_V, les_W = [Us, Vs, Ws]
    
    stc_U = df_struct_U[','.join(probe_name)].values
    stc_V = df_struct_V[','.join(probe_name)].values
    stc_W = df_struct_W[','.join(probe_name)].values
    
    return np.array([les_U, les_V, les_W]), np.array([stc_U, stc_V, stc_W])

def _stat(group, les, stc, mean_les, std_les, mean_stc, std_stc, datatype):
    if datatype == 'Velocity':
        mean_les[group].append(les.mean(axis=1))
        std_les[group].append(les.std(axis=1))
        mean_stc[group].append(stc.mean(axis=1))
        std_stc[group].append(stc.std(axis=1))
        
    else:
        mean_les[group].append(les.mean())
        std_les[group].append(les.std())
        mean_stc[group].append(stc.mean())
        std_stc[group].append(stc.std())
        
    return mean_les, std_les, mean_stc, std_stc
        
    
def get_probes_data(probe_locations, datatype='T'):
    """ Given probe_locations, output the mean and std values """
    # Data processing
    x_unique = np.unique(np.array(probe_locations)[:,0])
    
    mean_les = [[] for i in range(len(x_unique))]
    std_les = [[] for i in range(len(x_unique))]
    mean_stc = [[] for i in range(len(x_unique))]
    std_stc = [[] for i in range(len(x_unique))]

    
    for probe_name in tqdm(probe_locations):
        # Identify the group
        group = np.where(x_unique == probe_name[0])[0][0]
        
        # Get data
        if datatype =='T':
            les, stc = _pp_temp(probe_name)
        
        elif datatype =='TKE':
            les, stc = _pp_TKE(probe_name)
            
        elif datatype == 'Velocity':
            les, stc = _pp_Velocity(probe_name)
            
        print(group)
        mean_les, std_les, mean_stc, std_stc = _stat(group, les, stc, mean_les, std_les, mean_stc, std_stc, datatype)

            
    clear_output()
    return mean_les, std_les, mean_stc, std_stc