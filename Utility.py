import numpy as np
import pandas as pd
import glob, os
import wfdb
import filter_set
import ast



# compile all the csv corresponding to each record into a 
# single numpy file
def arrythmia_compile_data_frame(path_to_csv, path_to_db):
    y_df = []
    X_df = []
    fs_new = 100
    for direction in glob.glob(os.path.join(path_to_csv, '*.csv')):
        path, full_name = os.path.split(direction)
        # split filename and extension
        filename, _ = full_name.split('.')
        filename = filename.split('_')[0]

        temp_df = pd.read_csv(direction)
        sub_df = np.array([ele for ele in temp_df.values if ele[-1] == 'Normal' or ele[-1] == 'arrythmia'])
        if len(sub_df) > 0:
            record = wfdb.rdrecord(os.path.join(path_to_db, filename))
            idx_df = np.array([ast.literal_eval(ele) for ele in sub_df[:, 1]])
            idx_df = idx_df / (record.fs / fs_new)
            idx_df = np.array(idx_df).astype(int)
            p_signal =np.array(record.p_signal[:, 0])

            p_signal = filter_set.butter_highpass(data=p_signal, cutoff=0.5, fs=record.fs, order=2)
            p_signal = filter_set.down_sampling(data=p_signal, fs_orig=record.fs, fs_new=fs_new)

            subset_signal = [p_signal[start:end] for start, end in idx_df]

            X_df.append(subset_signal)

            binary_val = np.where(sub_df[:, -1] == 'Normal', 1, 0)

            y_df.append(binary_val)



    X_df = [ele for array in X_df for ele in array]
    y_df = [ele for array in y_df for ele in array]
    #X_df = np.stack(X_df, axis=0)
    #y_df = np.stack(y_df, axis=0)
    return X_df, y_df

def df_write_to_csv(path_to_save, X_df, y_df): 
    
    new_X_df = np.empty_like(shape=(X_df.shape[0], X_df.shape[1]), prototype= str)

    np.save(f'{path_to_save}/{os.path.split(path_to_save)[-1]}_X_df', X_df)
    np.save(f'{path_to_save}/{os.path.split(path_to_save)[-1]}_y_df', y_df)    
    '''
    for i in range(new_X_df.shape[0]):
        for j in range(new_X_df.shape[1]):
            new_X_df[i, j] = f'[{X_df[i, j, 0]}, {X_df[i, j, 1]}]'

    df = pd.DataFrame(
    new_X_df, y_df

    )
    df.to_csv(f'{path_to_save}/{os.path.split(path_to_save)[-1]}_data_frame.csv', index=True, header=False)
    '''



def AFIB_compile_data_frame(path_to_csv, path_to_db):
    y_df = []
    X_df = []
    for direction in glob.glob(os.path.join(path_to_csv, '*.csv')):
        path, full_name = os.path.split(direction)
        # split filename and extension
        filename, _ = full_name.split('.')
        filename = filename.split('_')[0]

        temp_df = pd.read_csv(direction)
        sub_df = np.array([ele for ele in temp_df.values if ele[-1] == '(N' or ele[-1] == '(AFIB'])
        if len(sub_df) > 0:
            record = wfdb.rdrecord(os.path.join(path_to_db, filename))
            idx_df = np.array([ast.literal_eval(ele) for ele in sub_df[:, 1]])
            idx_df = idx_df / (record.fs / 100)
            idx_df = np.array(idx_df).astype(int)
            p_signal =np.array(record.p_signal[:, 0])

            p_signal = filter_set.butter_highpass(data=p_signal, cutoff=0.5, fs=record.fs, order=2)
            p_signal = filter_set.down_sampling(data=p_signal, fs_orig=record.fs, fs_new=100)

            subset_signal = [p_signal[start:end] for start, end in idx_df]

            X_df.append(subset_signal)

            binary_val = np.where(sub_df[:, -1] == '(N', 1, 0)

            y_df.append(binary_val)


    X_df = [ele for array in X_df for ele in array]
    y_df = [ele for array in y_df for ele in array]
    num_normal = sum(y_df)

    print(f'number of normal is: {num_normal}')
    print(f'number of afib is: {len(y_df) - num_normal}')
    
            #X_df = np.stack(X_df, axis=0)
            #y_df = np.stack(y_df, axis=0)
    return X_df, y_df



def normalize(mean, std, mat):
    epsilon = 1e-8
    new_mat = (mat - mean) / (std + epsilon)
    return new_mat