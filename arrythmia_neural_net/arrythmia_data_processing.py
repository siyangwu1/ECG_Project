import sys
sys.path.append('C:/Users/User/Desktop/document/ECG_project')
import tensorflow as tf
import numpy as np
import pandas as pd
import Utility
import pywt



def main():
    '''
    path_to_arr_DB = 'arrythmia_db'
    path_to_arr_csv = 'arrythmia_csv/DS3'
    X_df, y_df = Utility.arrythmia_compile_data_frame(path_to_csv= path_to_arr_csv, path_to_db= path_to_arr_DB)

    X_df = np.array(X_df)
    print(X_df.shape)

    X_df = np.array([pywt.dwt(signal, 'db1') for signal in X_df])
    
    X_df = X_df.transpose(0, 2, 1)
    print(X_df.shape)
    Utility.df_write_to_csv(path_to_arr_csv, X_df=X_df, y_df=y_df)
    
    '''

    
    path_to_AFIB_DB = 'atrial_fibrillation_db'

    path_to_AFIB_csv = 'atrial_fibrillation_csv/DS3'

    X_df, y_df = Utility.AFIB_compile_data_frame(path_to_csv=path_to_AFIB_csv, path_to_db = path_to_AFIB_DB)


    X_df = np.array(X_df)
    y_df = np.array(y_df)
    print(X_df.shape)
    print(y_df.shape)
    X_df = np.array([pywt.dwt(signal, 'db1') for signal in X_df])
    X_df = X_df.transpose(0, 2, 1)
    print(X_df.shape)
    Utility.df_write_to_csv(path_to_AFIB_csv, X_df=X_df, y_df=y_df)   
    
if __name__ == "__main__":
    main()