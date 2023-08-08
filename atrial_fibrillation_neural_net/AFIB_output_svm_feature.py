import sys
sys.path.append('C:/Users/User/Desktop/document/ECG_project')
import tensorflow as tf
import numpy as np
import pandas as pd
import Utility
import pywt
import AFIB_LSTM_output_archtecture
import AFIB_LSTM_archetechture
from tensorflow import keras
tf.keras.mixed_precision.set_global_policy('float32')
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()


def main():
    lstm_unit = [100, 100]
    output_model = AFIB_LSTM_output_archtecture.create_output_model(input_shape=(500, 2), lstm_units=lstm_unit, dropout_rate= 0.4, optimizer= keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    output_model.load_weights('atrial_fibrillation_neural_net/train_log/saved_model')
    output_model.summary()

    path_to_ds1 = 'atrial_fibrillation_csv/DS1'
    path_to_ds2 = 'atrial_fibrillation_csv/DS2'
    path_to_ds3 = 'atrial_fibrillation_csv/DS3'
    path_to_svm = 'atrial_fibrillation_svm_feature'

    X_train = np.load(f'{path_to_ds1}/DS1_X_df.npy')
    y_train = np.load(f'{path_to_ds1}/DS1_y_df.npy')

    X_val = np.load(f'{path_to_ds2}/DS2_X_df.npy')
    y_val = np.load(f'{path_to_ds2}/DS2_y_df.npy')

    X_test = np.load(f'{path_to_ds3}/DS3_X_df.npy')
    y_test = np.load(f'{path_to_ds3}/DS3_y_df.npy')


    '''
    np.random.seed(42)
    train_index = np.random.choice(X_train.shape[0], 18000, replace=False)
    val_index = np.random.choice(X_val.shape[0], 3000, replace=False)
    test_index = np.random.choice(X_test.shape[0], 3000, replace=False)

    X_train = X_train[train_index]
    y_train = y_train[train_index]

    X_val = X_val[val_index]
    y_val = y_val[val_index]


    X_test = X_test[test_index]
    y_test = y_test[test_index]
    '''
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    #y_mean = np.mean(y_train, axis=0)
    #y_std = np.std(y_train, axis=0)

    X_train = Utility.normalize(X_mean, X_std, X_train)
    X_val = Utility.normalize(X_mean, X_std, X_val)
    X_test = Utility.normalize(X_mean, X_std, X_test)

    feature_train = output_model.predict(X_train)

    feature_val = output_model.predict(X_val)
    feature_test = output_model.predict(X_test)

    np.save(f"{path_to_svm}/X_train", feature_train)
    np.save(f"{path_to_svm}/y_train", y_train)
    
    np.save(f"{path_to_svm}/X_val", feature_val)
    np.save(f"{path_to_svm}/y_val", y_val)
    
    np.save(f"{path_to_svm}/X_test", feature_test)
    np.save(f"{path_to_svm}/y_test", y_test)
if __name__ == "__main__":
    main()