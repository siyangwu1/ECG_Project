import sys
sys.path.append('C:/Users/User/Desktop/document/ECG_project')
import tensorflow as tf
import numpy as np
import pandas as pd
import Utility
import pywt
import arrythmia_LSTM_output_archtecture
from sklearn.metrics import average_precision_score
import arrythmia_LSTM_archetechture
from tensorflow import keras
tf.keras.mixed_precision.set_global_policy('float32')
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()


def sk_pr_auc(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.float64)


def main():
    lstm_unit = [100, 100]
    accuracy = tf.keras.metrics.BinaryAccuracy()
    output_model = arrythmia_LSTM_output_archtecture.create_output_model(input_shape=(500, 2), lstm_units=lstm_unit, dropout_rate= 0, optimizer= keras.optimizers.Adam(), loss='binary_crossentropy', metrics=[accuracy, sk_pr_auc])
    output_model.load_weights('arrythmia_neural_net/train_log/saved_model')
    output_model.summary()

    path_to_ds1 = 'arrythmia_csv/DS1'
    path_to_ds2 = 'arrythmia_csv/DS2'
    path_to_ds3 = 'arrythmia_csv/DS3'
    path_to_svm = 'arrythmia_svm_features'

    X_train = np.load(f'{path_to_ds1}/DS1_X_df.npy')
    y_train = np.load(f'{path_to_ds1}/DS1_y_df.npy')

    X_val = np.load(f'{path_to_ds2}/DS2_X_df.npy')
    y_val = np.load(f'{path_to_ds2}/DS2_y_df.npy')

    X_test = np.load(f'{path_to_ds3}/DS3_X_df.npy')
    y_test = np.load(f'{path_to_ds3}/DS3_y_df.npy')

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