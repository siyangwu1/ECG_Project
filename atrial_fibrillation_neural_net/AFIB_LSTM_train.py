import sys
sys.path.append('C:/Users/User/Desktop/document/ECG_project')
import tensorflow as tf
import numpy as np
import pandas as pd
import Utility
import pywt
from sklearn.metrics import average_precision_score
import AFIB_LSTM_archetechture
from tensorflow import keras
tf.keras.mixed_precision.set_global_policy('float32')
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
from  sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
import matplotlib.pyplot as plt

def sk_pr_auc(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.float64)

def main():


    lstm_unit = [100, 100]
    path_to_ds1 = 'atrial_fibrillation_csv/DS1'
    path_to_ds2 = 'atrial_fibrillation_csv/DS2'
    path_to_ds3 = 'atrial_fibrillation_csv/DS3'

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

    accuracy = tf.keras.metrics.BinaryAccuracy()
    X_train = Utility.normalize(X_mean, X_std, X_train)
    X_val = Utility.normalize(X_mean, X_std, X_val)
    X_test = Utility.normalize(X_mean, X_std, X_test)


    test = AFIB_LSTM_archetechture.create_model(input_shape=(500, 2), lstm_units=lstm_unit, dropout_rate= 0.4, optimizer= keras.optimizers.Adam(), loss='binary_crossentropy', metrics=[accuracy, sk_pr_auc])
    #test.load_weights('arrythmia_neural_net/train_log/saved_model')
    #test.layers[3].trainable = False
    test.summary()


    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_sk_pr_auc', patience=5)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('atrial_fibrillation_neural_net/train_log/saved_model', monitor='val_sk_pr_auc', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only=True)

    test.fit(x= X_train, y=y_train, batch_size=256, validation_data=(X_val, y_val), epochs=50, shuffle=True, verbose=1, callbacks=[earlystopping, checkpoint])

    model = AFIB_LSTM_archetechture.create_model(input_shape=(500, 2), lstm_units=lstm_unit, dropout_rate= 0.4, optimizer= keras.optimizers.Adam(), loss='binary_crossentropy', metrics=[accuracy, sk_pr_auc])
    model.load_weights('atrial_fibrillation_neural_net/train_log/saved_model')

    y_pred = model.predict(X_test)
    auc_pr = average_precision_score(y_score=y_pred, y_true=y_test)
    y_pred_label = [1 if prob >= 0.5 else 0 for prob in y_pred]

    ConfusionMatrixDisplay.from_predictions(y_pred=y_pred_label, y_true=y_test)
    PrecisionRecallDisplay.from_predictions(y_pred=y_pred, y_true=y_test)
    plt.show()

    print(f"the auc precision and recall is: {auc_pr}")

if __name__ == "__main__":
    main()