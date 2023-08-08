import sklearn
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from  sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
 
def main():
    path_to_data = 'atrial_fibrillation_svm_feature'

    X_train = np.load(f'{path_to_data}/X_train.npy')
    y_train = np.load(f'{path_to_data}/y_train.npy')

    X_val = np.load(f'{path_to_data}/X_val.npy')
    y_val = np.load(f'{path_to_data}/y_val.npy')

    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)

    X_test = np.load(f'{path_to_data}/X_test.npy')
    y_test = np.load(f'{path_to_data}/y_test.npy')

    X_train = preprocessing.normalize(X_train)
    X_val = preprocessing.normalize(X_val)
    X_test = preprocessing.normalize(X_test)
    C_range = np.arange(start=0.08, stop=0.1, step=0.001)
    gamma_range =  np.arange(start=1.5, stop = 2.5, step=0.1)
    weights = []
    

    # We will give more weight to class 1, since it's the minority class
    for i in np.arange(0.3, 0.5, 0.01):
        weights.append({0: 1 - i, 1: i})
    


    svc = SVC()
    grid = GridSearchCV(svc, param_grid = {'C':[1.3], 'kernel':['rbf'], 'gamma' : [2.1], 'class_weight': [{0: 0.5, 1: 0.5}]}, n_jobs=-1, verbose=2, scoring='average_precision', cv  = 2)
    grid.fit(X_train, y_train)
    y_score = grid.decision_function(X_test)
    y_pred = grid.predict(X_test)
    auc_pr = average_precision_score(y_score=y_score, y_true=y_test)

    ConfusionMatrixDisplay.from_predictions(y_pred=y_pred, y_true=y_test)
    PrecisionRecallDisplay.from_predictions(y_pred = y_score, y_true = y_test)
    plt.show()

    best_param = grid.best_params_
    best_score = grid.best_score_
    print(f'test auc pr is :{auc_pr}, with parameter_C:')
    print(best_param)
    print(f'best validation score is {best_score}')


    
    #print(accuracy_score(y_pred=y_hat, y_true=y_test))


if __name__ == "__main__":
    main()