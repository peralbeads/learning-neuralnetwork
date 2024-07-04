# implementing different loss function
# for logistic regression log loss is to be chosen
# not mean square error or mean absolute error

import numpy as np

y_predicted = np.array([1, 1, 0, 0, 1])
y_true = np.array([.30, 0.7, 1, 0, 0.5])

def mean_absolute_error(y_tru, y_prdicted):
    total_error = 0
    for yt, yp in zip(y_tru, y_prdicted):
        print(yt, yp)
        total_error += abs(yt - yp)
        print("total error : ", total_error)
    mae = total_error / len(y_true)
    return mae


def simple_mae(y_tr, y_pre):
    return np.mean(np.abs(y_tr - y_pre))

def mean_squared(y_tr, y_pre):
    return np.mean(np.square(y_tr - y_pre))


# print(simple_mae(y_true,y_predicted))
# print(mean_absolute_error(y_true,y_predicted))
# print(mean_squared(y_true,y_predicted))

epsilon = 1e-15



def log_loss(y_true, y_predicted):
    y_predict_new = [max(i, epsilon) for i in y_predicted]
    y_predict_new = [min(i, 1 - epsilon) for i in y_predict_new]
    y_predict_new = np.array(y_predict_new)
    return -np.mean(y_true*np.log(y_predict_new)+(1-y_true)*np.log(1-y_predict_new))


print(log_loss(y_true, y_predicted))