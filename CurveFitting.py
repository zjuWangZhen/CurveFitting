# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2017/12/23
@Desc  : 
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np


def curve_fitting_nn(x, y):
    model = Sequential()
    model.add(Dense(100, activation='tanh', input_dim=1))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x, y, batch_size=12, nb_epoch=1000, shuffle=True)

    # x_test = np.linspace(-4.5, 1.5, 100).reshape([100, 1])
    # y_test = model.predict(x_test)
    return model


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import linear_model


def polynomial_curve_fitting(x, y):
    polynomial = Pipeline([('poly', PolynomialFeatures(degree=2)),
                           ('linear', linear_model.Ridge(fit_intercept=False))])

    polynomial.fit(x, y)
    return polynomial


from ExcelUtils import *

if __name__ == '__main__':
    (x_data, y_data) = read_col('data.xlsx', 29, 30)
    print(x_data, y_data)
    x_train = np.array(x_data).reshape([-1, 1])
    y_train = np.array(y_data).reshape([-1, 1])

    model = curve_fitting_nn(x_train, y_train)
    x_test = np.linspace(-4.4, 1.5, 100).reshape([-1, 1])
    y_test = model.predict(x_test)
    mse1 = sum((y_train - model.predict(x_train)) ** 2) / y_train.size
    print('MSE of wz:  ',mse1)

    polynomial = polynomial_curve_fitting(x_train, y_train)
    y_test1 = polynomial.predict(x_test)
    mse2 = sum((y_train - polynomial.predict(x_train)) ** 2 )/ y_train.size
    print('MSE of lc:  ', mse2)

    plt.scatter(x_train, y_train, label='training point')
    plt.plot(x_test, y_test, 'r-', label='curve fitting')

    plt.plot(x_test, y_test1, 'k.', label='polynomial curve fitting')
    plt.legend(loc='best')
    plt.show()

