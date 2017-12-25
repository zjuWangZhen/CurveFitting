# -*- coding: utf-8 -*-
'''
@Author: wzlab
@Date  : 2017/12/20
@Desc  : 从excel文件中读取
        %DE:Percentage dosimetric errors 剂量测定的误差百分比  (Dcompass-D_TPS )/D_TPS 9100
        %GP:percentage gamma passing rate 相对百分率伽马传递率
'''
import xlrd
import xlwt
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import xlutils.copy


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import linear_model



def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


def read_col(file_path, t1_col_num, t2_col_num):
    '''
    Read two columns of data from the table

    :param file_path: the path of the excel file,if it is in the project directory,write the excel file name directly
    :param t1_col_num: the number of the selected column in table 'DE new'
    :param t2_col_num:  the number of the selected column in table '3D gamma'
    :return:x, y
    '''
    # open the excel file and read the data
    data = xlrd.open_workbook(file_path)
    # chose a table by name
    table_DE = data.sheet_by_name('DE new')
    table_GP = data.sheet_by_name('3D gamma')
    # get the entire column values
    x = table_GP.col_values(t1_col_num, 2, 71)
    y = table_GP.col_values(t2_col_num, 2, 71)

    return x, y


def write_col(file_path, table_name, col1, col2, header, col_num=1):
    '''
    Write two columns data in a table
    :param file_path: file path and file name
    :param table_name: the table name you want to create
    :param col1:the first column data
    :param col2:the second column data
    :param header:the title of two columns
    :param col_num:the number of column you write col1,and col2 will start from col_num+1
    :return:NUll
    '''
    file = xlwt.Workbook()
    table = file.add_sheet(table_name, cell_overwrite_ok=True)
    # column name
    table.write_merge(0, 0, col_num, col_num + 1, header)

    for i in range(len(col1)):
        table.write(i + 1, col_num, col1[i])
        table.write(i + 1, col_num + 1, col2[i])
    file.save(file_path)

def update_col(file_path, table_name, col1, col2, header, col_num=1):
    '''
       Write two columns data in a table without modify or change the original data
       :param file_path: file path and file name
       :param table_name: the table name you want to create
       :param col1:the first column data
       :param col2:the second column data
       :param header:the title of two columns
       :param col_num:the number of column you write col1,and col2 will start from col_num+1
       :return:NUll

       NOTE: You should close the file to be changed before you run this program, or you will be warned "Permission denied"
       '''
    r_file = xlrd.open_workbook(file_path)
    w_file = xlutils.copy.copy(r_file)
    table = w_file.get_sheet(table_name)

    # write column name and merge cells
    table.write_merge(0, 0, col_num, col_num + 1, header)

    for i in range(len(col1)):
        table.write(i + 1, col_num, col1[i])
        table.write(i + 1, col_num + 1, col2[i])
    w_file.save(file_path)

if __name__ == '__main__':
    (x_data, y_data) = read_col('data.xlsx', 29, 30)
    print(x_data,y_data)
    #write_col('demo.xls', 'sheet1', x_data, y_data, 'mean_GTV_%2', 0)
    #update_col('demo.xls', 'sheet1', x_data, y_data, 'D50_RP_%2', 2)
    # plt.scatter(x_data,y_data)
    # plt.show()
    #print(np.mat(x_data).T)
    # x_data = np.array(np.mat(x_data))
    # y_data = np.array(np.mat(y_data))






    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data,label='training points')
    plt.ion()  # 画出图象后程序继续运行
    plt.show()
    Y = curve_fitting(np.array(np.mat(x_data).T),np.array(np.mat(y_data).T))
    x_plot = np.linspace(-4.5, 1.5, 100).reshape([100, 1])
    ax.plot(x_plot, Y, 'k*',label="wz's fitting line")
    # print(type(Y),np.array(np.mat(y_data).T).shape)
    polynomial = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', linear_model.Ridge(fit_intercept=False))])
    #polynomial.fit(np.array(np.mat(x_data).T), np.array(np.mat(y_data).T))
    polynomial.fit(np.array(np.mat(x_data).T),np.array(np.mat(y_data).T))
    x_plot = np.linspace(-4.5, 1.5, 1000).reshape([1000, 1])
    y_plot = polynomial.predict(x_plot)

    ax.plot(x_plot, y_plot, 'r-', label="lc's fitting line")

    plt.xlabel('GTV mean')
    plt.ylabel('Individual volume 2%/2 mm')
    plt.legend(loc='best')
    plt.pause(0.1)
    plt.ioff()
    plt.show()
