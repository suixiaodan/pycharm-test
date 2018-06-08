# -*- coding: utf-8 -*-

import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle
import xlrd
import pandas as pd
import csv

reload(sys)
sys.setdefaultencoding('utf8')


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def read_data1(data_file, radio): #data_file是数据文件， radio是测试集占整个数据集的比例,用了进行01，234分类的
    #data = xlrd.open_workbook(gtgrading)
    data = xlrd.open_workbook(data_file)
    table = data.sheets()[1]  # 通过索引顺序获取
    nrows = table.nrows
    traindata_num = int(round(nrows * (1 - radio)))
    testdata_num = nrows - traindata_num
    gtDRgrading = table.col_values(6)
    train_y = gtDRgrading[:traindata_num]
    train_y = np.array(train_y)
    test_y = gtDRgrading[traindata_num:]
    test_y = np.array(test_y)

    nrowsdr = table.nrows
    ma_num = table.col_values(0)
    max_ma_num = max(ma_num)
    hr_num = table.col_values(1)
    max_hr_num = max(hr_num)
    he_num = table.col_values(2)
    max_he_num = max(he_num)
    ma_area = table.col_values(3)
    max_ma_area = max(ma_area)
    hr_area = table.col_values(4)
    max_hr_area = max(hr_area)
    he_area = table.col_values(5)
    max_he_area = max(he_area)
    train_x = []
    for i in range(traindata_num):
        a = table.row_values(i)
        a = [a[0]/max_ma_num, a[3]/max_ma_area,]
        train_x.append(a)
    train_x = np.array(train_x)
    test_x = []
    for j in range(traindata_num, nrowsdr):
        b = table.row_values(j)
        b = [b[0]/max_ma_num, b[3]/max_ma_area]
        test_x.append(b)
    test_x = np.array(test_x)
    return train_x, train_y, test_x, test_y

def read_data2(data_file, radio): #data_file是数据文件， radio是测试集占整个数据集的比例,用了进行01，234分类的
    #data = xlrd.open_workbook(gtgrading)
    data = xlrd.open_workbook(data_file)
    table = data.sheets()[2]  # 通过索引顺序获取
    nrows = table.nrows
    traindata_num = int(round(nrows * (1 - radio)))
    testdata_num = nrows - traindata_num
    gtDRgrading = table.col_values(6)
    train_y = gtDRgrading[:traindata_num]
    train_y = np.array(train_y)
    test_y = gtDRgrading[traindata_num:]
    test_y = np.array(test_y)

    nrowsdr = table.nrows
    ma_num = table.col_values(0)
    max_ma_num = max(ma_num)
    hr_num = table.col_values(1)
    max_hr_num = max(hr_num)
    he_num = table.col_values(2)
    max_he_num = max(he_num)
    ma_area = table.col_values(3)
    max_ma_area = max(ma_area)
    hr_area = table.col_values(4)
    max_hr_area = max(hr_area)
    he_area = table.col_values(5)
    max_he_area = max(he_area)
    train_x = []
    for i in range(traindata_num):
        a = table.row_values(i)
        a = [a[0]/max_ma_num, a[1]/max_hr_num, a[3]/max_ma_area]
        train_x.append(a)
    train_x = np.array(train_x)
    test_x = []
    for j in range(traindata_num, nrowsdr):
        b = table.row_values(j)
        b = [b[0]/max_ma_num, b[1]/max_hr_num, b[3]/max_ma_area]
        test_x.append(b)
    test_x = np.array(test_x)
    return train_x, train_y, test_x, test_y

def read_data3(data_file, radio): #data_file是数据文件， radio是测试集占整个数据集的比例,用了进行01，234分类的
    #data = xlrd.open_workbook(gtgrading)
    data = xlrd.open_workbook(data_file)
    table = data.sheets()[3]  # 通过索引顺序获取
    nrows = table.nrows
    traindata_num = int(round(nrows * (1 - radio)))
    testdata_num = nrows - traindata_num
    gtDRgrading = table.col_values(6)

    train_y = gtDRgrading[:traindata_num]
    train_y = np.array(train_y)
    test_y = gtDRgrading[traindata_num:]
    test_y = np.array(test_y)

    nrowsdr = table.nrows
    ma_num = table.col_values(0)
    max_ma_num = max(ma_num)
    hr_num = table.col_values(1)
    max_hr_num = max(hr_num)
    he_num = table.col_values(2)
    max_he_num = max(he_num)
    ma_area = table.col_values(3)
    max_ma_area = max(ma_area)
    hr_area = table.col_values(4)
    max_hr_area = max(hr_area)
    he_area = table.col_values(5)
    max_he_area = max(he_area)
    train_x = []
    for i in range(traindata_num):
        a = table.row_values(i)
        a = [a[4]/max_hr_area]
        train_x.append(a)
    train_x = np.array(train_x)
    test_x = []
    for j in range(traindata_num, nrowsdr):
        b = table.row_values(j)
        b = [b[4]/max_hr_area]
        test_x.append(b)
    test_x = np.array(test_x)
    return train_x, train_y, test_x, test_y

def read_data4(data_file, radio): #data_file是数据文件， radio是测试集占整个数据集的比例,用了进行01，234分类的
    #data = xlrd.open_workbook(gtgrading)
    data = xlrd.open_workbook(data_file)
    table = data.sheets()[4]  # 通过索引顺序获取
    nrows = table.nrows
    traindata_num = int(round(nrows * (1 - radio)))
    testdata_num = nrows - traindata_num
    gtDRgrading = table.col_values(6)
    #print len(gtDRgrading)
    train_y = gtDRgrading[:traindata_num]
    train_y = np.array(train_y)
    test_y = gtDRgrading[traindata_num:]
    test_y = np.array(test_y)

    nrowsdr = table.nrows
    ma_num = table.col_values(0)
    max_ma_num = max(ma_num)
    hr_num = table.col_values(1)
    max_hr_num = max(hr_num)
    he_num = table.col_values(2)
    max_he_num = max(he_num)
    ma_area = table.col_values(3)
    max_ma_area = max(ma_area)
    hr_area = table.col_values(4)
    max_hr_area = max(hr_area)
    he_area = table.col_values(5)
    max_he_area = max(he_area)
    train_x = []
    for i in range(traindata_num):
        a = table.row_values(i)
        a = [a[4]/max_hr_area, a[5]/max_he_area]
        train_x.append(a)
    train_x = np.array(train_x)
    test_x = []
    for j in range(traindata_num, nrowsdr):
        b = table.row_values(j)
        b = [b[4]/max_hr_area, b[5]/max_he_area]
        test_x.append(b)
    test_x = np.array(test_x)
    return train_x, train_y, test_x, test_y

def read_data(data_file, gtgrading, radio): #data_file是数据文件， gtgrading是分级的ground truth, radio是测试集占整个数据集的比例
    #data = xlrd.open_workbook(gtgrading)
    data = xlrd.open_workbook(data_file)
    table = data.sheets()[0]  # 通过索引顺序获取
    #table = data.sheets()[1]  # 通过索引顺序获取
    nrows = table.nrows
    traindata_num = int(round(nrows * (1 - radio)))
    testdata_num = nrows - traindata_num
    gtDRgrading = table.col_values(6)
    #print len(gtDRgrading)
    train_y = gtDRgrading[:traindata_num]
    train_y = np.array(train_y)
    test_y = gtDRgrading[traindata_num:]
    test_y = np.array(test_y)
    #print train_y
    #print test_y
    datadr = xlrd.open_workbook(data_file)
    tabledr = datadr.sheets()[0]  # 通过索引顺序获取
    nrowsdr = tabledr.nrows
    ma_num = tabledr.col_values(0)
    #print ma_num
    max_ma_num = max(ma_num)
    hr_num = tabledr.col_values(1)
    max_hr_num = max(hr_num)
    he_num = tabledr.col_values(2)
    max_he_num = max(he_num)
    ma_area = tabledr.col_values(3)
    max_ma_area = max(ma_area)
    hr_area = tabledr.col_values(4)
    max_hr_area = max(hr_area)
    he_area = tabledr.col_values(5)
    max_he_area = max(he_area)

    #print '********************************'
    train_x = []
    for i in range(traindata_num):
        #train_x.append([ma_num[i], hr_num[i], he_num[i], ma_area[i], hr_area[i], he_area[i]])
        a = tabledr.row_values(i)
        #a = [a[0]/max_ma_num, a[1]/max_hr_num, a[2]/max_he_num, a[3]/max_ma_area, a[4]/max_hr_area, a[5]/max_he_area]
        #a = [a[0]/max_ma_num, a[1]/max_hr_num, a[2]/max_he_num, a[3]/max_ma_area, a[4]/max_hr_area] #最好的 0，1 和 2，3，4 分类
        a = [a[4]/max_hr_area, a[5]/max_he_area]
        #print a
        #train_x.append(tabledr.row_values(i))
        train_x.append(a)
    train_x = np.array(train_x)
    #print train_x
    #print '********************************'
    test_x1 = []
    for j in range(traindata_num, nrowsdr):
        #test_x.append([ma_num[j], hr_num[j], he_num[j], ma_area[j], hr_area[j], he_area[j]])
        b = tabledr.row_values(j)
        #b = [b[0]/max_ma_num, b[1]/max_hr_num, b[2]/max_he_num, b[3]/max_ma_area, b[4]/max_hr_area, b[5]/max_he_area]
        #b = [b[0]/max_ma_num, b[1]/max_hr_num, b[2]/max_he_num, b[3]/max_ma_area, b[4]/max_hr_area] #最好的 0，1 和 2，3，4 分类
        b = [b[4]/max_hr_area, b[5]/max_he_area]
        #print b
        #test_x.append(tabledr.row_values(j))
        test_x.append(b)
    test_x = np.array(test_x)
    #print len(test_y)

    return test_x1

def read_IDRID_data(df):
    # df = pd.DataFrame(pd.read_csv(data_file, header=0))
    # #df.rename(columns={'Retinopathy grade': 'DRGrading'}, inplace=True)
    # tempma = df['MA'].apply(lambda x: (x) / (1.0 * max(df['MA'])))
    # df.drop('MA', axis=1)
    # df['MA'] = tempma
    # temphr = df['HR'].apply(lambda x: (x) / (1.0 * max(df['HR'])))
    # df.drop('HR', axis=1)
    # df['HR'] = temphr
    # temphe = df['HE'].apply(lambda x: (x) / (1.0 * max(df['HE'])))
    # df.drop('HE', axis=1)
    # df['HE'] = temphe
    # tempareama = df['areaMA'].apply(lambda x: (x) / (1.0 * max(df['areaMA'])))
    # df.drop('areaMA', axis=1)
    # df['areaMA'] = tempareama
    # tempareahr = df['areaHR'].apply(lambda x: (x) / (1.0 * max(df['areaHR'])))
    # df.drop('areaHR', axis=1)
    # df['areaHR'] = tempareahr
    # tempareahe = df['areaHE'].apply(lambda x: (x) / (1.0 * max(df['areaHE'])))
    # df.drop('areaHE', axis=1)
    # df['areaHE'] = tempareahe
    #print df.shape
    #de0 = df.sort_values(by=['Image No'], ascending=[True])
    #print de0
    df0 = df.loc[df['MA'].isin([0])]
    Grading0ImageName = df0['Image No']   # 获得该列表中的值为Grading 0级
    #print Grading0Image
    dex = df.drop(df0.index)   #dex是删除MA数量等于0的图像之后剩余的数据
    #print Grading0.index
    # print Grading0.index
    # print dex.shape
    dexx = dex.reset_index()
    #Grading01 = dexx.loc[dexx['HE'].isin([0])]
    #print Grading01.index
    #print dexx
    temp_a = dexx.ix[:,['MA', 'areaMA']]
    #test_x = np.array(temp_a)
    test_x = temp_a
    #print test_x
    return test_x, dexx, Grading0ImageName

if __name__ == '__main__':
    data_file = "/Users/victor/code4suixiaodan/Mask_RCNN-master/code4work/ImageSynthesis/4lesions/MESSIDOR/LesionNumMESSIDOR.csv"
    data_file1 = "/Users/victor/code4suixiaodan/Mask_RCNN-master/code4work/ImageSynthesis/4lesions/LesionNumNewHE0870.xlsx"
    groundtruth_file = "IDRiD_Training_Set.xlsx"

    df = pd.DataFrame(pd.read_csv(data_file, header=0))
    # df.rename(columns={'Retinopathy grade': 'DRGrading'}, inplace=True)
    tempma = df['MA'].apply(lambda x: (x) / (1.0 * max(df['MA'])))
    df.drop('MA', axis=1)
    df['MA'] = tempma
    temphr = df['HR'].apply(lambda x: (x) / (1.0 * max(df['HR'])))
    df.drop('HR', axis=1)
    df['HR'] = temphr
    temphe = df['HE'].apply(lambda x: (x) / (1.0 * max(df['HE'])))
    df.drop('HE', axis=1)
    df['HE'] = temphe
    tempareama = df['areaMA'].apply(lambda x: (x) / (1.0 * max(df['areaMA'])))
    df.drop('areaMA', axis=1)
    df['areaMA'] = tempareama
    tempareahr = df['areaHR'].apply(lambda x: (x) / (1.0 * max(df['areaHR'])))
    df.drop('areaHR', axis=1)
    df['areaHR'] = tempareahr
    tempareahe = df['areaHE'].apply(lambda x: (x) / (1.0 * max(df['areaHE'])))
    df.drop('areaHE', axis=1)
    df['areaHE'] = tempareahe



    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = [ 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    print 'reading training and testing data...'
    train_x, train_y, test_x, test_y = read_data1(data_file1, 0.25)
    test_X, dexx, Grading0ImageName = read_IDRID_data(df)

    model = classifiers['RF'](train_x, train_y)
    predict = model.predict(test_X)

    location01 = np.array(np.where(predict == 0)).astype(np.int64)
    #print pd.Index(location01[0])
    #print len(location01[0])

    df01 = dexx.loc[pd.Index(location01[0])]
    df01 = df01.reset_index()
    df234 = dexx.drop(pd.Index(location01[0]))
    df234 = df234.reset_index()
    # print df234
    train_x1, train_y1, test_x1, test_y1 = read_data2(data_file1, 0.25)
    #print train_y1
    #print test_y1
    test_X1 =  df01.ix[:,['MA', 'HR', 'areaMA']]
    #print test_X1
    model1 = classifiers['KNN'](train_x1, train_y1)
    predict1 = model1.predict(test_X1)
    # predicttemp = model1.predict(test_x1)
    # precision = metrics.precision_score(test_y1, predicttemp)
    # recall = metrics.recall_score(test_y1, predicttemp)
    # accuracy = metrics.accuracy_score(test_y1, predicttemp)
    # print precision, recall, accuracy

    #print predict1
    location0 = np.array(np.where(predict1 == 0)).astype(np.int64)
    df0 = df01.loc[pd.Index(location0[0])]
    Grading00ImageName = df0['Image No']
    Grading0ImageName = Grading0ImageName.append(Grading00ImageName)
    Grading0ImageName = Grading0ImageName.reset_index(drop=True)
    # print Grading0ImageName.shape
    df1 = df01.drop(pd.Index(location0[0]))
    #print df1
    Grading1ImageName = df1['Image No'].reset_index(drop=True)
    # print "Grading1ImageName: ", Grading1ImageName.shape

    train_x2, train_y2, test_x2, test_y2 = read_data3(data_file1, 0.3)
    test_X2 = df234.ix[:,['areaHR']]

    model2 = classifiers['KNN'](train_x2, train_y2)
    predict2 = model2.predict(test_X2)

    # print predict2
    location2 = np.array(np.where(predict2 == 0)).astype(np.int64)
    df2 = df234.loc[pd.Index(location2[0])]
    Grading2ImageName = df2['Image No'].reset_index(drop=True)
    # print Grading2ImageName.shape
    df34 = df234.drop(pd.Index(location2[0]))
    df34 = df34.reset_index(drop=True)
    # print dff34

    train_x3, train_y3, test_x3, test_y3 = read_data4(data_file1, 0.25)
    test_X3 = df34.ix[:, ['areaHR', 'areaHE']]

    model3 = classifiers['KNN'](train_x3, train_y3)
    predict3 = model3.predict(test_X3)

    # print predict2
    location3 = np.array(np.where(predict3 == 0)).astype(np.int64)
    df3 = df34.loc[pd.Index(location3[0])]
    Grading3ImageName = df3['Image No'].reset_index(drop=True)
    df4 = df34.drop(pd.Index(location3[0]))
    Grading4ImageName = df4['Image No'].reset_index(drop=True)

    # print Grading3ImageName.shape
    # print len(Grading4ImageName)
    # print Grading0ImageName

    Pointdata = [['Image No', 'Retinopathy grade']]

    for i in range(len(Grading0ImageName)):
        #print i, Grading0ImageName[i]
        Pointdata.append([Grading0ImageName[i], 0])

    for j in range(len(Grading1ImageName)):
        Pointdata.append([Grading1ImageName[j], 1])

    for k in range(len(Grading2ImageName)):
        Pointdata.append([Grading2ImageName[k], 2])

    for l in range(len(Grading3ImageName)):
        Pointdata.append([Grading3ImageName[l], 3])

    for m in range(len(Grading4ImageName)):
        #print 4, m, Grading4ImageName[m]
        Pointdata.append([Grading4ImageName[m], 4])
    #
    csvFile2 = open('/Users/victor/code4suixiaodan/Mask_RCNN-master/code4work/ImageSynthesis/4lesions/MESSIDOR/MESSIDOR_Disese Grading_DR.csv', 'w')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvFile2)
    m = len(Pointdata)
    for i in range(m):
        writer.writerow(Pointdata[i])
    csvFile2.close()

    print '******************* Ok ********************'
    # num_train, num_feat = train_x.shape
    # num_test, num_feat = test_x.shape
    # is_binary_class = (len(np.unique(train_y)) == 2)
    # print '******************** Data Info *********************'
    # print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)
    # Predict = []
    # Accuracy = []
    # for classifier in test_classifiers:
    #     print '******************* %s ********************' % classifier
    #     start_time = time.time()
    #     model = classifiers[classifier](train_x, train_y)
    #     print 'training took %fs!' % (time.time() - start_time)
    #     predict = model.predict(test_x)
    #     Predict.append(predict)
    #     #print len(predict)
    #     if model_save_file != None:
    #         model_save[classifier] = model
    #     if is_binary_class:
    #         precision = metrics.precision_score(test_y, predict)
    #         recall = metrics.recall_score(test_y, predict)
    #         print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
    #     accuracy = metrics.accuracy_score(test_y, predict)
    #     Accuracy.append(accuracy)
    #     print 'accuracy: %.2f%%' % (100 * accuracy)
    #
    # #print Predict
    # #print Accuracy
    # # print model_save_file
    # if model_save_file != None:
    #     pickle.dump(model_save, open(model_save_file, 'wb'))