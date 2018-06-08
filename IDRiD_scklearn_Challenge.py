# -*- coding: utf-8 -*-

import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle
import xlrd

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
    #print '********************************'
    test_x = []
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

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    data_file = "/Users/victor/code4suixiaodan/Mask_RCNN-master/code4work/ImageSynthesis/4lesions/LesionNumNewHE0870.xlsx"
    groundtruth_file = "IDRiD_Training_Set.xlsx"
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
    train_x, train_y, test_x, test_y = read_data(data_file,groundtruth_file, 0.25)
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    is_binary_class = (len(np.unique(train_y)) == 2)
    print '******************** Data Info *********************'
    print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)
    Predict = []
    Accuracy = []
    for classifier in test_classifiers:
        print '******************* %s ********************' % classifier
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print 'training took %fs!' % (time.time() - start_time)
        predict = model.predict(test_x)
        Predict.append(predict)
        #print len(predict)
        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
        accuracy = metrics.accuracy_score(test_y, predict)
        Accuracy.append(accuracy)
        print 'accuracy: %.2f%%' % (100 * accuracy)

    #print Predict
    #print Accuracy
    # print model_save_file
    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))