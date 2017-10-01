# -*- coding: utf-8 -*-
# -*- coding:UTF-8 -*-
import time    
from sklearn import metrics    
import pickle as pickle    
import pandas as pd  
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import ShuffleSplit

  
    
# Multinomial Naive Bayes Classifier    
def naive_bayes_classifier(train_x, train_y):    
    from sklearn.naive_bayes import BernoulliNB  
    cv = ShuffleSplit(int(len(train_x)), n_iter=10, random_state=0, test_size=0.2)
    model = GridSearchCV(estimator=BernoulliNB(), cv=cv, param_grid=dict(alpha=[0.01,0.1,1]))
    model.fit(train_x, train_y)    
    return model    
    
    
# KNN Classifier    
def knn_classifier(train_x, train_y):    
    from sklearn.neighbors import KNeighborsClassifier    
    model = KNeighborsClassifier(n_neighbors=6)    
    model.fit(train_x, train_y)    
    return model    
    
    
# Logistic Regression Classifier    
def logistic_regression_classifier(train_x, train_y):    
    from sklearn.linear_model import LogisticRegression  
    cv = ShuffleSplit(int(len(train_x)), n_iter=10, random_state=0, test_size=0.2)
    param_grid = {'intercept_scaling':list([1,2,3]),'C':list(range(1,20))}
    model = GridSearchCV(estimator=LogisticRegression(), cv=cv, param_grid=param_grid)
    model.fit(train_x, train_y)    
    return model    
    
    
# Random Forest Classifier    
def random_forest_classifier(train_x, train_y):    
    from sklearn.ensemble import RandomForestClassifier    
    model = RandomForestClassifier(n_estimators=500, criterion='entropy', max_features=8, max_depth=25,
                                    min_samples_leaf = 1, bootstrap=True, oob_score=True, n_jobs=30, random_state=0)
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
    model = GradientBoostingClassifier(learning_rate=0.05,max_depth=15,min_samples_leaf=2,min_samples_split=0.5,n_estimators=500)    
    model.fit(train_x, train_y)    
    return model    
    
    
# SVM Classifier    
def svm_classifier(train_x, train_y):    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', C=10,gamma=0.01,probability=True)    
    model.fit(train_x, train_y)    
    return model    
    
# SVM Classifier using cross validation    
def svm_cross_validation(train_x, train_y):    
    from sklearn.grid_search import GridSearchCV    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [ 1, 10, 20], 'gamma': [0.001, 0.1,0.01]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model    
    
        
if __name__ == '__main__':    
    print('reading training and testing data...')     
    train = pd.read_csv('../datasets/dataset_iloc/train.csv')
    test = pd.read_csv('../datasets/dataset_iloc/test.csv')
    train.loc[train["class"]=="chloroplast","class"]=0
    train.loc[train["class"]=="cytoplasmic","class"]=1
    train.loc[train["class"]=="ER","class"]=2
    train.loc[train["class"]=="extracellular","class"]=3
    train.loc[train["class"]=="Golgi","class"]=4
    train.loc[train["class"]=="lysosomal","class"]=5
    train.loc[train["class"]=="mitochondrial","class"]=6
    train.loc[train["class"]=="nuclear","class"]=7
    train.loc[train["class"]=="peroxisomal","class"]=8
    train.loc[train["class"]=="plasma_membrane","class"]=9
    train.loc[train["class"]=="vacuolar","class"]=10
    
    test.loc[test["class"]=="chloroplast","class"]=0
    test.loc[test["class"]=="cytoplasmic","class"]=1
    test.loc[test["class"]=="ER","class"]=2
    test.loc[test["class"]=="extracellular","class"]=3
    test.loc[test["class"]=="Golgi","class"]=4
    test.loc[test["class"]=="lysosomal","class"]=5
    test.loc[test["class"]=="mitochondrial","class"]=6
    test.loc[test["class"]=="nuclear","class"]=7
    test.loc[test["class"]=="peroxisomal","class"]=8
    test.loc[test["class"]=="plasma_membrane","class"]=9
    test.loc[test["class"]=="vacuolar","class"]=10
    
    y_train = train['class']
    names_train = train['name']
    X_train = train.drop(['class', 'name', 'sequence'], axis=1)
    
    y_test = test['class']
    names_test = test['name']
    X_test = test.drop(['class', 'name', 'sequence'], axis=1)    
    std_scaler = StandardScaler().fit(X_train)
    # transform train and test set using standardization
    train_x = std_scaler.transform(X_train)
    test_x = std_scaler.transform(X_test)
    train_y = np.asarray(y_train, dtype='float64')
    test_y = np.asarray(y_test, dtype='float64')
    
    thresh = 0.5    
    model_save_file = None    
    model_save = {}    
     
    test_classifiers = ['SVM', 'SVMCV',"NB",'KNN', 'LR', 'RF', 'DT', 'GBDT']    
    classifiers = {
                  'SVM':svm_classifier, 
                'SVMCV':svm_cross_validation, 
                   "NB":naive_bayes_classifier,
                  'KNN':knn_classifier,    
                   'LR':logistic_regression_classifier,    
                   'RF':random_forest_classifier,    
                   'DT':decision_tree_classifier,                        
                 'GBDT':gradient_boosting_classifier    
    }    
        
        
    for classifier in test_classifiers:    
        print('******************* %s ********************' % classifier)    
        start_time = time.time()    
        model = classifiers[classifier](train_x, train_y)    
        print('training took %fs!' % (time.time() - start_time))    
        predict = model.predict(test_x)    
        if model_save_file != None:    
            model_save[classifier] = model    
        precision = metrics.precision_score(test_y, predict,average="weighted")    
        recall = metrics.recall_score(test_y, predict,average="weighted")    
        MCC = metrics.matthews_corrcoef(test_y, predict)
        print('precision: %.2f%%, recall: %.2f%%,MCC:%.2f'% (100 * precision, 100 * recall,MCC))    
        result = metrics.classification_report(test_y, predict)
        confusion_m = metrics.confusion_matrix(test_y, predict)
        print(result)
        print(confusion_m)
        accuracy = metrics.accuracy_score(test_y, predict)    
        print('accuracy: %.2f%%' % (100 * accuracy))     
    
    if model_save_file != None:    
        pickle.dump(model_save, open(model_save_file, 'wb'))     
