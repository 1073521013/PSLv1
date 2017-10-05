# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import numpy as np
import os.path as osp
from keras.datasets import mnist
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ds_base import ds_base

class MNIST(ds_base):
    def __init__(self, **kwargs):
        super(MNIST, self).__init__(**kwargs)
        # data_path = osp.abspath( osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir,
        #     'datasets/mnist/keras/mnist.pkl.gz') )
        # with gzip.open(data_path, 'rb') as f:
        #     (X_train, y_train), (X_test, y_test) = pickle.load(f)
        #
	#mnist = fetch_mldata('MNIST original', data_home='~/scikit-learn-datasets')
	#X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)
	#X_train = X_train.reshape((len(X_train), 28, 28))
	#X_test = X_test.reshape((len(X_test), 28, 28))
	train = pd.read_csv('dataset/train.csv')
	test = pd.read_csv('dataset/test.csv')
	train.loc[train["class"]=="Cell.membrane","class"]=0
	train.loc[train["class"]=="Cytoplasm","class"]=1
	train.loc[train["class"]=="Endoplasmic.reticulum","class"]=2
	train.loc[train["class"]=="Golgi.apparatus","class"]=3
	train.loc[train["class"]=="Lysosome/Vacuole","class"]=4
	train.loc[train["class"]=="Mitochondrion","class"]=5
	train.loc[train["class"]=="Nucleus","class"]=6
	train.loc[train["class"]=="Peroxisome","class"]=7
	train.loc[train["class"]=="Plastid","class"]=8
	train.loc[train["class"]=="Extracellular","class"]=9

	test.loc[test["class"]=="Cell.membrane","class"]=0
	test.loc[test["class"]=="Cytoplasm","class"]=1
	test.loc[test["class"]=="Endoplasmic.reticulum","class"]=2
	test.loc[test["class"]=="Golgi.apparatus","class"]=3
	test.loc[test["class"]=="Lysosome/Vacuole","class"]=4
	test.loc[test["class"]=="Mitochondrion","class"]=5
	test.loc[test["class"]=="Nucleus","class"]=6
	test.loc[test["class"]=="Peroxisome","class"]=7
	test.loc[test["class"]=="Plastid","class"]=8
	test.loc[test["class"]=="Extracellular","class"]=9

	# Split into labels, names and data
	y_train = train['class']
	names_train = train['name']
	X_train = train.drop(['class', 'name', 'sequence'], axis=1)

	y_test = test['class']
	names_test = test['name']
	X_test = test.drop(['class', 'name', 'sequence'], axis=1)
        std_scaler = StandardScaler()
	X_train = std_scaler.fit_transform(X_train)
	X_test = std_scaler.fit_transform(X_test)
	from sklearn.feature_selection import SelectKBest , f_classif

	X_train = SelectKBest(f_classif,k=64).fit_transform(X_train,y_train)
	X_test = SelectKBest(f_classif,k=64).fit_transform(X_test,y_test)

	y_train = np.asarray(y_train, dtype='int')
	y_test = np.asarray(y_test, dtype='int')
	#(X_train, y_train), (X_test, y_test) = mnist.load_data()
        if self.data_set == 'train':
            X = X_train
            y = y_train
        elif self.data_set == 'train-small':
            X = X_train[:2000]
            y = y_train[:2000]
        elif self.data_set == 'test':
            X = X_test
            y = y_test
        elif self.data_set == 'test-small':
            X = X_test[:1000]
            y = y_test[:1000]
        elif self.data_set == 'all':
            X = np.vstack((X_train, X_test))
            y = np.vstack((y_train, y_test))
        else:
            raise ValueError('MNIST Unsupported data_set: ', self.data_set)

        # normalization
        if self.norm:
            X = X.astype(np.float32) / 255
        X = X[:,np.newaxis,np.newaxis,:]
        X=X.reshape(len(X),1,8,8)
	X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y

