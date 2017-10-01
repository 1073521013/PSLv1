# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
# load dataset
train = pd.read_csv('../datasets/dataset_iloc/train.csv')
test = pd.read_csv('../datasets/dataset_iloc/test.csv')
y_train = train['class']
names_train = train['name']
x_train = train.drop(['class', 'name', 'sequence'], axis=1)

y_test = test['class']
names_test = test['name']
x_test = test.drop(['class', 'name', 'sequence'], axis=1)
## encode class values as integers
encoder = LabelEncoder()
encoded_y_train = encoder.fit_transform(y_train)
encoded_y_test = encoder.fit_transform(y_test)
## convert integers to dummy variables (one hot encoding)
#Y_train = np_utils.to_categorical(encoded_y_train)
#Y_test = np_utils.to_categorical(encoded_y_test)
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(x_train)
X_test = std_scaler.fit_transform(x_test)
# define model structure
def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim=500, input_dim=72, init="normal",activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(500, init='uniform', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=11, activation='softmax'))
    # Compile model
    #sgd=optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=20, batch_size=10)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
estimator.fit(X_train, encoded_y_train)

# make predictions

pred = estimator.predict(X_test)
print(accuracy_score(encoded_y_test,pred))
print(classification_report(encoded_y_test,pred))

cf_val=confusion_matrix(np.array(encoded_y_test), np.array(pred))
import matplotlib.pyplot as plt
# Plot confusion matrix 
plt.figure(figsize=(8,8))
cmap=plt.cm.Blues   
plt.imshow(cf_val, interpolation='nearest', cmap=cmap)
plt.title('Confusion matrix validation set')
plt.colorbar()
tick_marks = np.arange(11)
classes =['cytoplasmic', 'extracellular', 'nuclear', 'plasma_membrane',"ER","lysosomal",                  
          "Golgi","chloroplast","peroxisomal","mitochondrial","vacuolar"]

plt.xticks(tick_marks, classes, rotation=60)
plt.yticks(tick_marks, classes)

thresh = cf_val.max() / 2.
for i, j in itertools.product(range(cf_val.shape[0]), range(cf_val.shape[1])):
    plt.text(j, i, cf_val[i, j],
             horizontalalignment="center",
             color="white" if cf_val[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True location')
plt.xlabel('Predicted location');