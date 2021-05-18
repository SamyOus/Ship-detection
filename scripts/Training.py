import json
import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from tensorflow.python.keras.metrics import Metric
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D,Flatten,
                                    Dense, Dropout, Activation)

if __name__ == '__main__':

    #Metrics selection

    METRICS = [
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'), 
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
          keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    #Large number of epochs due to early stopping

    EPOCHS = 100

    #early stoppnig to avoid overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)


    #Select the initial bias to accelerate the speed of laerning 
    neg, pos = 3000, 1000
    initial_bias = np.log([pos/neg])
    output_bias = tf.keras.initializers.Constant(initial_bias)


    #Build the model
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(80, 80, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(32, (3, 3), input_shape=(80, 80, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax',bias_initializer=output_bias))

    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=METRICS)

    #Train the model
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        callbacks=[early_stopping],
                        validation_data=(X_val, y_val))


    #Save the model
    import time;
    ts = time.time()
    ts = str(int(ts))

    model.save('./models/CNN_'+ts+'.h5')
