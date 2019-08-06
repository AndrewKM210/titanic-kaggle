#!/usr/bin/env python
from keras.callbacks import ModelCheckpoint
from data_reader import DataReader
from graphics import Graphics
import numpy
import pandas as pd
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

__author__ = "Andrew Mackay"

def create_baseline():
    """
    Creates and returns a keras neural network model
    :return: keras neural network model
    """

    # Create the model
    m = Sequential()

    # Add layers
    m.add(Dense(60, input_dim=7, kernel_initializer='normal', activation='relu'))
    m.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile the model
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


reader = DataReader('train.csv', 'test.csv')
x, y, x_val = reader.obtain_data()

# Standarize training and test set
scaler = RobustScaler()
scaler.fit(x)
x_train = scaler.transform(x)
x_val = scaler.transform(x_val)

# Separate train and test from the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_val = pd.DataFrame(x_val, columns=x_train.columns)

# Choose which model to apply
print('1-XGBoost')
print('2-Keras kmeans')
print('3-Keras')
# algorithm = input("Enter an algorithm: ")
algorithm = '3'

if algorithm == '1':

    # Establish train and test sets
    eval_set = [(x_train, y_train), (x_test, y_test)]

    # Establish evaluation metrics
    eval_metric = ["auc", "error"]

    # Create the XGBClassifier, with 100 iterations, maximum depth of 6 and learning rate of 0.001
    model = XGBClassifier(silent=False, n_estimators=100, learning_rate=0.001, max_depth=6)

    # Train the model
    model.fit(x_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

    # Once the model is trained, we can make a prediction
    pred = model.predict_proba(x_val)

    # Obtain the metrics of the training
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # Draw the evolution of cost and accuray values during training
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()

if algorithm == '2':

    # Create random seed
    seed = 5
    numpy.random.seed(seed)

    # Create a keras classifier with a model returned by the create_baseline function
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=1)

    # Apply kfold with the keras classifier
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    # Obtain and print the results obtained
    results = cross_val_score(estimator, x_train, y_train, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

if algorithm == '3':

    # Create de model
    model = Sequential()

    # Add layers to the model
    model.add(Dense(64, input_dim=7, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    # Compile the model with an Adam optimizar and a learning rate of 0.02
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001))
    checkpoint = ModelCheckpoint("weights.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=10, callbacks=[checkpoint])
    model.load_weights("weights.hdf5")
    predictions = model.predict(x_test)
    predictions = predictions.round()
    # Plot the evolution of cost during the training
    # plot_loss(history)
    graphics = Graphics()
    graphics.load_data(history, predictions, y_test)
    graphics.confusion_matrix()