#!/usr/bin/env python
__author__ = "Andrew Mackay"

import numpy
import pandas as pd
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier


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


def plot_loss(h):
    """
    Plots the evolution of training loss during training of the model
    :param h: history of training loss
    """

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Obtain the data from the train.csv file
train = pd.read_csv('train.csv')

# Drop the name, passenger id, cabin and ticket columns, since they are not relevant for training
train = train.drop(['Name'], axis=1)
train = train.drop(['PassengerId'], axis=1)
train = train.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)

# Convert the sex and embarked columns in to integers
train['Sex'] = train['Sex'].astype('category').cat.codes
train['Embarked'] = train['Embarked'].astype('category').cat.codes

# Drop all remaining null rows
train = train.dropna()

# Separate x and y in the data set
x = train.drop(['Survived'], axis=1)
y = train['Survived']
y = to_categorical(y)

# Apply the same for the test.csv set (validation data)
x_val = pd.read_csv('test.csv')
x_val = x_val.drop(['Name'], axis=1)
x_val = x_val.drop(['PassengerId'], axis=1)
x_val = x_val.drop(['Cabin'], axis=1)
x_val = x_val.drop(['Ticket'], axis=1)
x_val['Sex'] = x_val['Sex'].astype('category').cat.codes
x_val['Embarked'] = x_val['Embarked'].astype('category').cat.codes
x_val = x_val.dropna()

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
algorithm = input("Enter an algorithm: ")

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
    model.add(Dense(300, init='uniform', input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(1, init='uniform', activation='softmax'))

    # Compile the model with an Adam optimizar and a learning rate of 0.02
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.02))

    # Train the model
    history = model.fit(x_train, y_train, batch_size=32, epochs=10)

    # Plot the evolution of cost during the training
    plot_loss(history)
