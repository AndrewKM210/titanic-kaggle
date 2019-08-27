#!/usr/bin/env python
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import model_from_json

from data_reader import DataReader
from graphics import Graphics
import numpy as np
import pandas as pd
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

__author__ = 'Andrew Mackay'


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


def save_model(_model):

    model_json = _model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("keras_models/model.h5")
    print("Saved model to disk")


def load_model():

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("keras_models/model.h5")
    print("Loaded model from disk")
    return loaded_model


def output_predictions(_predictions):
    _predictions = pd.DataFrame(_predictions)
    with open('output/predictions.csv', 'w') as f:
        f.write('PassengerId,Survived\n')

    _predictions.to_csv('output/predictions.csv', index=False, header=False, mode='a')


reader = DataReader('data', 'train.csv', 'test.csv')
x, y, x_val, ids = reader.obtain_data()

# Standarize training and test set
# scaler = RobustScaler()
# scaler.fit(x)
# x_train = scaler.transform(x)
# x_val = scaler.transform(x_val)

# Separate train and test from the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_val = pd.DataFrame(x_val, columns=x_train.columns)


# Choose which model to apply
print('1-XGBoost')
print('2-Keras kmeans')
print('3-Keras')
# algorithm = input("Enter an algorithm: ")
algorithm = '1'

if algorithm == '1':

    # Establish train and test sets
    eval_set = [(x_train, y_train), (x_test, y_test)]

    # Establish evaluation metrics
    eval_metric = ["error", "logloss"]

    # Create the XGBClassifier, with 100 iterations, maximum depth of 6 and learning rate of 0.001
    model = XGBClassifier(silent=False, n_estimators=1000, objective='binary:logistic', learning_rate=0.001, max_depth=10)

    # Train the model
    model.fit(x_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

    # Once the model is trained, we can make a prediction
    # Start with the test set, so we can evaluate the model
    pred = model.predict_proba(x_test)

    # Obtain the predictions of survival (not the probabilty of no survival)
    pred = pred[:, 1]

    # Round the predictions, so if the probability of survivability is greater then 0.5 then it predicts that the
    # person survives
    pred = pred.round()

    # Obtain the metrics of the training
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # Plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')

    # Plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')

    # Load the graphics class created
    graphics = Graphics()
    graphics.load_data(model, None, pred, y_test.values)

    # Plot a confusion matrix
    graphics.confusion_matrix()

    # Plot a feature importance graph
    graphics.plot_feature_importance()

    # Now, with the val set we can output to get results
    pred = model.predict_proba(x_val)[:, 1].round()

    # For the submission, the ids must appear next to the prediction

    # Create a DataFrame out of the two ndarrays
    final_prediction = pd.DataFrame({'id': ids.transpose(), 'survives': pred.transpose()})

    # Change the data to integers
    final_prediction['id'] = final_prediction['id'].astype(int)
    final_prediction['survives'] = final_prediction['survives'].astype(int)

    # Output the final submission
    output_predictions(final_prediction)

    # Show al the graphics
    graphics.show()

elif algorithm == '2':

    # Create random seed
    seed = 5
    np.random.seed(seed)

    # Create a keras classifier with a model returned by the create_baseline function
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=1)

    # Apply kfold with the keras classifier
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    # Obtain and print the results obtained
    results = cross_val_score(estimator, x_train, y_train, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

elif algorithm == '3':

    print('1- Load previous model')
    print('2- Train model')
    option = input('Choose option')

    if option == '2':
        # Create de model
        model = Sequential()

        # Add layers to the model
        model.add(Dense(64, input_dim=7, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model with an Adam optimizar and a learning rate of 0.02
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001))
        checkpoint = ModelCheckpoint('keras_models/weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

        # Train the model
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=4, epochs=100, callbacks=[checkpoint])
        model.load_weights('keras_models/weights.hdf5')
        save_model(model)

    else:
        model = load_model()

    # Make predictions
    predictions = model.predict(x_test)
    predictions = predictions.round()

    # Load prediction data and history to plot interesting graphs
    graphics = Graphics()
    # graphics.load_data(history, predictions, y_test)

    # Plot the evolution of cost during the training
    # graphics.plot_loss()

    # Plot a confusion matrix
    # graphics.confusion_matrix()

    predictions = model.predict(x_val)
    predictions = predictions.round()
    predictions = pd.DataFrame(predictions)
    ids = pd.DataFrame(ids)
    ids = ids.join(predictions).head()
    output_predictions(ids)
