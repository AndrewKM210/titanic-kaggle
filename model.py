#!/usr/bin/env python
from data_reader import DataReader
from graphics import Graphics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import model_from_json
from keras.optimizers import RMSprop
import pandas as pd

__author__ = 'Andrew Mackay'


def save_model(_model):
    """
    Saves a keras model and its respective weights in the model.json and model.h5 files
    :param _model: keras model to save
    """

    # Transform the model to json
    model_json = _model.to_json()

    # Create the json file
    with open("keras_models/model.json", "w") as json_file:
        json_file.write(model_json)

    # Save the weights in to the model.h5 file
    model.save_weights("keras_models/model.h5")
    print("Saved model to disk")


def load_model():
    """
    Returns a model loaded from the model.json and model.h5 files
    :return: model loaded from the model.json and model.h5 files
    """

    # Load model from json file
    json_file = open('keras_models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # Create model object from loaded json file
    loaded_model = model_from_json(loaded_model_json)

    # Load weights to the loaded model
    loaded_model.load_weights("keras_models/model.h5")
    print("Loaded model from disk")

    return loaded_model


def output_predictions(_predictions, name):
    """
    Outputs the predictions _predictions in to a file called "name".csv
    :param _predictions: predictions to output (pandas DataFrame)
    :param name: name of the file
    """

    # Writes the first line (header)
    with open('output/' + name + '.csv', 'w') as f:
        f.write('PassengerId,Survived\n')

    # Writes the predictions
    _predictions.to_csv('output/' + name + '.csv', index=False, header=False, mode='a')


# Start of main program, loads data from the DataReader class
reader = DataReader('data', 'train.csv', 'test.csv')
x, y, x_val, ids = reader.obtain_data()

# Separate train and test from the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_val = pd.DataFrame(x_val, columns=x_train.columns)

# Choose which model to apply
print('1-XGBoost')
print('2-Keras')
print('3-Random Forest')

algorithm = input("Enter an algorithm: ")

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

    # Load the graphics class created
    graphics = Graphics()
    graphics.load_data(pred, y_test.values)

    # Plot the log loss and classification error of the training
    graphics.xgboost_plot_classification_error(x_axis, results)
    graphics.xgboost_plot_log_loss(x_axis, results)

    # Plot a confusion matrix
    graphics.confusion_matrix()

    # Plot a feature importance graph
    graphics.plot_feature_importance(model)

    # Now, with the val set we can output to get results
    pred = model.predict_proba(x_val)[:, 1].round()

    # For the submission, the ids must appear next to the prediction
    # Create a DataFrame out of the two ndarrays
    final_prediction = pd.DataFrame({'id': ids.transpose(), 'survives': pred.transpose()})

    # Change the data to integers
    final_prediction['id'] = final_prediction['id'].astype(int)
    final_prediction['survives'] = final_prediction['survives'].astype(int)

    # Output the final submission
    output_predictions(final_prediction, 'xgboost')

    # Show al the graphics
    graphics.show()

elif algorithm == '2':

    print('1- Load previous model')
    print('2- Train model')
    option = input('Choose option: ')

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

    if option == "2":
        graphics.load_data(predictions, y_test)

        # Plot the evolution of cost during the training
        graphics.keras_plot_loss(history)

    else:
        graphics.load_data(predictions, y_test)

    # Plot a confusion matrix
    graphics.confusion_matrix()

    predictions = model.predict(x_val)
    predictions = predictions.round()

    # Create a DataFrame out of the two ndarrays
    final_prediction = pd.DataFrame({'id': ids.transpose(), 'survives': predictions[:, 0].transpose()})

    # Change the data to integers
    final_prediction['id'] = final_prediction['id'].astype(int)
    final_prediction['survives'] = final_prediction['survives'].astype(int)

    output_predictions(final_prediction, 'keras')

    graphics.show()

elif algorithm == "3":

    # Create the classifier
    clf = RandomForestClassifier()

    # Train the model
    model = clf.fit(x_train, y_train)

    # Make predictions
    predictions = model.predict(x_test)

    # Plot confussion matrix
    graphics = Graphics()
    graphics.load_data(predictions, y_test)
    graphics.confusion_matrix()

    # Make prediction with validation data
    predictions = model.predict(x_val)

    # Create a DataFrame out of the two ndarrays
    final_prediction = pd.DataFrame({'id': ids.transpose(), 'survives': predictions.transpose()})

    # Change the data to integers
    final_prediction['id'] = final_prediction['id'].astype(int)
    final_prediction['survives'] = final_prediction['survives'].astype(int)

    output_predictions(final_prediction, 'randomForest')

    # Show the graphics
    graphics.show()
