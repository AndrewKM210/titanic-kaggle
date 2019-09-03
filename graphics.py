import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from xgboost import plot_importance


class Graphics:
    """
    Class used for plotting useful graphics to analize results
    """

    def __init__(self):
        self.predictions = None
        self.groundtruth = None

    def load_data(self, predictions, groundtruth):
        self.predictions = predictions
        self.groundtruth = groundtruth

    def keras_plot_loss(self, history):
        """
        Plots the evolution of training loss during training of the model
        """

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

    def confusion_matrix(self):
        """
        Plot a confusion matrix with the obtained predictions
        """

        cm = confusion_matrix(self.groundtruth, self.predictions)
        plt.figure(figsize=(10, 16))
        sns.heatmap(cm, annot=True, square=True)

    def plot_feature_importance(self, model):
        """
        Plot a graph showing the importance given to each feature
        """

        plot_importance(model)


    def xgboost_plot_log_loss(self, x_axis, results):
        """
        Plots the evolution of the log loss of the training of an xgboost model
        :param x_axis: x axis
        :param results: results
        """
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
        ax.legend()
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')


    def xgboost_plot_classification_error(self, x_axis, results):
        """
        Plots the evolution of the classification error of the training of an xgboost model
        :param x_axis: x axis
        :param results: results
        """
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['error'], label='Train')
        ax.plot(x_axis, results['validation_1']['error'], label='Test')
        ax.legend()
        plt.ylabel('Classification Error')
        plt.title('XGBoost Classification Error')


    def show(self):
        """
        Show the graphics plotted
        :return:
        """
        plt.show()
