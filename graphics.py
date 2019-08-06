import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class Graphics:
    """
    Class used for plotting useful graphics to analize results
    """

    def __init__(self):
        self.history = None
        self.predictions = None
        self.groundtruth = None

    def load_data(self, history, predictions, groundtruth):
        self.history = history
        self.predictions = predictions
        self.groundtruth = groundtruth

    def plot_loss(self):
        """
        Plots the evolution of training loss during training of the model
        :param h: history of training loss
        """

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def confusion_matrix(self):
        """
        Plot a confusion matrix with the obtained predictions
        :param predictions: values predicted by model
        :param groundtruth: grountruth of prediction
        """

        # TODO: change zoom
        cm = confusion_matrix(self.groundtruth, self.predictions)
        plt.figure(figsize=(10, 16))
        sns.heatmap(cm, annot=True, square=True)
        plt.show()
