import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from xgboost import plot_importance


class Graphics:
    """
    Class used for plotting useful graphics to analize results
    """

    def __init__(self):
        self.model = None
        self.history = None
        self.predictions = None
        self.groundtruth = None

    def load_data(self, model, history, predictions, groundtruth):
        self.model = model
        self.history = history
        self.predictions = predictions
        self.groundtruth = groundtruth

    def plot_loss(self):
        """
        Plots the evolution of training loss during training of the model
        """

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

    def confusion_matrix(self):
        """
        Plot a confusion matrix with the obtained predictions
        """

        # TODO: change zoom
        cm = confusion_matrix(self.groundtruth, self.predictions)
        plt.figure(figsize=(10, 16))
        sns.heatmap(cm, annot=True, square=True)

    def plot_feature_importance(self):
        """
        Plot a graph showing the importance given to each feature
        """

        plot_importance(self.model)

    def show(self):
        """
        Show the graphics plotted
        :return:
        """
        plt.show()
