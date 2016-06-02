import matplotlib.pyplot as plt


class Visualisation:

    @staticmethod
    def plot2d(data, row_name):
        """
        Plots one feature in 2D, time on x axis.
        :param data:
        :param row_name:
        """
        plt.plot(data.all_data[row_name+'1'])
        plt.show()

    @staticmethod
    def plot_all_2d(data):
        """
        Plots all features in 2D on one graph, time on x axis.
        :param data:
        """
        for feature in data.feature_names:
            plt.plot(data.all_data[feature+'1'])
        plt.show()
