import matplotlib.pyplot as plt
import pandas as pd
import operator

import src.constants as const


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

    @staticmethod
    def create_feature_map(features):
        outfile = open(const.graphs_path + 'xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i += 1

        outfile.close()

    @staticmethod
    def feature_importance(model):
        importance = model.get_fscore(fmap=const.graphs_path+'xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))

        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()

        plt.figure()
        df.plot()
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        plt.gcf().savefig(const.graphs_path + 'feature_importance_xgb.png')
