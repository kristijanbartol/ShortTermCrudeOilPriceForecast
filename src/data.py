import pandas as pd
import numpy as np

from src.preprocess import Parser
import src.constants as const

# Default values
time_lag = 15
data_size = 600
test_ratio = 0.8


class Data:
    """
    Represents data container. Original data is available in non-lagged ndarray form.
    All data is parsed from input files and then lagged for time_lag.
    Lagged features are concatenated into a single data structure (data frame).
    Feature names are organized as 'FeatureK', where K is the index
    of lagged vector for that feature.
    """

    def __init__(self, input_files, params=(time_lag, data_size, test_ratio), fill_strategy='default'):
        """
        In this constructor, data is parsed
        and all available data is saved into data frame.
        Train/test set is randomly chosen.
        :param input_files:
        :param params:
        :param fill_strategy:
        """
        self._p = Parser(fill_strategy)

        self.time_lag = params[0]
        self.data_size = params[1]
        self.test_ratio = params[2]

        self.num_of_features = len(input_files)

        self.original_feature_names = [file.replace('.csv', '') for file in input_files]
        self.original_vectors = np.array([self._p.parse_file(file) for file in input_files])
        self.lagged_feature_names, self.all_data = self.make_data_frame(self.original_vectors)

        self.train_data, self.test_data = self.__split_data()

    def __save_to_csv(self, train, test):
        pd.DataFrame.to_csv(train, const.exports_path + 'train_' + str(self.time_lag) + '_' +
                                str(round(self.data_size * self.test_ratio)))
        pd.DataFrame.to_csv(test, const.exports_path + 'test_' + str(self.time_lag) + '_' +
                                str(round(self.data_size * (1 - self.test_ratio))))

    def __split_data(self):
        """
        Takes data_size rows and splits it into train/test set.
        """
        subset = self.all_data[:self.data_size]
        train = subset.sample(frac=self.test_ratio, random_state=200)
        test = subset.drop(train.index)
        self.__save_to_csv(train, test)

        return train, test

    def __extract_feature_names(self):
        """
        Generates feature names dinamically depending on number of input features (files in data/).
        """
        lagged_names_with_index = ['output']
        for names in self.original_feature_names:
            for i in range(self.time_lag, 0, -1):
                lagged_names_with_index.append(names + str(i))
        return lagged_names_with_index

    def __delay_feature(self, input_feature):
        """
        Lags all features for some time_lag and prepends output column to the beginning of data frame.
        :param input_feature:
        :return:
        """
        # TODO - what if there is > 1 feature (output?)
        # use original vectors -> make_data_frame()
        data_matrix = np.zeros((self.time_lag+1, self.data_size))
        for i in range(0, self.time_lag+1):
            for j in range(0, self.data_size):
                data_matrix[i][j] = input_feature[j + i]
        return data_matrix

    def make_data_frame(self, data):
        """
        Converts original ndarray data to data frame form.
        :param data:
        :return:
        """
        # Reduce to equal lengths
        data = [row[0:min([len(x) for x in data])] for row in data]
        feature_names_with_index = self.__extract_feature_names()
        data_dict = dict()
        index = 0
        for feature in data:
            lagged_data = self.__delay_feature(feature)
            for row in lagged_data:
                data_dict[feature_names_with_index[index]] = row
                index += 1

        return feature_names_with_index[1:], pd.DataFrame(data_dict)

    def update_parameters(self, params):
        """
        Method can be used to change data parameters to see their influence on the result.
        :param params: -> in tuple form
        """
        self.time_lag = params[0]
        self.data_size = params[1]
        self.test_ratio = params[2]

        self.__split_data()
