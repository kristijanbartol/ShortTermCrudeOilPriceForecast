import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

from src import constants as const
from src.preprocess import Parser

# Default values
time_lag = 15
data_size = 600
test_ratio = 0.2


class Data:
    """
    Represents data frame. Data is parsed from input files.
    Each feature is delayed (time lagged) for some time_lag.
    Lagged features are concatenated into a single data structure.
    Feature names are organized as 'FeatureK', where K is the index
    of lagged vector for that feature.
    """

    def __init__(self, input_files, params=(time_lag, data_size, test_ratio), fill_strategy='default'):
        self._p = Parser(fill_strategy)

        self.time_lag = params[0]
        self.data_size = params[1]
        self.test_ratio = params[2]

        self.num_of_features = len(input_files)

        self.feature_names = [file.replace('.csv', '') for file in input_files]
        self.original_vectors = [self._p.parse_file(file) for file in input_files]
        self.all_data = self.make_data_frame(self.original_vectors)

        self.train_data, self.train_output, self.test_data, self.test_output = self.__split_data()

    def __split_data(self):
        subset = self.all_data[:self.data_size]
        y = subset.pop('output')
        x = subset
        return train_test_split(x.index, y, test_size=self.test_ratio, random_state=42)

    def __extract_feature_names(self):
        feature_names_with_index = ['output']
        for names in self.feature_names:
            for i in range(self.time_lag, 0, -1):
                feature_names_with_index.append(names + str(i))
        return feature_names_with_index

    def __delay_feature(self, input_feature):
        # TODO - what if there is > 1 feature (output?)
        # use original vectors -> make_data_frame()
        data_matrix = np.zeros((self.time_lag+1, self.data_size))
        for i in range(0, self.time_lag+1):
            for j in range(0, self.data_size):
                data_matrix[i][j] = input_feature[j + i]
        return data_matrix

    def make_data_frame(self, data):
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

        return pd.DataFrame(data_dict)

    def update_parameters(self, params):
        self.time_lag = params[0]
        self.data_size = params[1]
        self.test_ratio = params[2]

        self.__split_data()
