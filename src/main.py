from os import listdir
from os.path import isfile, join
import seaborn

import xgboost as xgb
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
import operator

import src.constants as const
from src.data import Data
from src.evaluation import Evaluator as eval
from src.params import Params
from src.visualization import Visualisation as visual

np.random.seed(2016)

input_files = [f for f in listdir(const.data_path) if isfile(join(const.data_path, f))]

data = Data(input_files)
params = Params()

visual.create_feature_map(data.lagged_feature_names)

#visual.plot2d(data, 'crude_oil')
#visual.plot_all_2d(data)

# Global variables
xgb_num_rounds = 30
eta_list = [0.02] * 25

train_data = data.train_data.drop(['output'], axis=1).as_matrix()
train_label = data.train_data[['output']].as_matrix()
test_data = data.test_data.drop(['output'], axis=1).as_matrix()
test_label = data.test_data[['output']].as_matrix()

# Prepare data for the model
hyperparams = params.get_model_params()
xgtrain = xgb.DMatrix(train_data, train_label)
xgtest = xgb.DMatrix(test_data, test_label)

# Train model
model = xgb.train(hyperparams, xgtrain, xgb_num_rounds)
model.dump_model(const.exports_path + 'model.dump.out')
visual.feature_importance(model)

# Get predictions
train_predictions = model.predict(xgtrain, ntree_limit=model.best_iteration)
print('Train score is: ', round(eval.weighted_kappa(train_predictions, train_label), 6))

test_predictions = model.predict(xgtest, ntree_limit=model.best_iteration)
print('Test score is: ', round(eval.weighted_kappa(test_predictions, test_label), 6))
