from os import listdir
from os.path import isfile, join
from sklearn.datasets import dump_svmlight_file
import pandas

import xgboost as xgb

from src.data import Data
from src.visualization import Visualisation as visual
from src.params import Params as prms
from src.evaluation import Evaluator as eval
import src.constants as const

input_files = [f for f in listdir(const.data_path) if isfile(join(const.data_path, f))]

data = Data(input_files)

#data.train_data.to_csv('../graphs/sample.csv')
#dtrain = xgb.DMatrix('../graphs/sample.csv')

#dump_svmlight_file(data.train_data.drop(['output']), data.train_data[['output']], 'svmlight.dat', zero_based=True,
#                   multilabel=False)

data.all_data.to_pickle('train.pkl')

visual.plot2d(data, 'crude_oil')
visual.plot_all_2d(data)

# Global variables
xgb_num_rounds = 100
eta_list = [0.02] * 100

# Prepare data for the model
hyperparams = prms.get_model_params()
xgtrain = xgb.DMatrix(data.train_data, data.train_output)
xgtest = xgb.DMatrix(data.test_data, label=data.test_output)

# Train model
model = xgb.train(hyperparams, xgtrain, xgb_num_rounds, learning_rates=eta_list)

# Get predictions
train_predictions = model.predict(xgtrain, ntree_limit=model.best_iteration)
print('Train score is: ', eval.weighted_kappa(train_predictions, data.train_output))
test_predictions = model.predict(xgtest, ntree_limit=model.best_iteration)
