from os import listdir
from os.path import isfile, join
import seaborn

import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor

import src.constants as const
from src.data import Data
from src.evaluation import Evaluator as eval
from src.params import Params
from src.visualization import Visualisation as visual

input_files = [f for f in listdir(const.data_path) if isfile(join(const.data_path, f))]

data = Data(input_files, fill_strategy='gate')
params = Params()

visual.create_feature_map(data.lagged_feature_names)

#visual.plot2d(data, 'crude_oil')
#visual.plot_all_2d(data)

# Global variables
xgb_num_rounds = 80

td = np.array(data.train_data.drop(['output'], axis=1))

train_data = data.train_data.drop(['output'], axis=1).as_matrix()
train_label = data.train_data[['output']].as_matrix()
test_data = data.test_data.drop(['output'], axis=1).as_matrix()
test_label = data.test_data[['output']].as_matrix()

# Prepare data for the model
hyperparams = params.get_model_params()
xgtrain = xgb.DMatrix(train_data, train_label)
xgtest = xgb.DMatrix(test_data, test_label)

# Train model with best parameters
model = xgb.train(hyperparams, xgtrain, xgb_num_rounds)
model.dump_model(const.exports_path + 'model.dump.out')
visual.feature_importance(model)

# Get predictions
train_predictions = model.predict(xgtrain, ntree_limit=model.best_iteration)
print('Train score is: ', round(eval.rmse(train_predictions, train_label), 6))

test_predictions = model.predict(xgtest, ntree_limit=model.best_iteration)
print('Test score is: ', round(eval.rmse(test_predictions, test_label), 6))


# try random forest
rf = RandomForestRegressor(n_estimators=75, min_samples_split=2)
rf.fit(train_data, np.ravel(train_label))

r2 = eval.rmse(test_label, rf.predict(test_data))
print(r2)

# Plot "grid search" process
num_rounds_list = [30, 50, 80, 120]


def search_param(params_function, param_values):
    for param_value in param_values:
        model = xgb.train(params_function(param_value), xgtrain, xgb_num_rounds)

        train_predictions = model.predict(xgtrain, ntree_limit=model.best_iteration)
        print('Train score is: ', round(eval.rmse(train_predictions, train_label), 6))

        test_predictions = model.predict(xgtest, ntree_limit=model.best_iteration)
        print('Test score is: ', round(eval.rmse(test_predictions, test_label), 6))

'''
for num_rounds in num_rounds_list:
    xgtrain = xgb.DMatrix(train_data, train_label)
    xgtest = xgb.DMatrix(test_data, test_label)

    model = xgb.train(hyperparams, xgtrain, num_boost_round=num_rounds)

    train_predictions = model.predict(xgtrain, ntree_limit=model.best_iteration)
    print('Train score is: ', round(eval.rmse(train_predictions, train_label), 6))

    test_predictions = model.predict(xgtest, ntree_limit=model.best_iteration)
    print('Test score is: ', round(eval.rmse(test_predictions, test_label), 6))

search_param(params.set_eta, params.eta_list)
search_param(params.set_min_child_weight, params.min_child_weight_list)
search_param(params.set_subsample, params.subsample_list)
search_param(params.set_colsample_bytree, params.colsample_bytree_list)
search_param(params.set_max_depth, params.max_depth_list)

'''