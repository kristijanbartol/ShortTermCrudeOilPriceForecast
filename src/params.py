
import copy

class Params:

    eta_list = [0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    min_child_weight_list = [6, 5, 4, 3, 2]
    subsample_list = [0.6, 0.7, 0.8, 0.9]
    colsample_bytree_list = [0.9, 0.8, 0.7]
    max_depth_list = [3, 5, 7, 8, 9]

    gb_params = {'objective': 'reg:linear',
                 'eval_metric': 'rmse',
                 'seed': 2016,
                 'eta': 0.037,
                 'min_child_weight': 3,
                 'subsample': 0.7,
                 'colsample_bytree': 0.83,
                 'silent': 1,
                 'max_depth': 8
                 }

    data_params = {}

    def get_model_params(self):
        return self.gb_params

    def get_data_params(self):
        return self.data_params

    def set_eta(self, eta):
        p = copy.deepcopy(self.gb_params)
        p['eta'] = eta
        return p

    def set_min_child_weight(self, min_child_weight):
        p = copy.deepcopy(self.gb_params)
        p['min_child_weight'] = min_child_weight
        return p

    def set_subsample(self, subsample):
        p = copy.deepcopy(self.gb_params)
        p['subsample'] = subsample
        return p

    def set_colsample_bytree(self, colsample_bytree):
        p = copy.deepcopy(self.gb_params)
        p['colsample_bytree'] = colsample_bytree
        return p

    def set_max_depth(self, max_depth):
        p = copy.deepcopy(self.gb_params)
        p['max_depth'] = max_depth
        return p
