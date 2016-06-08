
class Params:
    gb_params = {'objective': 'reg:linear',
                 'eta': 0.05,
                 'min_child_weight': 5,
                 'subsample': 0.9,
                 'colsample_bytree': 1,
                 'silent': 1,
                 'max_depth': 3
                 }

    data_params = {}

    def get_model_params(self):
        return self.gb_params

    def get_data_params(self):
        return self.data_params
