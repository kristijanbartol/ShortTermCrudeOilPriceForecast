
class Params:

    @staticmethod
    def get_model_params():
        params = dict()

        params["objective"] = "reg:linear"
        params["eta"] = 0.05
        params["min_child_weight"] = 240
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.67
        params["silent"] = 1
        params["max_depth"] = 6

        return list(params.items())

    @staticmethod
    def get_data_params(self):
        pass
