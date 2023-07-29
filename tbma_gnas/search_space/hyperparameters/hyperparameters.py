from threading import Lock

from default_values import DEFAULT_HYPERPARAMETERS
from hyperparameter import HyperParameter


class HyperParameters:
    def __init__(self):
        self.lock = Lock()
        self.all_parameters = {
            layer_name: {param_name: HyperParameter(values=DEFAULT_HYPERPARAMETERS[layer_name][param_name]) for
                         param_name in DEFAULT_HYPERPARAMETERS[layer_name].keys()} for layer_name in
            DEFAULT_HYPERPARAMETERS.keys()}

    def query_for_layer(self, layer: str) -> dict:
        with self.lock:
            return {param_name: self.all_parameters[layer][param_name].query() for param_name in
                    self.all_parameters[layer].keys()}

    def learn_for_layer(self, layer: str, prev_values: dict, positive: bool):
        with self.lock:
            for param_name, prev_value in prev_values:
                self.all_parameters[layer][param_name].learn(prev_value, positive)
