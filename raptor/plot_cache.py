import os
import pickle
import numpy as np

class MockPath:
    def __init__(self, wp_array):
        self._wp = wp_array
    def get_waypoints_array(self):
        return self._wp

class MockOptResult:
    def __init__(self, energy_wh, penalty_history, convergence_history, optimized_path_array):
        self.energy_wh = energy_wh
        self.penalty_history = penalty_history
        self.convergence_history = convergence_history
        self.optimized_path = MockPath(optimized_path_array)
        self.parameter_history = [] # Not used directly if we pass trials
        
def save_plot_data(out_dir, uav_name, data):
    path = os.path.join(out_dir, f"plot_data_{uav_name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_plot_data(out_dir, uav_name):
    path = os.path.join(out_dir, f"plot_data_{uav_name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)
