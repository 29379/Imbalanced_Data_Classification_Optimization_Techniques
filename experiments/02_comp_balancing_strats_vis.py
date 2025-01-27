import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from visualisation_methods import (plot_combined_learning_curves, plot_combined_roc_curves, plot_learning_curve, plot_mean_performance, plot_mean_roc_curve)
from experiment_parameters import classifiers_ex02, balancing_data_strategies_ex02

results_data_dir = 'results/data/comparing_balancing_strats_exp_02'
results_visualisations_dir = 'results/visualisations/comparing_balancing_strats_exp_02'

for strat_name, strat in balancing_data_strategies_ex02.items():
    for model_name, model in classifiers_ex02.items():
        combined_name = f"{strat_name}_{model_name}"
        plot_mean_performance(combined_name, results_data_dir, results_visualisations_dir)
        plot_mean_roc_curve(combined_name, results_data_dir, results_visualisations_dir)
        # plot_learning_curve(combined_name, results_data_dir, results_visualisations_dir)

# plot_combined_learning_curves(results_data_dir, results_visualisations_dir)
plot_combined_roc_curves(results_data_dir, results_visualisations_dir)
