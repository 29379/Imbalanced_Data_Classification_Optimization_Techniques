import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from visualisation_methods import (plot_combined_learning_curves, plot_combined_roc_curves, plot_learning_curve, plot_mean_performance, plot_mean_roc_curve)
from experiment_parameters import classifiers_ex03

results_data_dir = 'results/data/comparing_ensamble_methods_exp_03'
results_visualisations_dir = 'results/visualisations/comparing_ensamble_methods_exp_03'

for model_name, model in classifiers_ex03.items():
    plot_mean_performance(model_name, results_data_dir, results_visualisations_dir)
    plot_mean_roc_curve(model_name, results_data_dir, results_visualisations_dir)
    # plot_learning_curve(model_name, results_data_dir, results_visualisations_dir)

# plot_combined_learning_curves(results_data_dir, results_visualisations_dir)
plot_combined_roc_curves(results_data_dir, results_visualisations_dir)
