"""
This script performs repeated runs of training using the top configurations identified 
from hyperparameter optimization (HPO) for classification of superionic materials.

The script:
- Loads preprocessed training and validation data.
- Extracts the top N configurations from an HPO results file based on specified metrics.
- Repeats the training process multiple times (default: 10) for each top configuration.
- Saves training results, including model histories and trained models, for each run.

Main Functions:
- `extract_best_hpo_configs`: Extracts the top configurations based on the average or final metric values.
- `main`: Loads the data, applies the best configurations to train the models, and performs repeated runs.

Usage:
    python script.py <feature> <feature_type> <fold> <validation_type> <pretrained_model> <hpo_run_filepath>

Arguments:
- feature: Name of the feature to use for training (e.g., 'structure_simplified_megnet_site_feature_level_1_2019_4_1_formation_energy').
- feature_type: Type of feature (options: 'megnet_site', 'megnet_structure', 'm3gnet_structure').
- fold: The fold number to use for cross-validation.
- validation_type: The method of validation (e.g., 'lococv', 'kfold').
- pretrained_model: Name of the pre-trained model to use for transfer learning.
- hpo_run_filepath: Path to the pickle file containing the HPO run results.

Example:
    python script.py graph_based_features megnet_site 0 random pre_trained_model hpo_run_results.pkl

Outputs:
- Saves repeated run results in directories organized by validation fold, top configuration, and run number.
- Model training histories and models are saved in each run-specific directory.

"""

import os
import numpy as np
import pandas as pd
import pickle
import sys
import random
import re

from maml.models import AtomSetsReg, MLP

from tensorflow_addons.optimizers.cyclical_learning_rate import Triangular2CyclicalLearningRate
from tensorflow_addons.optimizers import CyclicalLearningRate
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
import keras as keras
from keras import metrics

import warnings
warnings.filterwarnings('ignore')

def set_reproducibility(seed=None):
    if seed is None:
        seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    random.seed(seed)


def extract_best_hpo_configs(hpo_filename, metric='val_pr_auc', top_n=5):

    with open(hpo_filename, 'rb') as f:
        analysis = pickle.load(f)

    window_size = 10

    # Initialize variables to keep track of maximum metric values
    max_average_metric_values = []
    max_metric_values_at_end_of_training_run = []

    # Iterate through the dictionary
    for key, df in analysis.trial_dataframes.items():
        if df.empty:
            continue
        else:
            avg_metric_over_window_size = df[metric].rolling(window=window_size).mean()
            max_average_values = avg_metric_over_window_size.max()

            # Append the top values
            max_average_metric_values.append((max_average_values, key))

            # Calculate the maximum metric value at the end of the training run
            max_value_at_end = df[metric].iloc[-window_size:-1].mean()

            # Append the top values
            max_metric_values_at_end_of_training_run.append((max_value_at_end, key))

    # Sort the lists to get the top configurations
    max_average_metric_values.sort(reverse=True)
    max_metric_values_at_end_of_training_run.sort(reverse=True)

    top_max_average_runs = {}
    top_max_final_runs = {}

    # Extract top configurations for max_final_run
    for value, key in max_metric_values_at_end_of_training_run:
        if len(top_max_final_runs) >= top_n:
            break
        config = analysis.results[key]['config']
        if config not in top_max_average_runs.values() and config not in top_max_final_runs.values():
            top_max_final_runs[key] = config

    # Extract top configurations for max_average_run
    for value, key in max_average_metric_values:
        if len(top_max_average_runs) >= top_n:
            break
        config = analysis.results[key]['config']
        if config not in top_max_average_runs.values() and config not in top_max_final_runs.values():
            top_max_average_runs[key] = config

    return {'max_average_run': top_max_average_runs,
            'max_final_run': top_max_final_runs}


def get_triangle2_lr_scheduler(init_lr, max_lr, step_size):
    scale_fn = lambda x: 1 / (2. ** (x - 1))

    # Create the cyclic learning rate scheduler
    clr = CyclicalLearningRate(
        initial_learning_rate=init_lr,
        maximal_learning_rate=max_lr,
        scale_fn=scale_fn,
        step_size=step_size
    )
    return clr


def main(feature, feature_type, fold, validation_type, pretrained_model, hpo_run_filepath):

    warnings.filterwarnings('ignore')

    base_dir = r"../data/ionic_conductivity_database_11022023_train_val_test_splits2"

    model_type = "classification"

    training_df_file_path = os.path.join(base_dir, model_type, validation_type, f"fold_{fold}", "train.csv")
    validation_df_file_path = os.path.join(base_dir, model_type, validation_type, f"fold_{fold}", "val.csv")

    train_df = pd.read_csv(training_df_file_path)
    val_df = pd.read_csv(validation_df_file_path)

    feature_df_filepath = r"../data/graph_based_features_11022023_wo_duplicates.pkl"

    print("Using structure features: ", feature)

    feature_df = pd.read_pickle(feature_df_filepath)

    common_columns = list(set(feature_df.columns) & set(train_df.columns))
    columns_to_drop = set(common_columns) - {'icsd_collectioncode'}
    feature_df = feature_df.drop(columns=columns_to_drop)

    train_feature_df = pd.merge(train_df, feature_df, on='icsd_collectioncode')
    val_feature_df = pd.merge(val_df, feature_df, on='icsd_collectioncode')

    target = "is_superionic"

    train_features_fold = train_feature_df[feature].to_list()
    train_targets_fold = train_feature_df[target].to_list()
    val_features_fold = val_feature_df[feature].to_list()
    val_targets_fold = val_feature_df[target].to_list()

    best_configs = extract_best_hpo_configs(hpo_filename=hpo_run_filepath, metric='val_pr_auc', top_n=5)

    validation_fold_dir = os.path.join(validation_type, f"fold_{fold}_repeated_runs")
    print(f"Saving results to {validation_fold_dir}")
    os.makedirs(validation_fold_dir, exist_ok=True)

    num_runs = 10

    for criterion, trial_run in best_configs.items():

        criterion_dir = os.path.join(validation_fold_dir, criterion)
        os.makedirs(criterion_dir, exist_ok=True)

        for config_name, config in trial_run.items():

            config_dir = os.path.join(criterion_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)

            set_reproducibility()

            for run in range(num_runs):

                run_dir = os.path.join(criterion_dir, config_name, f"run_{run + 1}")
                os.makedirs(run_dir, exist_ok=True)

                steps_per_epoch = int(np.ceil(len(train_features_fold) / config["batch_size"]))
                lr_schedule = Triangular2CyclicalLearningRate(initial_learning_rate=config['init_lr'],
                                                                  maximal_learning_rate=config['max_lr'],
                                                                  step_size=steps_per_epoch * 10)

                opt_name = config['optimizer']
                # If specified set up lamb_lookahead optimizer
                if opt_name == 'lamb_lookahead':
                   opt = tfa.optimizers.LAMB(learning_rate=lr_schedule)
                   opt = tfa.optimizers.Lookahead(opt)

                # Default to adam optimizer
                else:
                   opt = "adam"

                n_neurons = tuple([config["n_neurons"] for i in range(config["n_layers"])])
                n_neurons_final = tuple([config["n_neurons"] for i in range(config["n_layers"])])

                level_pattern = r'level_(\d+)'

                level_match = re.search(level_pattern, feature)

                if level_match:
                    level_number = level_match.group(1)

                else:
                    print("No match found.")


                if feature_type == 'megnet_site':

                    if level_number == '0':
                        input_dim = 16
                    else:
                        input_dim = 32

                    model = AtomSetsReg(input_dim=input_dim,
                                        is_embedding=False,
                                        compile_metrics=[metrics.AUC(curve="PR", name="pr_auc"), metrics.AUC(curve="ROC", name="roc_auc"), metrics.BinaryAccuracy(), metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()],
                                        loss='binary_crossentropy',
                                        is_classification=True,
                                        symmetry_func='set2set',
                                        optimizer=opt,
                                        dropout_prob_init=config["dropout_prob_init"],
                                        dropout_init=True,
                                        dropout_prob_final=config["dropout_prob_final"],
                                        dropout_final=True,
                                        kernel_reg_value_init=config["kernel_reg_value_init"],
                                        kernel_reg_init=True,
                                        kernel_reg_value_final=config["kernel_reg_value_final"],
                                        kernel_reg_final=True,
                                        n_neurons=n_neurons,
                                        n_neurons_final=n_neurons_final)

                elif feature_type == 'megnet_structure':
                    input_dim = 96
                    model = MLP(input_dim=input_dim,
                                compile_metrics=[metrics.AUC(curve="PR", name="pr_auc"), metrics.AUC(curve="ROC", name="roc_auc"), metrics.BinaryAccuracy(), metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()],
                                loss='binary_crossentropy',
                                is_classification=True,
                                optimizer=opt,
                                dropout=True,
                                dropout_prob=config["dropout_prob"],
                                n_neurons=n_neurons,
                                )

                elif feature_type == 'm3gnet_structure':
                    input_dim = 128
                    model = MLP(input_dim=input_dim,
                                compile_metrics=[metrics.AUC(curve="PR", name="pr_auc"), metrics.AUC(curve="ROC", name="roc_auc"), metrics.BinaryAccuracy(), metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()],
                                loss='binary_crossentropy',
                                is_classification=True,
                                optimizer=opt,
                                dropout=True,
                                dropout_prob=config["dropout_prob"],
                                n_neurons=n_neurons,
                                )

                pretrained_models_dir = r"../pretrained_models"
                pretrained_model_name = f"{pretrained_model}_trained_300epochs_nl{config['n_layers']}nn{config['n_neurons']}"
                pretrained_model_path = os.path.join(pretrained_models_dir, pretrained_model_name, "model_weights.hdf5")
                print("Using pre-trained weights from following location: ", pretrained_model_path)

                #model.model.load_weights(pretrained_model_path)

                print(f"Training_config_{config_name}")

                history = model.fit(train_features_fold, train_targets_fold, val_features_fold, val_targets_fold,
                                    epochs=500, batch_size=config["batch_size"], verbose=2)

                history_pickle_name = os.path.join(run_dir, f"history.pkl")
                model_file_name = os.path.join(run_dir, f"model")

                with open(history_pickle_name, 'wb') as f:
                    pickle.dump(history.history, f)

                #model.save(model_file_name)


if __name__ == "__main__":

    if len(sys.argv) != 7:
        print(
            "Usage: python script.py <feature> <feature_type> <fold> <validation_Type> <pretrained_model> <hpo_run_filepath>")
        sys.exit(1)

    feature = sys.argv[1]
    feature_type = sys.argv[2]
    fold = int(sys.argv[3])
    validation_type = sys.argv[4]
    pretrained_model = sys.argv[5]
    hpo_run_filepath = sys.argv[6]

    print('Feature: ', feature)
    print('Feature Type: ', feature_type)
    print('Fold: ', fold)
    print('Validation Type: ', validation_type)
    print('Pretrained Model: ', pretrained_model)

    if feature_type not in ['megnet_site', 'megnet_structure', 'm3gnet_structure']:
        raise ValueError(
            "Invalid feature type. Supported feature types are 'megnet_site', 'megnet_structure', and 'm3gnet_structure'.")

    main(feature, feature_type, fold, validation_type, pretrained_model, hpo_run_filepath)
