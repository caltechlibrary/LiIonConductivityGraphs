"""
This script performs hyperparameter optimization (HPO) for a classification model to predict 
superionic conductivity in materials. It uses Ray Tune for HPO and supports models based on 
AtomSets and MLP architectures with different input feature types (MEGNet and M3GNet).

Main Steps:
- Data loading from pre-split training and validation sets
- Model definition with hyperparameter optimization
- Training and validation using predefined metrics
- Saving the best model and results in a pickle file

Usage:
    python hpo_classification.py <feature> <feature_type> <fold> <validation_type> <pretrained_model>

Arguments:
- feature: Name of the feature to use in training (e.g., graph-based features).
- feature_type: Type of feature (options: 'megnet_site', 'megnet_structure', 'm3gnet_structure').
- fold: Which fold of the data split to use (for cross-validation).
- validation_type: Validation method (kfold or lococv).
- pretrained_model: Name of the pre-trained model to use.

Example:
    python hpo_classification.py structure_simplified_megnet_site_feature_level_1_2019_4_1_formation_energy megnet_site 0 lococv matbench_v_0_1_is_metal_structure_megnet_site_feature_level_1_2019_4_1_formation_energy

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

from ray import tune
from ray import train
import ray as ray
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter

import warnings

warnings.filterwarnings('ignore')

def set_reproducibility(seed=None):
    if seed is None:
        seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    random.seed(seed)

class TuneReporter(tf.keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, reporter=None, freq="epoch", logs=None):
        """Initializer.
        Args:
            freq (str): Sets the frequency of reporting intermediate results.
        """
        self.iteration = 0
        logs = logs or {}
        self.freq = freq
        super(TuneReporter, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        from ray import tune
        logs = logs or {}
        if not self.freq == "epoch":
            return
        self.iteration += 1
        train.report(logs)


def get_triangle2_lr_scheduler(init_lr, max_lr, step_size):

    scale_fn = lambda x: 1/(2.**(x-1))

    # Create the cyclic learning rate scheduler
    clr = CyclicalLearningRate(
        initial_learning_rate=init_lr,
        maximal_learning_rate=max_lr,
        scale_fn=scale_fn,
        step_size=step_size
    )
    return clr


class TrainableAtomsetsReg():

    def __init__(self, train_data, train_targets, val_data, val_targets, input_dim, pretrained_model):
        self.train_data = train_data
        self.val_data = val_data
        self.train_targets = train_targets
        self.val_targets = val_targets
        self.input_dim = input_dim
        self.pretrained_model = pretrained_model

    def train(self, config):

        print("Training data size: ", len(self.train_data))
        print("Validation data size: ", len(self.val_data))
        
        set_reproducibility()
        print("Setting seed for reproducibility")

        steps_per_epoch = int(np.ceil(len(self.train_data) / config["batch_size"]))
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

        pretrained_models_dir = r"../data/pretrained_models"
        pretrained_model_name = f"{self.pretrained_model}_trained_300epochs_nl{config['n_layers']}nn{config['n_neurons']}"
        pretrained_model_path = os.path.join(pretrained_models_dir, pretrained_model_name, "model_weights.hdf5")
        print("Using pre-trained weights from following location: ", pretrained_model_path)

        model = AtomSetsReg(input_dim=self.input_dim,
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

        #model.model.load_weights(pretrained_model_path)

        reporter_callback = TuneReporter()
        early_stopping_callback = EarlyStopping(monitor='val_pr_auc', min_delta=0.001, verbose=1, mode='max',
                                                restore_best_weights=True, start_from_epoch=30, patience=100)

        callbacks = [reporter_callback, early_stopping_callback]

        history = model.fit(self.train_data, self.train_targets, self.val_data, self.val_targets, epochs=500,
                            batch_size=config["batch_size"], callbacks=callbacks, verbose=2)

        # Extract only the final value of each metric from history.history
        final_metrics = {}
        for metric, values in history.history.items():
            final_metrics[metric] = values[-1]

        return final_metrics


class TrainableMLP():

    def __init__(self, train_data, train_targets, val_data, val_targets, input_dim, pretrained_model):
        self.train_data = pd.DataFrame(train_data)
        self.val_data = pd.DataFrame(val_data)
        self.train_targets = np.array(train_targets)
        self.val_targets = np.array(val_targets)
        self.input_dim = input_dim
        self.pretrained_model = pretrained_model

    def train(self, config):

        set_reproducibility()
        print("Setting seed for reproducibility")

        steps_per_epoch = int(np.ceil(len(self.train_data) / config["batch_size"]))
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

        pretrained_models_dir = r"/central/groups/SeeGroup/McHaffie/maml/pretrained_models"
        pretrained_model_name = f"{self.pretrained_model}_trained_300epochs_nl{config['n_layers']}nn{config['n_neurons']}"
        pretrained_model_path = os.path.join(pretrained_models_dir, pretrained_model_name, "model_weights.hdf5")
        #print("Using pre-trained weights from following location: ", pretrained_model_path)

        model = MLP(input_dim=self.input_dim,
                    compile_metrics=["AUC", "BinaryAccuracy"],
                    loss='binary_crossentropy',
                    is_classification=True,
                    optimizer=opt,
                    dropout=True,
                    dropout_prob=config["dropout_prob"],
                    n_neurons=n_neurons,
                    )

        #model.model.load_weights(pretrained_model_path)

        reporter_callback = TuneReporter()
        early_stopping_callback = EarlyStopping(monitor='val_pr_auc', min_delta=0.001, verbose=1, mode='max',
                                                restore_best_weights=True, start_from_epoch=40, patience=100)

        callbacks = [reporter_callback, early_stopping_callback]

        history = model.fit(self.train_data, self.train_targets, self.val_data, self.val_targets, epochs=500,
                            batch_size=config["batch_size"], callbacks=callbacks, verbose=2)

        # Extract only the final value of each metric from history.history
        final_metrics = {}
        for metric, values in history.model.history.history.items():
            final_metrics[metric] = values[-1]
    
        return final_metrics


def main(feature, feature_type, fold, validation_type, pretrained_model):

    warnings.filterwarnings('ignore')

    base_dir = r"../data/ionic_conductivity_database_11022023_train_val_test_splits"

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

    ray.init()

    scheduler = AsyncHyperBandScheduler(time_attr='training_iteration',
                                        metric="val_pr_auc",
                                        mode="max",
                                        grace_period=40,
                                        max_t=500)

    search_space = {"init_lr": tune.choice([2E-5, 4E-5, 6E-5, 8E-5, 1E-4]),
                    "max_lr": tune.choice([1E-4, 1E-3, 3E-3, 5E-3, 7E-3]),
                    "batch_size": tune.choice([32, 64, 128]),
                    "optimizer": "lamb_lookahead",
                    "dropout_prob_init": tune.choice([0.1, 0.15, 0.2, 0.25, 0.3]),
                    "dropout_prob_final": tune.choice([0.1, 0.15, 0.2, 0.25, 0.3]),
                    "kernel_reg_value_init": tune.choice([1e-4, 1e-3, 1e-2, 1e-1, 1]),
                    "kernel_reg_value_final": tune.choice([1e-4, 1e-3, 1e-2, 1e-1, 1]),
                    "n_layers": tune.choice([2, 3, 4]),
                    "n_neurons": tune.choice([32, 64, 128, 256])
                    }

    search_alg = HyperOptSearch(search_space,
                                metric="val_pr_auc",
                                mode="max",
                                random_state_seed=42
                                )

    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)


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
        
        print(f'Using input dim {input_dim}')

        trainer = TrainableAtomsetsReg(train_data=train_features_fold, train_targets=train_targets_fold,
                                    val_data=val_features_fold, val_targets=val_targets_fold, input_dim=input_dim, pretrained_model=pretrained_model)
    elif feature_type == 'megnet_structure':
        input_dim = 96
        trainer = TrainableMLP(train_data=train_features_fold, train_targets=train_targets_fold,
                                    val_data=val_features_fold, val_targets=val_targets_fold, input_dim=input_dim, pretrained_model=pretrained_model)

    elif feature_type == 'm3gnet_structure':
        input_dim = 128
        trainer = TrainableMLP(train_data=train_features_fold, train_targets=train_targets_fold,
                                    val_data=val_features_fold, val_targets=val_targets_fold, input_dim=input_dim, pretrained_model=pretrained_model)
    
    pretraining_parent_model = pretrained_model
    analysis_pickle_name = f"hpo_pretrained_{pretraining_parent_model}_{model_type}_{validation_type}_fold{fold}_{feature}.pkl"
    analysis_results_path = f"..data/raytune_results/hpo_pretrained_{pretraining_parent_model}_{model_type}_{validation_type}_fold{fold}_{feature}"

    analysis = tune.run(trainer.train, num_samples=250, search_alg=search_alg, scheduler=scheduler,
                        resources_per_trial={"cpu": 1}, verbose=1, local_dir=analysis_results_path, storage_path=analysis_results_path, raise_on_failed_trial=False)

    with open(analysis_pickle_name, 'wb') as f:
        pickle.dump(analysis, f)


if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Usage: python script.py <feature> <feature_type> <fold>")
        sys.exit(1)
    
    feature = sys.argv[1]
    feature_type = sys.argv[2]
    fold = int(sys.argv[3])
    validation_type = sys.argv[4]
    pretrained_model = sys.argv[5]

    print('Feature: ', feature)
    print('Feature Type: ', feature_type)
    print('Fold: ', fold)
    print('Validation Type: ', validation_type)
    print('Pretrained Model: ', pretrained_model)

    if feature_type not in ['megnet_site', 'megnet_structure', 'm3gnet_structure']:
        raise ValueError(
            "Invalid feature type. Supported feature types are 'megnet_site', 'megnet_structure', and 'm3gnet_structure'.")

    set_reproducibility()
    print("Setting seed for reproducibility")

    main(feature, feature_type, fold, validation_type, pretrained_model)