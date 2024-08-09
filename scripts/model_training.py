import numpy as np
import random
import os
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa
import keras as keras
from keras import metrics

from maml.models import AtomSets


def set_reproducibility(seed=None):
    if seed is None:
        seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    random.seed(seed)


def train_atomsets_classification_kfold(train_val_split_path, fold, feature_df, feature, pretraining, pretrained_model_path, target, subset_feature=False, V=1):

    print(f"Performing cross-validation with feature: {feature}")
    print(f"Using pretraining: {pretraining}")
    
    print(f"Training: {fold}") 

    set_reproducibility()

    if pretraining:
        name = feature + "_pretrained"
    else:
        name = feature

    output_dir = os.path.join(train_val_split_path, fold, name)
    print(f"Saving to: {output_dir}") 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    training_df_file_path = os.path.join(train_val_split_path, fold, "train.csv")
    validation_df_file_path = os.path.join(train_val_split_path, fold, "val.csv")
    train_df = pd.read_csv(training_df_file_path)
    val_df = pd.read_csv(validation_df_file_path)
    
    common_columns = list(set(feature_df.columns) & set(train_df.columns))
    columns_to_drop = set(common_columns) - {'icsd_collectioncode'}
    feature_df = feature_df.drop(columns=columns_to_drop)

    if subset_feature:
        feature_df = feature_df[~feature_df[subset_feature].isnull()]
    
    train_feature_df = pd.merge(train_df, feature_df, on='icsd_collectioncode')
    val_feature_df = pd.merge(val_df, feature_df, on='icsd_collectioncode')
    train_features_fold = train_feature_df[feature].to_list()
    train_targets_fold = train_feature_df[target].to_list()
    val_features_fold = val_feature_df[feature].to_list()
    val_targets_fold = val_feature_df[target].to_list()

    print(f"Training set size: {str(len(train_features_fold))}")
    print(f"Validation set size: {str(len(val_features_fold))}") 

    opt = tfa.optimizers.LAMB()
    opt = tfa.optimizers.Lookahead(opt)
    
    if V==0:
        input_dim=16
    else:
        input_dim=32
    model = AtomSets(input_dim=input_dim,
                 is_embedding=False,
                 compile_metrics=[metrics.AUC(curve="PR", name="pr_auc"), metrics.AUC(curve="ROC", name="roc_auc"), metrics.BinaryAccuracy(), metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()],
                 loss='binary_crossentropy',
                 is_classification=True,
                 symmetry_func='set2set',
                 optimizer=opt,
                 dropout=False,
                 )
    if pretraining:
        print(f"Pretrained model: {pretrained_model_path}")
        model.model.load_weights(pretrained_model_path)

    history = model.fit(train_features_fold, train_targets_fold, val_features_fold, val_targets_fold, epochs=500, verbose=3)

    history_pickle_name = os.path.join(output_dir, f"history.pkl")
    model_file_name = os.path.join(output_dir, f"model")

    with open(history_pickle_name, 'wb') as f:
        pickle.dump(history.history, f)

    model.save(model_file_name)