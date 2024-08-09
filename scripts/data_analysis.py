import os
import pickle
import pandas as pd
import json
import numpy as np


def get_run_history(history_path):
    """
    Load the run history from a pickle file.

    Args:
        history_path (str): The path to the pickle file containing the run history.

    Returns:
        pd.DataFrame: A DataFrame containing the run history.
    """
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    return pd.DataFrame(history)


def get_kfold_run_data(data_dir, feature):
    """
    Get the run data for a specific feature across all folds.

    Args:
        data_dir (str): The directory containing the fold data.
        feature (str): The specific feature to extract run data for.

    Returns:
        list: A list of DataFrames containing the run history for each fold.
    """
    feature_metrics = []
    
    for fold_dir in os.listdir(data_dir):
        fold_path = os.path.join(data_dir, fold_dir)
        if os.path.isdir(fold_path):
            feature_path = os.path.join(fold_path, feature)
            if os.path.isdir(feature_path):
                history_path = os.path.join(feature_path, "history.pkl")
                if os.path.exists(history_path):
                    feature_metrics.append(get_run_history(history_path))

    return feature_metrics


def get_kfold_run_average_std(data_dir, feature):
    """
    Calculate the average and standard deviation of run data for a specific feature across all folds.

    Args:
        data_dir (str): The directory containing the fold data.
        feature (str): The specific feature to calculate statistics for.

    Returns:
        tuple: A tuple containing two DataFrames - the average and the standard deviation of the feature data.
    """
    feature_metrics = get_kfold_run_data(data_dir=data_dir, feature=feature)
    feature_df = pd.concat(feature_metrics, axis=0)
    average_feature_data = feature_df.groupby(feature_df.index).mean()
    std_feature_data = feature_df.groupby(feature_df.index).std()

    return average_feature_data, std_feature_data


def get_controls(controls_filepath, validation_type, metric, fold=None):
    """
    Get control metrics from a JSON file.

    Args:
        controls_filepath (str): The path to the JSON file containing control metrics.
        validation_type (str): The type of validation (e.g., 'kfold', 'loocv').
        metric (str): The metric to extract controls for.
        fold (int, optional): The fold number, if applicable.

    Returns:
        tuple: A tuple containing the mean and standard deviation for both the average and shuffled controls.
    """
    with open(controls_filepath, 'r') as f:
        controls = json.load(f)
    metric_name = metric.split("val_")[1]
    if fold:
        mean_control_avg = controls['fold_averages'][validation_type][f"fold_{fold}"][f"{metric_name}_avg"]
        shuffled_control_avg = controls['fold_averages'][validation_type][f"fold_{fold}"][f"{metric_name}_shuffled"]
        mean_control_std = controls['fold_standard_deviations'][validation_type][f"fold_{fold}"][f"{metric_name}_avg"]
        shuffled_control_std = controls['fold_standard_deviations'][validation_type][f"fold_{fold}"][f"{metric_name}_shuffled"]
    else:
        mean_control_avg = controls['overall_averages'][validation_type][f"{metric_name}_avg"]
        shuffled_control_avg = controls['overall_averages'][validation_type][f"{metric_name}_shuffled"]
        mean_control_std = controls['overall_standard_deviations'][validation_type][f"{metric_name}_avg"]
        shuffled_control_std = controls['overall_standard_deviations'][validation_type][f"{metric_name}_shuffled"]
        
    return mean_control_avg, mean_control_std, shuffled_control_avg, shuffled_control_std
    

def get_run_best_epoch_and_value(history_path, metric, min_epoch=30, window_size=10):
    """
    Get the best epoch and its corresponding metric value from the run history.

    Args:
        history_path (str): The path to the pickle file containing the run history.
        metric (str): The metric to evaluate.
        min_epoch (int): The minimum epoch to consider.
        window_size (int): The size of the rolling window to smooth the metric values.

    Returns:
        tuple: A tuple containing the best epoch and the best metric value.
    """
    run_history = get_run_history(history_path)
    run_filtered = run_history[run_history.index >= min_epoch]
    rolling_avg = run_filtered[metric].rolling(window=window_size).mean()
    best_epoch_idx = rolling_avg.idxmax()
    best_epoch = best_epoch_idx - (window_size - 1) // 2

    best_metric_value = rolling_avg.max()
    return best_epoch, best_metric_value


def get_best_epoch_across_kfolds(data_dir, feature, metric, top_n=1, min_epoch=30):
    """
    Get the average best epoch across all folds for a specific feature.

    Args:
        data_dir (str): The directory containing the fold data.
        feature (str): The specific feature to evaluate.
        metric (str): The metric to evaluate.
        top_n (int): The number of top epochs to consider.
        min_epoch (int): The minimum epoch to consider.

    Returns:
        int: The average best epoch across all folds.
    """
    feature_best_epochs = []
    for fold in os.listdir(data_dir):
        fold_num = int(fold.split("_")[-1])
        fold_dir = os.path.join(data_dir, fold)
        history_path = os.path.join(data_dir, fold, feature, "history.pkl")
        best_epoch, _ = get_run_best_epoch_and_value(history_path=history_path, metric=metric, min_epoch=30,
                                                     window_size=10)
        feature_best_epochs.append(best_epoch)
    average_best_epoch = int(np.mean(feature_best_epochs))
    return average_best_epoch


def get_repeated_run_data(config_dir):
    run_histories = []
    for run_dir in os.listdir(config_dir):
        history_path = os.path.join(config_dir, run_dir, 'history.pkl')
        run_histories.append(get_run_history(history_path))
    return run_histories


def get_fold_repeated_run_data(data_dir):
    config_histories = {}
    for config in os.listdir(data_dir):
            config_path = os.path.join(data_dir, config)
            config_histories[config] = get_repeated_run_data(config_path)
    return config_histories


def get_best_hpo_configs_averaged_across_runs(data_dir, metric, top_n=1, min_epoch=30):

    average_data = {}
    config_histories = get_fold_repeated_run_data(data_dir)
    for config, config_data in config_histories.items():
        config_df = pd.concat(config_data, axis=0)
        average_data[config] = config_df.groupby(config_df.index).mean()

    window_size = 10

    # Initialize variables to keep track of maximum metric values
    max_average_metric_values = []
    max_metric_values_at_end_of_training_run = []

    # Iterate through the dictionary
    for config, average_df in average_data.items():
        if average_df.empty:
            continue
        else:
            # Filter average values for epochs greater than or equal to min_epoch
            average_df = average_df.loc[average_df.index >= min_epoch]

            avg_metric_over_window_size = average_df[metric].rolling(window=window_size).mean()
            max_average_values = avg_metric_over_window_size.max()

            # Append the top values
            max_average_metric_values.append((max_average_values, config))

    max_average_metric_values.sort(reverse=True)

    top_max_average_configs = [i[1] for i in max_average_metric_values[0:top_n]]

    return top_max_average_configs


def get_average_best_hpo_configs_averaged_across_runs(data_dir, metric, top_n=1):

    config_histories = get_fold_repeated_run_data(data_dir)
    top_max_average_configs = get_best_hpo_configs_averaged_across_runs(data_dir, metric, top_n=top_n)

    best_configs_data = []

    for config in top_max_average_configs:
        best_configs_data += config_histories[config]

    fold_df = pd.concat(best_configs_data, axis=0)
    fold_average = fold_df.groupby(fold_df.index).mean()
    fold_std = fold_df.groupby(fold_df.index).std()

    return fold_average, fold_std


def get_best_epoch_across_locofolds(data_dir, feature, metric, top_n=1, min_epoch=30):
    
    feature_best_epochs = []
    for fold in os.listdir(data_dir):
        fold_num = int(fold.split("_")[-1])
        fold_dir = os.path.join(data_dir, fold)
        fold_feature_dir = os.path.join(fold_dir, feature)
        
        top_configs = get_best_hpo_configs_averaged_across_runs(data_dir=fold_feature_dir, metric=metric, top_n=1, min_epoch=30)
        for config in top_configs:
            config_dir = os.path.join(fold_feature_dir, config)
            for run_dir in os.listdir(config_dir):
                history_path = os.path.join(config_dir, run_dir, "history.pkl")
                best_epoch, _ = get_run_best_epoch_and_value(history_path=history_path, metric=metric, min_epoch=30, window_size=10)
                feature_best_epochs.append(best_epoch)
                
    average_best_epoch = int(np.mean(feature_best_epochs))
    return average_best_epoch


def get_best_epoch_across_runs(config_dir, feature, metric, min_epoch=30, window_size=10):

    config_best_epochs = []
    for run_dir in os.listdir(config_dir):
        history_path = os.path.join(config_dir, run_dir, "history.pkl")
        best_epoch, _ = get_run_best_epoch_and_value(history_path=history_path, metric=metric, min_epoch=min_epoch, window_size=window_size)
        config_best_epochs.append(best_epoch)
                
    average_best_epoch = int(np.mean(config_best_epochs))
    return average_best_epoch


def get_hpo_analysis(analysis_filename):
    with open(analysis_filename, 'rb') as f:
        analysis = pickle.load(f)   
    return analysis