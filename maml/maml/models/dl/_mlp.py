"""Multi-layer perceptron models."""
from __future__ import annotations

from maml.base import BaseDescriber, KerasModel
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from ._keras_utils import deserialize_keras_activation, deserialize_keras_optimizer
import numpy as np
import tensorflow as tf
import os
import json
import joblib


def construct_mlp(
    input_dim: int,
    n_neurons: tuple = (64, 64),
    activation: str = "relu",
    n_targets: int = 1,
    is_classification: bool = False,
    optimizer: str = "adam",
    loss: str = "mse",
    compile_metrics: tuple = (),
    dropout: bool = False,
    dropout_prob: float = 0.5,
):
    """
    Constructor for multi-layer perceptron models.

    Args:
        input_dim (int): input dimension, i.e., feature dimension
        n_neurons (tuple): list of hidden neuron sizes
        activation (str): activation function
        n_targets (int): number of targets
        is_classification (bool): whether the target is a classification problem
        optimizer (str): optimizer
        loss (str): loss function
        compile_metrics (tuple): metrics to evaluate during epochs
    """
    from tensorflow.keras.layers import Dense, Input, Dropout
    from tensorflow.keras.models import Model

    inp = Input(shape=(input_dim,))
    out_ = inp
    for n_neuron in n_neurons:
        out_ = Dense(n_neuron, activation=activation)(out_)
        if dropout:  # Add dropout layer if specified
            out_ = Dropout(dropout_prob)(out_)

    if is_classification:
        final_act: str | None = "sigmoid"
        compile_metrics += [MCCMetric(), F1ScoreMetric()]
    else:
        final_act = None
    out = Dense(n_targets, activation=final_act)(out_)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer, loss, metrics=compile_metrics)
    return model


class MCCMetric(Metric):
    def __init__(self, name='mcc', **kwargs):
        super(MCCMetric, self).__init__(name=name, **kwargs)
        self.tp = TruePositives()
        self.tn = TrueNegatives()
        self.fp = FalsePositives()
        self.fn = FalseNegatives()

    def reset_state(self):
        # Reset internal state of all tracked metrics
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred)
        self.tn.update_state(y_true, y_pred)
        self.fp.update_state(y_true, y_pred)
        self.fn.update_state(y_true, y_pred)

    def result(self):
        mcc_denominator = (self.tp.result() + self.fp.result()) * (self.tp.result() + self.fn.result()) * \
                          (self.tn.result() + self.fp.result()) * (self.tn.result() + self.fn.result())
        mcc = K.switch(K.equal(mcc_denominator, 0),
                       0.0,
                       (self.tp.result() * self.tn.result() - self.fp.result() * self.fn.result()) /
                       K.sqrt(mcc_denominator))
        return mcc


class F1ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        self.tp = TruePositives()
        self.tn = TrueNegatives()
        self.fp = FalsePositives()
        self.fn = FalseNegatives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred)
        self.tn.update_state(y_true, y_pred)
        self.fp.update_state(y_true, y_pred)
        self.fn.update_state(y_true, y_pred)

    def result(self):
        precision = self.tp.result() / (self.tp.result() + self.fp.result() + K.epsilon())
        recall = self.tp.result() / (self.tp.result() + self.fn.result() + K.epsilon())
        f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_score

    def reset_state(self):
        # Reset internal state of all tracked metrics
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()


class MLP(KerasModel):
    """This class implements the multi-layer perceptron models."""

    def __init__(
        self,
        input_dim: int | None = None,
        describer: BaseDescriber | None = None,
        n_neurons: tuple = (64, 64),
        activation: str = "relu",
        n_targets: int = 1,
        is_classification: bool = False,
        optimizer: str = "adam",
        loss: str = "mse",
        compile_metrics: tuple = (),
        dropout_prob: float = 0.5,
        dropout: bool = False,
        **kwargs,
    ):
        """
        Constructor for multi-layer perceptron models.

        Args:
            input_dim (int): input dimension, i.e., feature dimension
            activation (str): activation function
            n_targets (int): number of targets
            is_classification (bool): whether the target is a classification problem
            optimizer (str): optimizer
            loss (str): loss function
            compile_metrics (tuple): metrics to evaluate during epochs
        """
        input_dim = self.get_input_dim(describer, input_dim)

        if input_dim is None:
            raise ValueError("input_dim is not known and cannot be inferred")

        optimizer = deserialize_keras_optimizer(optimizer)
        activation = deserialize_keras_activation(activation)

        model = construct_mlp(
            input_dim=input_dim,
            n_neurons=n_neurons,
            activation=activation,
            n_targets=n_targets,
            is_classification=is_classification,
            optimizer=optimizer,
            loss=loss,
            compile_metrics=compile_metrics,
            dropout_prob=dropout_prob,
            dropout=dropout,
        )

        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.n_targets = n_targets
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.compile_metrics = compile_metrics
        self.is_classification = is_classification
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        #self.include_custom_descriptors = include_custom_descriptors

        super().__init__(describer=describer, model=model, **kwargs)

    def save(self, dirname: str):
        """Save the models and describers.

        Arguments:
            dirname (str): dirname for save
        """

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        def _to_json_type(d):
            if isinstance(d, dict):
                return {i: _to_json_type(j) for i, j in d.items()}
            if isinstance(d, np.float32):
                return float(d)
            if isinstance(d, np.int32):
                return int(d)
            if isinstance(d, (list, tuple)):
                return [_to_json_type(i) for i in d]
            return d

        with open(os.path.join(dirname, "config.json"), "w") as f:
            json.dump(
                {
                    "input_dim": self.input_dim,
                    "n_neurons": self.n_neurons,
                    "n_targets": self.n_targets,
                    "activation": _to_json_type(tf.keras.activations.serialize(self.activation)),
                    "optimizer": _to_json_type(tf.keras.optimizers.serialize(self.optimizer)),
                    "loss": self.loss,
                    "is_classification": self.is_classification,
                    "dropout": self.dropout,
                    "dropout_prob": self.dropout_prob,
                },
                f,
            )
        self.model.save(os.path.join(dirname, "model_weights.hdf5"))
