from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, LSTM, Input
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from losses import categorical_focal_loss, binary_focal_loss
import numpy as np
from music21 import note, spanner

def create_model(inputs_1, inputs_2, lstm_units, dropout, layers, loss_functions, optimiser):
    """Create a neural network based on some hyperparametes.

    Arguments:
        inputs_1 {list} -- inputs for first layer of network
        inputs_2 {list} -- inputs for final layer of network
        layer_names {list} -- names for each input
        lstm_units {integer} -- number of lstm units per layer
        dropout {float} -- dropout after each lstm layer
        layers {integer} -- number of lstm layers
        loss_functions {list} -- loss function for each input
        optimiser {string} -- optimiser for training network

    Returns:
        tuple -- (model {keras model object}: model object, hyper_param_summary {string}: summary of hyperparameters)
    """    
    hyper_param_summary = "=" * 50 + "\nModel Hyperparameter Summary:\n" + "=" * 50 + "\n\tLSTM layers: {}\n".format(layers) + "\tLSTM units: {}\n".format(lstm_units) + "\tDropout: {}\n".format(dropout) + "\tSingle label loss function: {}\n".format(loss_functions["Note_o"]) + "\tMulti label loss function: {}\n".format(loss_functions["Slur_o"] if "Slur_o" in inputs_1.keys() else "N/a") + "\tOptimiser: {}\n".format(optimiser) + "=" * 50
    print(hyper_param_summary)
    inputs_1 = {k:np.reshape(v, (-1,v.shape[1],1)) if len(v.shape)==2 else v for k,v in inputs_1.items()}
    inputs_2 = {k:np.reshape(v, (-1,v.shape[1],1)) if len(v.shape)==2 else v for k,v in inputs_2.items()}
    network_inputs = []
    for k, v in inputs_1.items():
        network_inputs.append(Input(shape=(v.shape[1:]), name=k))
    if len(network_inputs) == 1:
        merged_inputs = network_inputs[0]
    else:
        merged_inputs = Concatenate(axis=2)(network_inputs)
    last_layer = merged_inputs
    for _ in range(layers-1):
        lstm = LSTM(lstm_units,return_sequences=True,recurrent_dropout = 0.2)(last_layer)
        last_layer = Dropout(dropout)(lstm)
    network_inputs_2 = []
    for k, v in inputs_2.items():
        network_inputs_2.append(Input(shape=(v.shape[1:]), name=k))
    if network_inputs_2 != []:
        merged_inputs = Concatenate(axis=2)(network_inputs_2+[last_layer])
        last_layer = merged_inputs
    lstm = LSTM(lstm_units, recurrent_dropout = 0.2)(last_layer)
    last_dropout = Dropout(dropout)(lstm)
    network_outputs = []
    inputs = {**inputs_1, **inputs_2}
    accuracy_functions = {}
    for k, v in loss_functions.items():
        output_nodes = inputs[k[:-2]].shape[2]
        if "binary" in v:
            activation_function = "sigmoid"
            accuracy_function = metrics.binary_accuracy
        else:
            activation_function = "softmax"
            accuracy_function = metrics.categorical_accuracy
        accuracy_functions.update({k:accuracy_function})
        network_outputs.append(Dense(output_nodes, activation=activation_function, name=k)(last_dropout))
    model = Model(network_inputs+network_inputs_2, network_outputs)
    loss_functions = {k:categorical_focal_loss(alpha=.25, gamma=2) if v == 'categorical_focal_loss' else v for k,v in loss_functions.items()}
    loss_functions = {k:binary_focal_loss(alpha=.25, gamma=2) if v == 'binary_focal_loss' else v for k,v in loss_functions.items()}
    model.compile(loss=loss_functions, optimizer=optimiser, metrics=accuracy_functions)
    print(model.get_config())
    return model, hyper_param_summary
