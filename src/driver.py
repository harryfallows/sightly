import glob
import os
import pickle
from preprocessing import encode_sequences, one_hot_sequences
from model import create_model
from train import train_model
import random
import numpy as np
from time import strftime, gmtime
from contextlib import redirect_stdout
from random_words import RandomWords
from sklearn.model_selection import train_test_split
import sys
from tensorflow.keras.utils import plot_model
import json
import argparse
from music21 import note, spanner, expressions, articulations, dynamics
import pandas as pd
import time


def main(
    system_folder,
    lstm_units,
    dropout,
    lstm_layers,
    single_loss_function,
    multi_loss_function,
    optimiser,
    last_layer_inputs,
    time_signatures,
    features,
    augmented_data_flag,
    epochs,
    early_stopping_flag,
):
    """Loads the preprocessed data and trains the model based on a number of
    hyperparameters specified in the command line parameters.

    Arguments:
        system_folder {string} -- folder containing prepreoceesed data, requirements and original data
        lstm_units {integer} -- number of lstm units per layer
        dropout {float} -- dropout after each layer
        lstm_layers {integer} -- number of lstm layers
        single_loss_function {string} -- loss function for single label classification problems
        multi_loss_function {string} -- loss function for multi label classification problems
        optimiser {string} -- optimiser for network
        last_layer_inputs {boolean} -- input sparser features in final layer of network only
        time_signatures {string} -- time signatures for model training
        features {set} -- all features the network will take into account
        augmented_data_flag {boolean} -- whether or not to use the augmented data
    """
    preprocessed_data_folder = "preprocessed_data"
    preprocessed_data_folder += "_augmented" if augmented_data_flag else ""
    trained_models_folder = system_folder + "/trained_models"
    parent_data_folder = "{}/{}".format(system_folder, preprocessed_data_folder)
    try:
        os.mkdir(trained_models_folder)
    except FileExistsError:
        pass
    feature_key = {
        "n": "Note",
        "s": "Slur",
        "d": "Dynamic",
        "p": "Spanner",
        "a": "Articulation",
        "t": "TextExpression",
    }
    obj_features = set([])
    for a, f in feature_key.items():
        if a in features:
            obj_features.add(f)
    single_label_features = set([f for f in obj_features if f in ["Note", "Dynamic"]])
    features_string = ""
    for f in features:
        features_string += f
    for ts in time_signatures:
        ts_folder_format = "{}-{}".format(ts.partition("/")[0], ts.partition("/")[-1])
        r = RandomWords()
        folder_name_prefix = r.random_word()
        folder_name = strftime(
            "{}_features-{}_augmented_{}-lstm_units-{}_dropout-{}_lstm_layers-{}_single_loss_function-{}_multi_loss_function-{}_optimiser-{}_epochs-{}_last_layer_inputs-{}_%a-%d-%b-%H-%M-%S".format(
                folder_name_prefix,
                features_string,
                "T" if augmented_data_flag else "F",
                lstm_units,
                dropout,
                lstm_layers,
                single_loss_function,
                multi_loss_function,
                optimiser,
                epochs,
                "T" if last_layer_inputs else "F",
            ),
            gmtime(),
        )
        model_hyper_params = pd.Series(
            [
                folder_name_prefix,
                lstm_units,
                dropout,
                lstm_layers,
                single_loss_function,
                multi_loss_function,
                optimiser,
                epochs,
                last_layer_inputs,
            ],
            [
                "folder_name_prefix",
                "lstm_units",
                "dropout",
                "lstm_layers",
                "single_loss_function",
                "multi_loss_function",
                "optimiser",
                "epochs",
                "last_layer_inputs",
            ],
        )
        parent_folder = "{}/{}".format(trained_models_folder, ts_folder_format)
        try:
            os.mkdir(parent_folder)
        except FileExistsError:
            pass
        folder_dir = "{}/{}".format(parent_folder, folder_name)
        os.mkdir(folder_dir)
        # pickle.dump(single_label_features, open("{}/single_label_features.pickle".format(folder_dir), "wb"))
        # pickle.dump(obj_features, open("{}/features.pickle".format(folder_dir), "wb"))
        with open("{}/resource_info.json".format(folder_dir), "w+") as f:
            json.dump(
                {
                    "system_folder": system_folder,
                    "preprocessed_data_folder": "{}/{}".format(
                        preprocessed_data_folder, ts_folder_format
                    ),
                },
                f,
            )

        pickle.dump(
            {
                "time_signature": ts,
                "single_label_features": single_label_features,
                "obj_features": obj_features,
            },
            open("{}/model_choices.pickle".format(folder_dir), "wb"),
        )

        data_folder = "{}/{}".format(parent_data_folder, ts_folder_format)
        X_train = pickle.load(open("{}/X_train.pickle".format(data_folder), "rb"))
        X_val = pickle.load(open("{}/X_val.pickle".format(data_folder), "rb"))
        y_val = pickle.load(open("{}/y_val.pickle".format(data_folder), "rb"))
        # X_train = {k: np.reshape(v, (-1, v.shape[1], 1)) for k, v in X_train.items() if k in obj_features}
        X_train = {k: v for k, v in X_train.items() if k in obj_features}
        X_val = {k: v for k, v in X_val.items() if k in obj_features}
        y_val = {k: v for k, v in y_val.items() if k in obj_features}

        inputs_1 = {}
        inputs_2 = {}
        loss_functions = {}
        for k, v in X_train.items():
            if k in ["Note", "Slur", "Dynamic"]:
                if k == "Note":
                    inputs_1.update({k: v})
                else:
                    inputs_2.update({k: v})
                if k == "Slur":
                    loss_functions.update({k + "_o": multi_loss_function})
                else:
                    loss_functions.update({k + "_o": single_loss_function})
            else:
                inputs_2.update({k: v})
                loss_functions.update({k + "_o": multi_loss_function})
        y_train = pickle.load(open("{}/y_train.pickle".format(data_folder), "rb"))
        outputs = {k + "_o": v for k, v in y_train.items()}
        if last_layer_inputs == False:
            inputs_1.update(inputs_2)
            inputs_2 = {}

        model, hyper_param_summary = create_model(
            inputs_1,
            inputs_2,
            lstm_units,
            dropout,
            lstm_layers,
            loss_functions,
            optimiser,
        )
        model.save("{}/model.h5".format(folder_dir))

        with open(folder_dir + "/model_summary.txt", "w") as f:
            with redirect_stdout(f):
                model.summary()
                print(hyper_param_summary)

        plot_model(model, to_file=folder_dir + "/model.png")
        output_weights_folder = "{}/output_weights".format(folder_dir)
        os.mkdir(output_weights_folder)

        start_time = time.time()
        model_training_results = train_model(
            model,
            {**inputs_1, **inputs_2},
            outputs,
            X_val,
            {k + "_o": v for k, v in y_val.items()},
            output_weights_folder,
            early_stopping_flag,
            epochs,
        )
        training_time = time.time() - start_time

        for i in ["loss", "val_loss"]:
            loss_folder = output_weights_folder + "/" + i
            list_of_files = glob.glob(loss_folder + "/*.hdf5")
            latest_file = max(list_of_files, key=os.path.getctime)
            epoch = int(latest_file.split("/")[-1].split("|")[1])
            if i == "loss":
                loss_epoch = epoch
            else:
                val_loss_epoch = epoch
            list_of_files.remove(latest_file)
            for dead_weight_file in list_of_files:
                os.remove(dead_weight_file)

        hyper_params_and_training_results = pd.Series(
            [loss_epoch, val_loss_epoch], ["loss_epoch", "val_loss_epoch"]
        )
        for s, v in model_training_results.items():
            reduced = pd.Series(
                v[val_loss_epoch - 1 if "val_loss" in s else loss_epoch - 1], [s]
            )
            hyper_params_and_training_results = hyper_params_and_training_results.append(
                reduced
            )

        hyper_params_and_training_results = hyper_params_and_training_results.append(
            pd.Series([training_time], ["training_time"])
        )

        pickle.dump(
            hyper_params_and_training_results.append(model_hyper_params),
            open(output_weights_folder + "/training_results.pickle", "wb"),
        )


if __name__ == "__main__":
    """Parse all the parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", dest="folder", help="Folder containing data.")
    parser.add_argument(
        "--features",
        dest="features",
        help="Which extra musical features to have. Slurs: s, Dynamics: d, Spanners: p, Articulations: a, Text expressions: t. Example format: --features sdpat",
        default="s",
    )
    parser.add_argument(
        "-a",
        "--dont_augment_data",
        dest="augmented_data_flag",
        default=True,
        help="Don't use augmented data flag. -a for don't augment, nothing for augment.",
        action="store_false",
    )
    parser.add_argument(
        "--early_stopping",
        dest="early_stopping_flag",
        default=False,
        help="Stop the model once the validation loss has reached a minimum.",
        action="store_true",
    )
    parser.add_argument(
        "-u",
        "--lstm_units",
        dest="lstm_units",
        default=512,
        help="Number of LSTM units in each layer.",
        type=int,
    )
    parser.add_argument(
        "-d",
        "--dropout",
        dest="dropout",
        default=0.3,
        help="Dropout after each LSTM layer.",
        type=float,
    )
    parser.add_argument(
        "-l",
        "--lstm_layers",
        dest="lstm_layers",
        default=3,
        help="Number of LSTM layers in network.",
        type=int,
    )
    parser.add_argument(
        "-s",
        "--single_loss_function",
        dest="single_loss_function",
        default="categorical_crossentropy",
        help="Loss function for single label classification features (e.g. notes).",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--multi_loss_function",
        dest="multi_loss_function",
        default="binary_crossentropy",
        help="Loss function for multi label classification features (e.g. expressions).",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--optimiser",
        dest="optimiser",
        default="adam",
        help="Optimiser for network.",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--last_layer_inputs",
        dest="last_layer_inputs",
        default=False,
        help="Input multi-label features before final layer of network.",
        action="store_true",
    )
    parser.add_argument(
        "--time_signatures",
        dest="time_signatures",
        default=[],
        help="Time signatures to create models for.",
        nargs="+",
    )

    parser.add_argument(
        "--epochs",
        help="Number of epochs to train for.",
        dest="epochs",
        default=60,
        type=int,
    )
    args = parser.parse_args()
    folder = args.folder
    augmented_data_flag = args.augmented_data_flag
    lstm_units = args.lstm_units
    dropout = args.dropout
    lstm_layers = args.lstm_layers
    single_loss_function = args.single_loss_function
    multi_loss_function = args.multi_loss_function
    optimiser = args.optimiser
    last_layer_inputs = args.last_layer_inputs
    epochs = args.epochs
    early_stopping_flag = args.early_stopping_flag
    requirements = json.load(open("{}/requirements.json".format(folder)))
    time_signatures = (
        args.time_signatures
        if args.time_signatures != []
        else requirements["time_signatures"]
    )
    features = args.features + "n"
    main(
        folder,
        int(lstm_units),
        float(dropout),
        int(lstm_layers),
        str(single_loss_function),
        str(multi_loss_function),
        str(optimiser),
        bool(last_layer_inputs),
        list(time_signatures),
        set(features),
        bool(augmented_data_flag),
        int(epochs),
        early_stopping_flag,
    )

