from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import pandas as pd


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    output_weights_folder,
    early_stopping_flag,
    epochs=100,
):
    """trains the model, stores the weights, plots the loss and accuracy
    histories.

    Arguments:
        model {keras model} -- model
        inputs {list} -- list of network inputs
        outputs {list} -- list of network outputs
        output_weights_folder {string} -- folder for trained model weights
        output_names {list} -- list of network output names
    """
    loss_weights = output_weights_folder + "/loss"
    val_loss_weights = output_weights_folder + "/val_loss"
    os.mkdir(loss_weights)
    os.mkdir(val_loss_weights)
    loss_filepath = loss_weights + "/epoch-|{epoch:02d}|_loss-|{loss:.4f}|.hdf5"
    val_loss_filepath = (
        val_loss_weights + "/epoch-|{epoch:02d}|_val_loss-|{val_loss:.4f}|.hdf5"
    )
    loss_save_weights = ModelCheckpoint(
        loss_filepath,
        monitor="loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        period=1,
    )
    val_loss_save_weights = ModelCheckpoint(
        val_loss_filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        period=1,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=10)
    callbacks_list = [loss_save_weights, val_loss_save_weights]
    if early_stopping_flag:
        callbacks_list.append(early_stop)
    history = model.fit(
        X_train,
        y_train,
        validation_data=tuple([X_val, y_val]),
        epochs=epochs,
        callbacks=callbacks_list,
        batch_size=128,
    )
    accuracy_history = {k: v for (k, v) in history.history.items() if "acc" in k}
    loss_history = {k: v for (k, v) in history.history.items() if "loss" in k}
    pickle.dump(
        accuracy_history, open(output_weights_folder + "/accuracy_history.pickle", "wb")
    )
    pickle.dump(
        loss_history, open(output_weights_folder + "/loss_history.pickle", "wb")
    )
    for k in accuracy_history.keys():
        plt.plot(history.history[k])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend([k for k in accuracy_history.keys()], loc="upper left")
    plt.savefig(output_weights_folder + "/accuracy_history.png")
    plt.clf()

    for k in loss_history.keys():
        plt.plot(history.history[k])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend([k for k in loss_history.keys()], loc="upper left")
    plt.savefig(output_weights_folder + "/loss_history.png")
    plt.clf()

    for k in X_train.keys():
        output_metrics = [m for m in accuracy_history.keys() if k in m]
        for mk in output_metrics:
            plt.plot(history.history[mk])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(output_metrics, loc="upper left")
        plt.savefig(output_weights_folder + "/{}_accuracy_history.png".format(k))
        plt.clf()

        output_metrics = [m for m in loss_history.keys() if k in m]
        for mk in output_metrics:
            plt.plot(history.history[mk])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(output_metrics, loc="upper left")
        plt.savefig(output_weights_folder + "/{}_loss_history.png".format(k))
        plt.clf()

    accuracy_keys = []
    accuracy_values = []
    for k, v in accuracy_history.items():
        accuracy_keys.append(k)
        accuracy_values.append(v)

    loss_keys = []
    loss_values = []
    for k, v in loss_history.items():
        loss_keys.append(k)
        loss_values.append(v)

    model_training_results = pd.Series(
        accuracy_values + loss_values, accuracy_keys + loss_keys
    )
    return model_training_results
