import numpy as np
import math
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def create_confusion_matrices(
    model, test_inputs, test_outputs, decodings, test_scores_folder, model_choices
):
    """Creates confusion matrices based on test data.

    Arguments:
        model {keras model} -- trained model
        test_inputs {dictionary} -- network test inputs
        test_outputs {dictionary} -- network test outputs
        decodings {dictionary} -- decodings for each feature
        test_scores_folder {string} -- folder to save confusion matrices
        model_choices {dictionary} -- contains certain model choices e.g. which features used
    """
    print("=" * 50)
    print("Creating confusion matrices.")
    print("=" * 50)
    predicted_outputs = model.predict(test_inputs)
    if len(model.output_names) != 1:
        predicted_outputs = {
            k[:-2]: v for k, v in zip(model.output_names, predicted_outputs)
        }
    else:
        predicted_outputs = {"Note": predicted_outputs}
    create_confusion_matrix_notes(
        model,
        predicted_outputs["Note"],
        test_outputs["Note"],
        decodings["Note"],
        test_scores_folder,
    )
    single_label_keys = [
        k
        for k in predicted_outputs.keys()
        if k in model_choices["single_label_features"] and k != "Note"
    ]
    multi_label_keys = [
        k
        for k in predicted_outputs.keys()
        if k not in model_choices["single_label_features"]
    ]
    for k in single_label_keys:
        create_confusion_matrix_single_label(
            model,
            predicted_outputs[k],
            test_outputs[k],
            decodings[k],
            test_scores_folder,
            k,
        )
    for k in multi_label_keys:
        create_confusion_matrix_multi_label(
            model,
            predicted_outputs[k],
            test_outputs[k],
            decodings[k],
            test_scores_folder,
        )


def create_confusion_matrix_notes(
    model, o_pred, o_test, pitch_decodings, test_scores_folder
):
    """Creates confusion matrices for notes.

    Arguments:
        model {keras model} -- trained model
        o_pred {nd numpy array} -- predicted note pitches
        o_test {nd numpy array} -- real note pitches
        pitch_decodings {dictionary} -- pitch decodings
        test_scores_folder {string} -- folder to save figures into
    """
    o_pred_class = [pitch_decodings[np.argmax(o)] for o in o_pred]
    o_test_class = [pitch_decodings[np.argmax(o)] for o in o_test]

    cm = confusion_matrix(
        o_test_class, o_pred_class, labels=list(pitch_decodings.values())
    )
    df_cm = pd.DataFrame(
        cm, index=pitch_decodings.values(), columns=pitch_decodings.values()
    )
    normalized_df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    plt.figure(figsize=(12, 12))
    sn.heatmap(normalized_df_cm, cmap="Blues", annot=True)

    plt.title("Note Confusion Matrix")
    plt.savefig(test_scores_folder + "/note_confusion_matrix.png")
    plt.clf()

    single_octave_o_pred_class = [
        o_c[:-1] if o_c != "Rest" else "Rest" for o_c in o_pred_class
    ]
    single_octave_o_test_class = [
        o_c[:-1] if o_c != "Rest" else "Rest" for o_c in o_test_class
    ]
    pitch_decodings_single_octave = list(
        set([p[:-1] if p != "Rest" else "Rest" for p in pitch_decodings.values()])
    )
    so_cm = confusion_matrix(
        single_octave_o_test_class,
        single_octave_o_pred_class,
        labels=list(pitch_decodings_single_octave),
    )
    df_so_cm = pd.DataFrame(
        so_cm,
        index=pitch_decodings_single_octave,
        columns=pitch_decodings_single_octave,
    )
    normalized_df_so_cm = df_so_cm.div(df_so_cm.sum(axis=1), axis=0)
    plt.figure(figsize=(12, 12))
    sn.heatmap(normalized_df_so_cm, cmap="Blues", annot=True)

    plt.title("Note Confusion Matrix (Single Octave)")
    plt.savefig(test_scores_folder + "/note_confusion_matrix_single_octave.png")
    plt.clf()


def create_confusion_matrix_single_label(
    model, o_pred, o_test, decodings, test_scores_folder, obj_name
):
    """Creates confusion matrices for single label features (dynamics).

    Arguments:
        model {keras model} -- trained model
        o_pred {nd numpy array} -- predicted note pitches
        o_test {nd numpy array} -- real note pitches
        pitch_decodings {dictionary} -- object decodings
        test_scores_folder {string} -- folder to save figures into
        obj_name {string} -- name of object
    """
    o_pred_class = [decodings[np.argmax(o)] for o in o_pred]
    o_test_class = [decodings[np.argmax(o)] for o in o_test]

    cm = confusion_matrix(o_test_class, o_pred_class, labels=list(decodings.values()))
    df_cm = pd.DataFrame(cm, index=decodings.values(), columns=decodings.values())
    normalized_df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    plt.figure(figsize=(12, 12))
    sn.heatmap(normalized_df_cm, cmap="Blues", annot=True)

    plt.title("{} Confusion Matrix".format(obj_name))
    plt.savefig(test_scores_folder + "/{}_confusion_matrix.png".format(obj_name))
    plt.clf()


def create_confusion_matrix_multi_label(
    model, o_pred, o_test, decodings, test_scores_folder
):
    """Creates confusion matrices for multi label features.

    Arguments:
        model {keras model} -- trained model
        o_pred {nd numpy array} -- predicted note pitches
        o_test {nd numpy array} -- real note pitches
        pitch_decodings {dictionary} -- object decodings
        test_scores_folder {string} -- folder to save figures into
    """
    for i, obj_name in decodings.items():
        str_name = obj_name
        labels = ["not {}".format(str(str_name)), str(str_name)]
        o_pred_class = [labels[1] if o_p[i] > 0.5 else labels[0] for o_p in o_pred]
        o_test_class = [labels[int(o[i])] for o in o_test]

        cm = confusion_matrix(o_test_class, o_pred_class, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        normalized_df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
        plt.figure(figsize=(12, 12))
        sn.heatmap(normalized_df_cm, cmap="Blues", annot=True)

        plt.title("{} Confusion Matrix".format(str_name))
        plt.savefig(test_scores_folder + "/{}_confusion_matrix.png".format(str_name))
        plt.clf()
