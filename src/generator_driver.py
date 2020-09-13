import glob
import os
import pickle
from preprocessing import (
    encode_sequences,
    encode_multi_sequences,
    one_hot_sequences,
    multi_hot_sequences,
)

import random
import numpy as np
from tensorflow.keras.models import load_model
import sys
from generate import generate_pieces_from_seeds, generate_piece_from_artificial_seeds
from test import create_confusion_matrices
from losses import categorical_focal_loss, binary_focal_loss
import dill
import argparse
import json
from music21 import note
from statistics import mean


def main(
    model_folder,
    no_generated_pieces,
    loss_flag,
    val_loss_flag,
    end_sequences_flag,
    artificial_seeds_flag,
):
    """Generates confusion matrices and pieces.

    Arguments:
        model_folder {string} -- folder containing trained model to use
        no_generated_pieces {int} -- number of pieces to generate
        loss_flag {bool} -- whether or not to use weights saved at the minimum loss of training
        val_loss_flag {bool} -- whether or not to use weights saved at the minimum validation loss of training
        end_sequences_flag {bool} -- whether or not to use end sequences as seeds
        artificial_seeds_flag {bool} -- whether or not to use artificial seeds as seeds

    Raises:
        IndexError: raised if there aren't enough end seeds to use
    """
    model_resource_info = json.load(
        open("{}/resource_info.json".format(model_folder), "rb")
    )
    system_folder = model_resource_info["system_folder"]
    preprocessed_data_folder = "{}/{}".format(
        system_folder, model_resource_info["preprocessed_data_folder"]
    )
    grade_reqs = json.load(open("{}/requirements.json".format(system_folder)))
    numbers_of_bars = mean(grade_reqs["length"])
    X_test = pickle.load(
        open("{}/X_test.pickle".format(preprocessed_data_folder), "rb")
    )
    y_test = pickle.load(
        open("{}/y_test.pickle".format(preprocessed_data_folder), "rb")
    )

    encodings = pickle.load(
        open("{}/encodings.pickle".format(preprocessed_data_folder), "rb")
    )
    decodings = pickle.load(
        open("{}/decodings.pickle".format(preprocessed_data_folder), "rb")
    )

    results_folder = "{}/test_results".format(model_folder)
    try:
        os.mkdir(results_folder)
    except FileExistsError:
        pass

    preprocessed_end_sequences = pickle.load(
        open(
            "{}/preprocessed_end_sequences.pickle".format(preprocessed_data_folder),
            "rb",
        )
    )

    key_signatures = pickle.load(
        open("{}/key_signatures.pickle".format(preprocessed_data_folder), "rb")
    )

    model_choices = pickle.load(
        open("{}/model_choices.pickle".format(model_folder), "rb")
    )

    system_choices = pickle.load(
        open("{}/system_choices.pickle".format(preprocessed_data_folder), "rb")
    )

    rhythmic_patterns = pickle.load(
        open(
            "{}/unique_whole_bar_seed_rhythms.pickle".format(preprocessed_data_folder),
            "rb",
        )
    )

    losses = []
    if loss_flag:
        losses.append("loss")
    if val_loss_flag:
        losses.append("val_loss")
    for loss_val_loss in losses:
        loss_results_folder = "{}/{}".format(results_folder, loss_val_loss)
        try:
            os.mkdir(loss_results_folder)
        except FileExistsError:
            pass
        model = load_model(
            model_folder + "/model.h5",
            custom_objects={
                "categorical_focal_loss": categorical_focal_loss,
                "categorical_focal_loss_fixed": dill.loads(
                    dill.dumps(categorical_focal_loss(gamma=2.0, alpha=0.25))
                ),
                "binary_focal_loss": binary_focal_loss,
                "binary_focal_loss_fixed": dill.loads(
                    dill.dumps(binary_focal_loss(gamma=2.0, alpha=0.25))
                ),
            },
        )
        list_of_files = glob.glob(
            "{}/output_weights/{}/*".format(model_folder, loss_val_loss)
        )
        latest_file = max(list_of_files, key=os.path.getctime)
        model.load_weights(latest_file)
        sequence_length = len(preprocessed_end_sequences["Note"][0])
        test_scores_folder = "{}/generated_scores".format(loss_results_folder)
        try:
            os.mkdir(test_scores_folder)
        except FileExistsError:
            pass

        create_confusion_matrices(
            model, X_test, y_test, decodings, test_scores_folder, model_choices
        )

        if end_sequences_flag:
            from_ends_scores = "{}/from_ends".format(test_scores_folder)
            os.mkdir(from_ends_scores)

            print("=" * 50)
            print("Generating {} pieces from end sequences".format(no_generated_pieces))
            print("=" * 50)
            preprocessed_end_sequences = ignore_rest_ends(
                preprocessed_end_sequences, encodings
            )
            end_sequences_count = len(preprocessed_end_sequences["Note"])
            if no_generated_pieces > end_sequences_count:
                raise IndexError(
                    "There are {} available end sequences. You have requested {}.".format(
                        end_sequences_count, no_generated_pieces
                    )
                )
            random_end_seeds = random.sample(
                range(end_sequences_count), no_generated_pieces
            )
            end_seeds = {
                k: [hes[i] for i in random_end_seeds]
                for k, hes in preprocessed_end_sequences.items()
            }

            generate_pieces_from_seeds(
                model,
                system_choices["feature_shapes"],
                end_seeds,
                model_choices,
                ["end_seed_piece_{}".format(i) for i in range(no_generated_pieces)],
                from_ends_scores,
                decodings,
                grade_reqs,
                key_signatures=None,
            )

        if artificial_seeds_flag:
            from_artificial_seeds_scores = test_scores_folder + "/from_artificial_seeds"
            os.mkdir(from_artificial_seeds_scores)

            print("=" * 50)
            print(
                "Generating {} pieces from artificial seeds".format(no_generated_pieces)
            )
            print("=" * 50)

            generate_piece_from_artificial_seeds(
                model,
                system_choices,
                model_choices,
                from_artificial_seeds_scores,
                encodings,
                decodings,
                grade_reqs,
                no_generated_pieces,
                rhythmic_patterns,
            )


def ignore_rest_ends(preprocessed_end_sequences, encodings):
    """Removes seeds ending in rests (could be anacruses)

    Arguments:
        preprocessed_end_sequences {dictionary} -- dictionary of lists of end seeds
        encodings {dictionary} -- dictionary of encodings for each feature

    Returns:
        dictionary -- dictionary of end seeds with seeds ending in rests removed
    """
    rest_ends = []
    for i, _ in enumerate(preprocessed_end_sequences["Note"]):
        if preprocessed_end_sequences["Note"][i][-1][encodings["Note"]["Rest"]] == 1:
            rest_ends.append(i)
    rest_ends = sorted(rest_ends, reverse=True)
    for k in preprocessed_end_sequences.keys():
        preprocessed_end_sequences[k] = np.delete(
            preprocessed_end_sequences[k], rest_ends, axis=0
        )
    return preprocessed_end_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_folder",
        dest="model_folder",
        help="Folder containing model.",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--no_generated_pieces",
        dest="no_generated_pieces",
        help="Number of generated pieces.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-l",
        "--loss_flag",
        help="Generate music from the weights saved at the minimum loss of the network during training.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-v",
        "--val_loss_flag",
        help="Generate music from the weights saved at the minimum validation loss of the network during training.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-e",
        "--end_sequences_flag",
        help="Generate music from end sequence seeds.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-a",
        "--artificial_seeds_flag",
        help="Generate music from artificial seeds.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    model_folder = args.model_folder
    no_generated_pieces = args.no_generated_pieces
    loss_flag = args.loss_flag
    val_loss_flag = args.val_loss_flag
    if not (loss_flag or val_loss_flag):
        loss_flag = True
        val_loss_flag = True
    end_sequences_flag = args.end_sequences_flag
    artificial_seeds_flag = args.artificial_seeds_flag
    if not (end_sequences_flag or artificial_seeds_flag):
        end_sequences_flag = True
        artificial_seeds_flag = True
    if not os.path.isdir(model_folder):
        sys.exit("{} is not a directory".format(model_folder))
    main(
        model_folder,
        no_generated_pieces,
        loss_flag,
        val_loss_flag,
        end_sequences_flag,
        artificial_seeds_flag,
    )
