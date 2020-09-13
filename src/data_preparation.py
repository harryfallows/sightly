from music21 import note, dynamics, expressions, articulations, spanner, key
from preprocessing import (
    read_mxl,
    get_notes,
    pitch_only,
    get_dynamics,
    get_spanners,
    get_slurs,
    get_articulations,
    get_text_expressions,
    get_sequences,
    get_sequences_transposed,
    transpositions_for_spanners,
    encode_data,
    one_hot_sequences,
    encode_sequences,
    multi_hot_sequences,
    encode_multi_sequences,
    flatten_sequence_structure,
)
import pickle
import sys
import os
from sklearn.model_selection import train_test_split
import numpy as np
from statistics import mean
import json
import argparse
from context_bar_rhythms import get_context_bar_rhythms

def filter_data_by_time_signature(data_folder, allowed_time_sigs):
    """Separates pieces of music by their time signatures.

    Arguments:
        data_folder {string} -- folder containing data
        allowed_time_sigs {list} -- time signatures allowed in grade

    Returns:
        dictionary -- keys: time signatures, values: music parts, key signatures and file
    """

    parts, time_signatures, key_signatures, filenames = read_mxl(data_folder)
    ts_separated_parts = {
        ts: {"parts": [], "key_signatures": [], "filenames": []}
        for ts in allowed_time_sigs
    }
    for i, ts in enumerate(time_signatures):
        if ts in allowed_time_sigs:
            ts_separated_parts[ts]["parts"].append(parts[i])
            ts_separated_parts[ts]["key_signatures"].append(key_signatures[i])
            ts_separated_parts[ts]["filenames"].append(filenames[i])
    return ts_separated_parts

def preprocess_data(pickle_folder, ts_parts_dict, time_signature, requirements):
    """Takes data in a musical represenation and processes it into something
    that can be input into the neural network. This includes splitting the data
    into training and test data.

    Arguments:
        pickle_folder {string} -- folder for preprocessed data
        ts_parts_dict {dictionary} --
        time_signature {[type]} -- [description]
        requirements {[type]} -- [description]
    """
    bar_length = int(int(time_signature[0]) / int(time_signature[-1]) * 16)
    sequence_length = bars_context * bar_length

    # extract parts, key signatures and filenames from time signature split dataset
    parts = ts_parts_dict["parts"]
    parts_key_signatures = ts_parts_dict["key_signatures"]
    parts_filenames = ts_parts_dict["filenames"]

    # get notes and other musical aspects from pieces
    dump_folder = "{}/{}-{}".format(
        pickle_folder, time_signature[0], time_signature[-1]
    )
    os.mkdir(dump_folder)

    pickle.dump(parts_filenames, open("{}/filenames.pickle".format(dump_folder), "wb"))
    pieces, all_pitches, _ = get_notes(parts)
    pieces_slurs = get_slurs(parts)
    pieces_text_expressions, all_text_expressions = get_text_expressions(parts)
    pieces_dynamics, all_dynamics = get_dynamics(parts)
    pieces_articulations, all_articulations = get_articulations(parts)
    pieces_spanners, all_spanners = get_spanners(parts)

    # get encodings for all musical aspects
    dynamics_encodings, dynamics_decodings = encode_data(all_dynamics)
    text_expression_encodings, text_expression_decodings = encode_data(
        all_text_expressions
    )
    articulation_encodings, articulation_decodings = encode_data(all_articulations)
    spanner_encodings, spanner_decodings = encode_data(all_spanners)
    pieces = pitch_only(pieces)

    # get inputs and outputs for each musical feature
    (
        dynamics_input_sequences,
        dynamics_outputs,
        dynamics_end_sequences,
    ) = get_sequences(pieces_dynamics, sequence_length)
    (
        text_expression_input_sequences,
        text_expression_outputs,
        text_expression_end_sequences,
    ) = get_sequences(pieces_text_expressions, sequence_length)
    (
        articulation_input_sequences,
        articulation_outputs,
        articulation_end_sequences,
    ) = get_sequences(pieces_articulations, sequence_length)
    (spanner_input_sequences, spanner_outputs, spanner_end_sequences,) = get_sequences(
        pieces_spanners, sequence_length
    )
    slur_input_sequences, slur_outputs, slur_end_sequences = get_sequences(
        pieces_slurs, sequence_length
    )

    if data_augmentation_flag:
        (
            input_sequences,
            outputs,
            end_sequences,
            sequence_key_signatures,
            all_pitches,
        ) = get_sequences_transposed(
            pieces,
            sequence_length,
            parts_key_signatures,
            note.Note(requirements["pitch_range"][1]),
            note.Note(requirements["pitch_range"][0]),
            requirements["key_signatures"],
            set(all_pitches),
        )
    else:
        (input_sequences, outputs, end_sequences) = get_sequences(
            pieces, sequence_length
        )
        sequence_key_signatures = [[k] for k in parts_key_signatures]
        input_sequences = [[[j] for j in i] for i in input_sequences]
        outputs = [[[j] for j in i] for i in outputs]
        end_sequences = [[i] for i in end_sequences]

    pickle.dump(
        sequence_key_signatures,
        open("{}/key_signatures.pickle".format(dump_folder), "wb"),
    )
    pitch_encodings, pitch_decodings = encode_data(all_pitches)

    # concatenate inputs, outputs, encodings, decodings from single label classification features (notes and dynamics)
    single_label_input_sequences = {
        "Note": input_sequences,
        "Dynamic": dynamics_input_sequences,
    }
    single_label_outputs = {"Note": outputs, "Dynamic": dynamics_outputs}

    single_label_end_sequences = {
        "Note": [[es] for es in end_sequences],
        "Dynamic": [[es] for es in dynamics_end_sequences],
    }
    single_label_encodings = {
        "Note": pitch_encodings,
        "Dynamic": dynamics_encodings,
    }
    single_label_decodings = {
        "Note": pitch_decodings,
        "Dynamic": dynamics_decodings,
    }

    # concatenate inputs, outputs, encodings, decodings from multi label classification features (text expressions, articulations, spanners, slurs)
    multi_label_input_sequences = {
        "TextExpression": text_expression_input_sequences,
        "Articulation": articulation_input_sequences,
        "Spanner": spanner_input_sequences,
        "Slur": slur_input_sequences,
    }
    multi_label_outputs = {
        "TextExpression": text_expression_outputs,
        "Articulation": articulation_outputs,
        "Spanner": spanner_outputs,
        "Slur": slur_outputs,
    }
    multi_label_end_sequences = {
        "TextExpression": [[es] for es in text_expression_end_sequences],
        "Articulation": [[es] for es in articulation_end_sequences],
        "Spanner": [[es] for es in spanner_end_sequences],
        "Slur": [[es] for es in slur_end_sequences],
    }
    multi_label_encodings = {
        "TextExpression": text_expression_encodings,
        "Articulation": articulation_encodings,
        "Spanner": spanner_encodings,
        "Slur": {spanner.Slur: 0},
    }
    multi_label_decodings = {
        "TextExpression": text_expression_decodings,
        "Articulation": articulation_decodings,
        "Spanner": spanner_decodings,
        "Slur": {0: spanner.Slur},
    }

    X_train = {}
    X_test = {}
    X_val = {}
    y_train = {}
    y_test = {}
    y_val = {}
    preprocessed_end_sequences = {}

    for single_label_object in single_label_input_sequences.keys():
        input_sequences = single_label_input_sequences[single_label_object]
        outputs = single_label_outputs[single_label_object]
        end_sequences = single_label_end_sequences[single_label_object]
        if single_label_object is not "Note":
            input_sequences, outputs = transpositions_for_spanners(
                single_label_input_sequences["Note"],
                input_sequences,
                outputs=single_label_outputs["Note"],
                spanner_outputs=outputs,
            )
            end_sequences, _ = transpositions_for_spanners(
                single_label_end_sequences["Note"], end_sequences
            )
        class_X, class_y = flatten_sequence_structure(input_sequences, outputs)
        class_end_X, _ = flatten_sequence_structure(end_sequences)
        encodings = single_label_encodings[single_label_object]
        encoded_input_seqs, encoded_outputs = encode_sequences(
            class_X, encodings, class_y
        )
        encoded_end_seqs, _ = encode_sequences(class_end_X, encodings)

        one_hot_input_seqs, one_hot_outputs = one_hot_sequences(
            encoded_input_seqs, len(encodings), encoded_outputs
        )
        one_hot_end_seqs, _ = one_hot_sequences(encoded_end_seqs, len(encodings))

        class_X_train, class_X_test, class_y_train, class_y_test = train_test_split(
            one_hot_input_seqs,
            one_hot_outputs,
            test_size=0.2,
            random_state=42,
            shuffle=False,
        )
        class_X_test, class_X_val, class_y_test, class_y_val = train_test_split(
            class_X_test, class_y_test, test_size=0.5, random_state=42, shuffle=True,
        )
        preprocessed_end_sequences.update({single_label_object: one_hot_end_seqs})
        X_train.update({single_label_object: class_X_train})
        X_test.update({single_label_object: class_X_test})
        X_val.update({single_label_object: class_X_val})
        y_train.update({single_label_object: class_y_train})
        y_test.update({single_label_object: class_y_test})
        y_val.update({single_label_object: class_y_val})

    for multi_label_object in multi_label_input_sequences.keys():
        input_sequences = multi_label_input_sequences[multi_label_object]
        outputs = multi_label_outputs[multi_label_object]
        end_sequences = multi_label_end_sequences[multi_label_object]
        input_sequences, outputs = transpositions_for_spanners(
            single_label_input_sequences["Note"],
            input_sequences,
            outputs=single_label_outputs["Note"],
            spanner_outputs=outputs,
        )
        end_sequences, _ = transpositions_for_spanners(
            single_label_end_sequences["Note"], end_sequences
        )
        class_X, class_y = flatten_sequence_structure(input_sequences, outputs)
        class_end_X, _ = flatten_sequence_structure(end_sequences)
        encodings = multi_label_encodings[multi_label_object]
        encoded_input_seqs, encoded_outputs = encode_multi_sequences(
            class_X, encodings, class_y
        )
        encoded_end_seqs, _ = encode_multi_sequences(class_end_X, encodings)
        multi_hot_input_seqs, multi_hot_outputs = multi_hot_sequences(
            encoded_input_seqs, len(encodings), encoded_outputs
        )
        multi_hot_end_seqs, _ = multi_hot_sequences(encoded_end_seqs, len(encodings))
        class_X_train, class_X_test, class_y_train, class_y_test = train_test_split(
            multi_hot_input_seqs,
            multi_hot_outputs,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )
        class_X_test, class_X_val, class_y_test, class_y_val = train_test_split(
            class_X_test, class_y_test, test_size=0.5, random_state=42, shuffle=True,
        )
        preprocessed_end_sequences.update({multi_label_object: multi_hot_end_seqs})
        X_train.update({multi_label_object: class_X_train})
        X_test.update({multi_label_object: class_X_test})
        X_val.update({multi_label_object: class_X_val})
        y_train.update({multi_label_object: class_y_train})
        y_test.update({multi_label_object: class_y_test})
        y_val.update({multi_label_object: class_y_val})
    pickle.dump(
        preprocessed_end_sequences,
        open("{}/preprocessed_end_sequences.pickle".format(dump_folder), "wb"),
    )
    pickle.dump(X_train, open("{}/X_train.pickle".format(dump_folder), "wb"))
    pickle.dump(X_test, open("{}/X_test.pickle".format(dump_folder), "wb"))
    pickle.dump(X_val, open("{}/X_val.pickle".format(dump_folder), "wb"))
    pickle.dump(y_train, open("{}/y_train.pickle".format(dump_folder), "wb"))
    pickle.dump(y_test, open("{}/y_test.pickle".format(dump_folder), "wb"))
    pickle.dump(y_val, open("{}/y_val.pickle".format(dump_folder), "wb"))
    end_sequences = {**single_label_end_sequences, **multi_label_end_sequences}
    pickle.dump(
        {**single_label_end_sequences, **multi_label_end_sequences},
        open("{}/end_sequences.pickle".format(dump_folder), "wb"),
    )
    input_sequences = {**single_label_input_sequences, **multi_label_input_sequences}
    pickle.dump(
        {**single_label_input_sequences, **multi_label_input_sequences},
        open("{}/input_sequences.pickle".format(dump_folder), "wb"),
    )
    pickle.dump(
        {**single_label_encodings, **multi_label_encodings},
        open("{}/encodings.pickle".format(dump_folder), "wb"),
    )
    pickle.dump(
        {**single_label_decodings, **multi_label_decodings},
        open("{}/decodings.pickle".format(dump_folder), "wb"),
    )

    unique_rhythms = get_context_bar_rhythms(input_sequences, end_sequences, bar_length)
    pickle.dump(
        unique_rhythms,
        open("{}/unique_whole_bar_seed_rhythms.pickle".format(dump_folder), "wb"),
    )

    pickle.dump(
        {
            "bar_length": bar_length,
            "sequence_length": sequence_length,
            "feature_shapes": {k: v.shape[1:] for k, v in X_train.items()},
        },
        open("{}/system_choices.pickle".format(dump_folder), "wb"),
    )


def main(folder, time_signatures):
    """driver code.

    Arguments:
        folder {string} -- folder containing requirements and data
    """
    preprocessed_data_folder = "{}/preprocessed_data{}".format(
        folder, "_augmented" if data_augmentation_flag else ""
    )
    try:
        os.mkdir(preprocessed_data_folder)
    except FileExistsError:
        pass
    requirements = json.load(open("{}/requirements.json".format(folder)))
    requirements["key_signatures"] = [
        key.KeySignature(k) for k in requirements["key_signatures"]
    ]
    ts_separated_parts = filter_data_by_time_signature(
        folder + "/data", requirements["time_signatures"]
    )
    if time_signatures == []:
        time_signatures = [k for k in ts_separated_parts.keys()]
    for ts, p in ts_separated_parts.items():
        if ts in time_signatures:
            preprocess_data(preprocessed_data_folder, p, ts, requirements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        help="Folder containing requirements (requirements.json) and data (data/).",
        required=True,
    )
    parser.add_argument(
        "--time_signatures",
        dest="time_signatures",
        default=[],
        help="Time signatures to create data for.",
        nargs="+",
    )
    parser.add_argument(
        "-a",
        "--dont_augment_data",
        dest="dont_augment_data",
        default=True,
        help="Don't augment data flag. -a for don't augment, nothing for augment.",
        action="store_false",
    )
    parser.add_argument(
        "-b",
        "--bars_context",
        dest="bars_context",
        default=4,
        help="Length of musical context given to the neural network.",
        type=int,
    )
    args = parser.parse_args()
    folder = args.folder
    global bars_context
    bars_context = args.bars_context
    global data_augmentation_flag
    data_augmentation_flag = args.dont_augment_data
    time_signatures = args.time_signatures
    main(folder, time_signatures)
