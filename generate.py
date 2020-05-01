import numpy as np
from post_processing import join_notes, generate_score
from preprocessing import encode_sequences, one_hot_sequences
from music21 import key
import random
import pandas as pd
from random import randrange
from music21 import note
from statistics import mean


def generate_pieces_from_seeds(
    model,
    feature_shapes,
    seeds,
    model_choices,
    files,
    output_folder,
    decodings,
    grade_reqs,
    key_signatures=None,
):
    """Generates pieces of music from a given number of seeds.

    Arguments:
        model {keras model} -- trained model to generate music from
        feature_shapes {list} -- list of tuples defining the input shapes required for each feature
        seeds {list} -- list of nd numpy lists representing input seeds for network
        model_choices {dictionary} -- options chosen for model
        files {list} -- list of strings to saved files under
        output_folder {string} -- folder to save generated music to
        decodings {dictionary} -- decodings for each feature
        grade_reqs {dictionary} -- quantitative grade requirements

    Keyword Arguments:
        key_signatures {list} -- key signatures to apply to each generated piece (default: {None})
    """
    number_of_bars = int(mean(grade_reqs["length"]))
    time_signature = model_choices["time_signature"]
    single_label_features = model_choices["single_label_features"]
    piece_length = int(
        number_of_bars * (16 / int(time_signature[-1])) * int(time_signature[0])
    )
    for idx, _ in enumerate(seeds["Note"]):
        input_seq = {k: v[idx] for k, v in seeds.items()}
        if key_signatures is not None:
            key_sig = key_signatures[idx]
        else:
            key_sig = None

        prediction = generate_piece(
            model,
            input_seq,
            piece_length,
            feature_shapes,
            single_label_features,
            model_choices["obj_features"],
        )
        decoded_prediction = {}
        for k, v in prediction.items():
            if k in single_label_features:
                decoded_prediction.update({k: decode_output(v, decodings[k])})
            else:
                decoded_prediction.update({k: decode_multi_output(v, decodings[k])})

        decoded_text_expressions = []
        if "TextExpression" in decoded_prediction.keys():
            decoded_text_expressions = decoded_prediction["TextExpression"].copy()
            del decoded_prediction["TextExpression"]
        joined_notes = join_notes(
            decoded_prediction,
            grade_reqs["quarter_lengths"],
            grade_reqs["rest_lengths"],
            time_signature=(int(time_signature[0]), int(time_signature[-1])),
        )

        generate_score(
            joined_notes,
            output_folder,
            text_expressions=decoded_text_expressions,
            file_name=files[idx],
            key_signature=key_sig,
            time_signature=time_signature,
        )


def generate_piece_from_artificial_seeds(
    model,
    system_choices,
    model_choices,
    output_folder,
    encodings,
    decodings,
    grade_reqs,
    no_generated_pieces,
    rhythmic_patterns,
):
    """Generate piece of music from artificial seed. Generates an artificial
    seed, then calls the generate_piece_from_seeds function.

    Arguments:
        model {keras model} -- trained model to generate music from
        system_choices {dictionary} -- dictionary containing choices about the system e.g. context length
        model_choices {dictionary} -- options chosen for model
        output_folder {string} -- folder to save generated music to
        encodings {dictionary} -- encodings for each feature
        decodings {dictionary} -- decodings for each feature
        grade_reqs {dictionary} -- quantitative grade requirements
        no_generated_pieces {int} -- number of pieces to be generated
        rhythmic_patterns {dictionary} -- rhythmic and slur patterns from dataset analysis
    """
    seeds = {
        k: np.zeros(tuple([no_generated_pieces] + list(shape)))
        for k, shape in system_choices["feature_shapes"].items()
        if k in model_choices["obj_features"]
    }
    key_signatures = []
    files = []
    note_seeds = []
    dynamic_seeds = []
    slur_seeds = []
    s = sum(rhythmic_patterns.values())
    unique_rhythms_distribution = {k: (v / s) for k, v in rhythmic_patterns.items()}
    rhythms = []
    rhythms_probabilities = []
    for k, v in unique_rhythms_distribution.items():
        rhythms.append(k)
        rhythms_probabilities.append(v)
    rhythms_numbered = {k: v for k, v in enumerate(rhythms)}
    chosen_bar_rhythms = np.random.choice(
        list(rhythms_numbered.keys()),
        no_generated_pieces,
        replace=True,
        p=rhythms_probabilities,
    )
    chosen_bar_rhythms = [rhythms_numbered[i] for i in chosen_bar_rhythms]
    for piece in range(no_generated_pieces):
        chosen_key = key.Key(random.choice(grade_reqs["keys"]))
        key_signatures.append(chosen_key.sharps)
        key_object = chosen_key.getScale(
            mode="minor" if str(chosen_key).islower() else "major"
        )
        notes_in_key = [
            str(p)
            for p in key_object.getPitches(
                str(grade_reqs["pitch_range"][0]), str(grade_reqs["pitch_range"][1])
            )
        ]
        notes = []
        last_n = 0
        for n in chosen_bar_rhythms[piece][0]:
            if n != last_n:
                last_note = random.choice(notes_in_key)
                while last_note not in encodings["Note"].keys():
                    last_note = random.choice(notes_in_key)
            notes.append(last_note)
            last_n = n
        note_seeds.append(notes)
        dynamic_seeds.append(
            [randrange(system_choices["feature_shapes"]["Dynamic"][1])]
            * system_choices["sequence_length"]
        )
        slur_seeds.append(list(chosen_bar_rhythms[piece][1]))
        files.append("artificial_{}".format(piece))
    encoded_note_seeds, _ = encode_sequences(np.array(note_seeds), encodings["Note"])
    one_hot_note_seeds, _ = one_hot_sequences(
        encoded_note_seeds, system_choices["feature_shapes"]["Note"][1]
    )
    one_hot_dynamic_seeds, _ = one_hot_sequences(
        np.array(dynamic_seeds), system_choices["feature_shapes"]["Dynamic"][1]
    )
    seeds["Note"] = one_hot_note_seeds
    try:
        seeds["Dynamic"] = one_hot_dynamic_seeds
    except KeyError:
        pass

    try:
        seeds["Slur"] = np.array([s for s in slur_seeds])
    except KeyError:
        pass
    print({k: v.shape for k, v in seeds.items()})
    generate_pieces_from_seeds(
        model,
        system_choices["feature_shapes"],
        seeds,
        model_choices,
        files,
        output_folder,
        decodings,
        grade_reqs,
        key_signatures=None,
    )


def generate_piece(
    model, input_seed, piece_length, feature_shapes, single_label_features, obj_features
):
    """Generates a piece of music from a seed.

    Arguments:
        model {keras model} -- model to generate music using
        input_seed {nd numpy array} -- initial seed for network
        piece_length {int} -- length of desired piece in time-steps
        feature_shapes {dictionary} -- dictionary of tuples showing shapes of inputs required for each feature
        single_label_features {set} -- single label features (Notes, Dynamics)
        obj_features {set} -- all features in network

    Returns:
        nd numpy array -- generated piece in matrix form
    """
    output_piece = {k: [] for k in obj_features}
    for _ in range(piece_length):
        for k in input_seed.keys():
            input_seed[k] = np.reshape(
                input_seed[k], (1, feature_shapes[k][0], feature_shapes[k][1])
            )
        prediction = model.predict(input_seed, verbose=0)
        if len(model.output_names) != 1:
            prediction = {k[:-2]: v for k, v in zip(model.output_names, prediction)}
        else:
            prediction = {"Note": prediction}
        for k in single_label_features:
            single_prediction = np.argmax(prediction[k][0])
            normalised_prediction = [0.0 for i in range(feature_shapes[k][1])]
            normalised_prediction[single_prediction] = 1.0
            prediction[k] = normalised_prediction
        for k in obj_features:
            if k in single_label_features:
                continue
            prediction[k] = [1 if mp > 0.5 else 0 for mp in prediction[k][0]]
        for k, v in prediction.items():
            shape = feature_shapes[k]
            input_seed[k] = np.reshape(
                np.concatenate((input_seed[k][0][1:], [v])), shape
            )
            if k in single_label_features:
                loc = np.where(np.array(v) == 1.0)
                if len(loc) > 0:
                    output_piece[k].append(loc[0][0])
            else:
                loc = np.where(np.array(v) == 1.0)[0]
                if len(loc) > 0:
                    output_piece[k].append([i for i in loc])
                else:
                    output_piece[k].append([])
    return output_piece


def decode_output(output_piece, decodings):
    """Decodes a single-label feature from the output piece.

    Arguments:
        output_piece {list} -- encoded values of predicted piece (for one feature)
        decodings {dictionary} -- features decodings

    Returns:
        list -- decoded output
    """
    decoded_output = []
    for o in output_piece:
        decoded_output.append(decodings[o])
    return decoded_output


def decode_multi_output(output_piece, decodings):
    """Decodes a multi-label feature from the output piece.

    Arguments:
        output_piece {list} -- encoded values of predicted piece (for one feature)
        decodings {dictionary} -- features decodings

    Returns:
        list -- decoded output
    """
    decoded_output = []
    for o in output_piece:
        decoded_o = []
        for e in o:
            decoded_o.append(decodings[e])
        decoded_output.append(decoded_o)
    return decoded_output
