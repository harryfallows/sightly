import random
import numpy as np
import time


def generate_piece(model, sequence_length, input_seed, piece_length, domain_sizes):
    output_piece = [[] for _ in range(len(domain_sizes))]
    split = 2
    for idx in range(piece_length):
        for i in range(len(input_seed)):
            input_seed[i] = np.reshape(
                input_seed[i], (1, sequence_length, domain_sizes[i])
            )

        prediction = model.predict(input_seed, verbose=0)

        for single in range(len(prediction[:split])):
            single_prediction = np.argmax(prediction[single][0])
            normalised_prediction = [0.0 for i in range(domain_sizes[single])]
            normalised_prediction[single_prediction] = 1.0
            prediction[single] = normalised_prediction
        for multi in range(len(prediction[split:])):
            prediction[multi + split] = [
                1 if mp > 0.5 else 0 for mp in prediction[multi + split][0]
            ]
        for i, p in enumerate(prediction):
            shape = input_seed[i].shape
            input_seed[i] = np.reshape(
                np.concatenate((input_seed[i][0][1:], [p])), shape
            )
            if i < split:
                loc = np.where(np.array(p) == 1.0)
            if len(loc) > 0:
                output_piece[i].append(loc[0][0])
            else:
                loc = np.where(np.array(p) == 1.0)[0]
            if len(loc) > 0:
                output_piece[i].append([i for i in loc])
            else:
                output_piece[i].append([])
    return output_piece


def decode_output(output_piece, decodings):
    decoded_output = []
    for o in output_piece:
        decoded_output.append(decodings[o])
    return decoded_output


def decode_multi_output(output_piece, decodings):
    decoded_output = []
    for o in output_piece:
        decoded_o = []
        for e in o:
            decoded_o.append(decodings[e])
        decoded_output.append(decoded_o)
    return decoded_output


def generate_note(
    model, all_pitches, sequence_length, starting_notes, piece_length, domain_size
):
    prediction_output = []
    output_piece = np.zeros((piece_length, domain_size))
    output_piece[-sequence_length:] = starting_notes
    input_sequence = starting_notes

    input_sequence = np.reshape(input_sequence, (1, sequence_length, domain_size))

    prediction = model.predict(input_sequence, verbose=0)[0]
    prediction = np.argmax(prediction)
    normalised_prediction = [0.0 for i in range(domain_size)]
    normalised_prediction[prediction] = 1.0
    output_piece[0:-1] = output_piece[1:]
    output_piece[-1] = normalised_prediction
    new_input_sequence = output_piece[-sequence_length:]
    input_sequence = new_input_sequence
    output_piece = [np.where(note == 1.0) for note in output_piece]
    return output_piece[-1]
