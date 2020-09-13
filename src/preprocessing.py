from music21 import converter, instrument, note, chord, stream, pitch, spanner, tie, key
import os
import numpy as np
from fractions import Fraction
import math


def read_mxl(filepath):
    """read all of the musicxml files in the dataset.

    Arguments:
        filepath {string} -- name of the parent folder

    Returns:
        tuple of lists -- (parts: all of the parts found, time_signatures: time signature of each part, key_signatures: key signature for each part, files: names of all files)
    """
    parts = []
    time_signatures = []
    key_signatures = []
    files = []
    for i, file in enumerate(os.listdir(filepath)):
        if ".mxl" not in file and ".xml" not in file:
            continue
        print("Parsing file {}: {}".format(i, str(file)))
        piece = converter.parse(filepath + "/" + file)
        instrument_parts = instrument.partitionByInstrument(piece)
        violin_part = instrument_parts[0]
        parts.append(violin_part)
        time_signatures.append(violin_part.timeSignature.ratioString)
        try:
            key_signatures.append(violin_part.keySignature.sharps)
        except AttributeError:
            key_signatures.append(0)
        files.append(file)
    return parts, time_signatures, key_signatures, files


def get_notes(parts, single_octave=False):
    """Gets the notes from each part.

    Arguments:
        parts {list} -- list of parts

    Keyword Arguments:
        single_octave {bool} -- whether to account for octaves or remain as a single octave (default: {False})

    Returns:
        tuple of lists -- (pieces: list of pieces converted to lists of notes, all of the pitches found, all of the quarter_lengths found)
    """
    pieces = []
    all_pitches = set()
    all_quarterLengths = set()
    for p in parts:
        part = p.recurse()
        notes = []
        for n in part:
            if isinstance(n, note.Note):
                n_pitch = str(pitch.Pitch(str(n.pitch)))
                quarterLength = n.quarterLength
            elif isinstance(n, chord.Chord):
                n_pitch = str(pitch.Pitch(str(n.root())))
                quarterLength = n.quarterLength
            elif isinstance(n, note.Rest):
                n_pitch = "Rest"
                quarterLength = n.quarterLength
            else:
                continue
            try:
                if quarterLength == 0:
                    continue
                if single_octave and n_pitch != "Rest":
                    n_pitch = n_pitch[:-1]
                all_pitches.add(n_pitch)
                all_quarterLengths.add(quarterLength)
                notes.append([n_pitch, quarterLength])
            except UnboundLocalError:
                pass
        pieces.append(notes)
    return pieces, sorted_pitches(all_pitches), sorted(all_quarterLengths)


def get_dynamics(part_streams):
    """Retrieves all discrete dynamics found in all pieces.

    Arguments:
        part_streams {list of music21 streams} -- list of music21 streams representing each piece of music found in the dataset

    Returns:
        tuple -- (all dynamics found in parts {list}, all dynamics found {set})
    """
    dynamics_in_parts = []
    all_dynamics = set([])
    for ps in part_streams:
        ps_dynamic = [0] * int(ps.duration.quarterLength * 4)
        for dynamic in ps.getElementsByClass("Dynamic"):
            all_dynamics.add(dynamic.value)
            when = int(dynamic.offset * 4)
            if ps_dynamic[0] == 0:
                when = 0
            ps_dynamic[when:] = [dynamic.value] * (len(ps_dynamic) - when)
        dynamics_in_parts.append(ps_dynamic)
    return dynamics_in_parts, all_dynamics


def get_text_expressions(part_streams):
    """Retrieves all text expressions found in all pieces.

    Arguments:
        part_streams {list of music21 streams} -- list of music21 streams representing each piece of music found in the dataset

    Returns:
        tuple -- (all text expressions found in parts {list}, all text expressions found {set})
    """
    text_in_parts = []
    all_text_expressions = set([])
    for ps in part_streams:
        ps_text = [[]] * int(ps.duration.quarterLength * 4)
        for text in ps.getElementsByClass("TextExpression"):
            all_text_expressions.add(text.content)
            when = int(text.offset * 4)
            ps_text[when] = ps_text[when] + [text.content]
        text_in_parts.append(ps_text)
    return text_in_parts, all_text_expressions


def get_slurs(part_streams):
    """Retrieves all slurs and ties found in all pieces.

    Arguments:
        part_streams {list of music21 streams} -- list of music21 streams representing each piece of music found in the dataset

    Returns:
        tuple -- (all slurs found in parts {list}, all slurs found {set})
    """
    slurs_in_parts = []
    for ps in part_streams:
        ps_slur = [[]] * int(ps.duration.quarterLength * 4)
        for slur_obj in ps.getElementsByClass("Slur"):
            start = int(slur_obj.getFirst().offset * 4)
            end = int(slur_obj.getLast().offset * 4)
            for t in range(start, end):
                ps_slur[t] = [spanner.Slur]
        for note in ps.getElementsByClass("Note"):
            if note.tie is not None:
                if note.tie.type in ["start", "continue"]:
                    start = int(note.offset * 4)
                    end = start + int(note.quarterLength * 4)
                    for t in range(start, end):
                        ps_slur[t] = [spanner.Slur]
        slurs_in_parts.append(ps_slur)
    return slurs_in_parts


def get_spanners(part_streams):
    """Retrieves all spanners found in all pieces.

    Arguments:
        part_streams {list of music21 streams} -- list of music21 streams representing each piece of music found in the dataset

    Returns:
        tuple -- (all spanners found in parts {list}, all spanners found {set})
    """
    spanner_in_parts = []
    all_spanners = set([])
    for ps in part_streams:
        ps_spanner = [[]] * int(ps.duration.quarterLength * 4)
        for spanner_obj in ps.getElementsByClass("Spanner"):
            if isinstance(spanner_obj, spanner.Slur):
                continue
            all_spanners.add(type(spanner_obj))
            start = int(spanner_obj.getFirst().offset * 4)
            end = int(spanner_obj.getLast().offset * 4)
            for t in range(start, end):
                ps_spanner[t] = ps_spanner[t] + [type(spanner_obj)]
        spanner_in_parts.append(ps_spanner)
    return spanner_in_parts, all_spanners


def get_articulations(part_streams):
    """Retrieves all note expressions and articulations found in all pieces.

    Arguments:
        part_streams {list of music21 streams} -- list of music21 streams representing each piece of music found in the dataset

    Returns:
        tuple -- (all note expressions and articulations found in parts {list}, all note expressions and articulations found {set})
    """
    articulations_in_parts = []
    all_articulations = set([])
    for ps in part_streams:
        ps_articulations = [[]] * int(ps.duration.quarterLength * 4)
        for note in ps.getElementsByClass("Note"):
            articulations = [type(a) for a in note.articulations + note.expressions]
            for a in articulations:
                all_articulations.add(a)
            start = int(note.offset * 4)
            end = int((note.offset * 4) + int(note.duration.quarterLength * 4))
            for t in range(start, end):
                ps_articulations[t] = ps_articulations[t] + articulations

        articulations_in_parts.append(ps_articulations)
    return articulations_in_parts, all_articulations


def sorted_pitches(pitches):
    """Sorts an list of pitches.

    Arguments:
        pitches {list-like} -- list of pitch names

    Returns:
        list -- pitch names sorted
    """
    pitches.remove("Rest")
    pitches_as_notes = [note.Note(p) for p in pitches]
    return [str(n.pitch) for n in sorted(pitches_as_notes)] + ["Rest"]


def get_sequences(pieces, sequence_length):
    """Gets musical feature inputs and outputs for network from a piece of
    music.

    Arguments:
        pieces {list} -- list of music pieces
        sequence_length {integer} -- length of network context/input

    Returns:
        tuple -- (network inputs, network outputs, end sequence of each piece)
    """
    input_sequences = []
    outputs = []
    end_sequences = []
    for _, ps in enumerate(pieces):
        piece_inp_seqs = []
        piece_outs = []
        if len(ps) < sequence_length + 1:
            continue
        for seq_start in range(0, len(ps) - sequence_length):
            piece_inp_seqs.append(ps[seq_start : seq_start + sequence_length])
            piece_outs.append(ps[seq_start + sequence_length])
        input_sequences.append(piece_inp_seqs)
        outputs.append(piece_outs)
        end_sequences.append(ps[seq_start + 1 :])
    return input_sequences, outputs, end_sequences


def get_sequences_transposed(
    pieces,
    sequence_length,
    key_signatures,
    highest_note,
    lowest_note,
    legal_key_signatures,
    all_pitches,
):
    """Gets notes inputs and outputs for network from a piece of music and
    transposes each of them to increase the dataset.

    Arguments:
        pieces {list} -- list of music pieces
        sequence_length {integer} -- length of network context/input music
        highest_note {string} -- highest note allowed in grade
        legal_key_signatures {list} -- list of key signatures allowed in grade
        lowest_note {list} -- lowest note allowed in grade
        legal_key_signatures {list} -- list of key signatures allowed in grade
        all_pitches {set} -- all note pitches in dataset

    Returns:
        tuple -- (input_sequences: network inputs, outputs: network outputs, end_sequences: end sequence of each piece, sequence_key_signatures: new key signature of each transposed sequence, all_pitches: all note pitches in dataset)
    """
    sequence_key_signatures = []
    input_sequences = []
    outputs = []
    end_sequences = []
    real_sequences = 0
    for i, p in enumerate(pieces):
        print(
            "Getting sequences for piece {} out of {}. Found {} suitable sequences so far out of {} real sequences.".format(
                i + 1,
                len(pieces),
                sum([sum([len(k) for k in j]) for j in input_sequences]),
                real_sequences,
            )
        )
        if len(p) < sequence_length + 1:
            continue
        piece_inp_seqs = []
        piece_outs = []
        piece_key_sigs = []
        for seq_start in range(0, len(p) - sequence_length):
            seq_inp_seqs = []
            seq_outs = []
            real_sequences += 1
            transposed_sequences, transposed_key_sigs = transpose_sequence(
                p[seq_start : seq_start + sequence_length + 1],
                key_signatures[i],
                highest_note,
                lowest_note,
                legal_key_signatures,
            )
            for seq in transposed_sequences:
                for n in seq:
                    all_pitches.add(n)
                seq_inp_seqs.append(seq[:-1])
                seq_outs.append(seq[-1])
            piece_inp_seqs.append(seq_inp_seqs)
            piece_outs.append(seq_outs)
            piece_key_sigs.append(transposed_key_sigs)
        input_sequences.append(piece_inp_seqs)
        outputs.append(piece_outs)
        sequence_key_signatures.append(piece_key_sigs)
        end_sequences.append([seq[1:] for seq in transposed_sequences])
    return input_sequences, outputs, end_sequences, sequence_key_signatures, all_pitches


def transpose_sequence(
    sequence, key_sig, highest_note, lowest_note, legal_key_signatures
):
    """Transpose each sequence into all possible transpositions that still fit
    the grade requirements.

    Arguments:
        sequence {list} -- list of notes in sequence
        key_sig {int} -- key signature of piece the sequence is from
        highest_note {string} -- highest note allowed in grade
        lowest_note {string} -- lowest note allowed in grade
        legal_key_signatures {list} -- list of key signatures allowed in grade

    Returns:
        tuple -- (transposed_sequences: all transpositions of sequence, transposed_key_sigs: transposed keys for each transposed sequence)
    """
    sequence_as_objs = [
        note.Note(n) if n != "Rest" else note.Rest() for i, n in enumerate(sequence)
    ]
    just_notes = [n for n in sequence_as_objs if not isinstance(n, note.Rest)]
    max_note = max(just_notes)
    min_note = min(just_notes)
    key_sig_obj = key.KeySignature(key_sig)
    transposed_sequences = [sequence]
    transposed_key_sigs = [key_sig_obj]
    steps_transposed = 0

    while max_note < highest_note:
        steps_transposed += 1
        max_note = max_note.transpose(1)
        transposed_key_sig = key_sig_obj.transpose(steps_transposed)
        if transposed_key_sig in legal_key_signatures:
            transposed_seq = [
                str(n.transpose(steps_transposed).pitch)
                if not isinstance(n, note.Rest)
                else "Rest"
                for n in sequence_as_objs
            ]
            transposed_sequences.append(transposed_seq)
            transposed_key_sigs.append(transposed_key_sig)

    steps_transposed = 0
    while min_note > lowest_note:
        steps_transposed -= 1
        min_note = min_note.transpose(-1)
        transposed_key_sig = key_sig_obj.transpose(steps_transposed)
        if transposed_key_sig in legal_key_signatures:
            transposed_seq = [
                str(n.transpose(steps_transposed).pitch)
                if not isinstance(n, note.Rest)
                else "Rest"
                for n in sequence_as_objs
            ]
            transposed_sequences.append(transposed_seq)
            transposed_key_sigs.append(transposed_key_sig)
    return transposed_sequences, transposed_key_sigs


def transpositions_for_spanners(
    input_sequences, spanner_input_sequences, outputs=None, spanner_outputs=None
):
    """Copies the spanners format into the same format as the transposed input
    sequences, i.e. duplicates some of the slurring.

    Arguments:
        input_sequences {list} -- note network inputs
        outputs {list} -- note network outputs
        spanner_input_sequences {list} -- musical feature network inputs
        spanner_outputs {list} -- musical feature network outputs

    Returns:
        tuple -- (spanner_input_sequences: duplicated spanner input sequences, spanner_outputs: duplicates spanner outputs)
    """
    for i, p in enumerate(input_sequences):
        for j, s in enumerate(p):
            spanner_input_sequences[i][j] = [spanner_input_sequences[i][j]] * len(s)
            if outputs is not None:
                spanner_outputs[i][j] = [spanner_outputs[i][j]] * len(s)
    return spanner_input_sequences, spanner_outputs


def flatten_sequence_structure(input_sequences, outputs=None):
    """Currently the sequences are structures such that they retain knowledge
    of the piece, sequence and transposition they come from this functions
    creates one list of sequences. input shape: (pieces, sequences,
    transposed_sequences) output shape: (transposed_sequences)

    Arguments:
        input_sequences {list} -- musical feature network inputs
        outputs {list} -- musical feature outputs

    Returns:
        tuple -- (np_flatten_input_sequences: 2d numpy array of input sequences, np_flatten_outputs: 1d numpy array of outputs)
    """
    flattened_input_sequences = []
    flattened_outputs = []
    for i, p in enumerate(input_sequences):
        for j, s in enumerate(list(p)):
            for k, ts in enumerate(list(s)):
                flattened_input_sequences.append(ts)
                if outputs is not None:
                    flattened_outputs.append(outputs[i][j][k])
    np_flatten_input_sequences = np.array(flattened_input_sequences)
    np_flatten_outputs = np.array(flattened_outputs)
    return np_flatten_input_sequences, np_flatten_outputs


def pitch_only(pieces):
    """Duplicates notes to represent timesteps.

    Arguments:
        pieces {list} -- list of all pieces

    Raises:
        ValueError: if a piece containing a triplet is found

    Returns:
        list -- list of pieces split into timesteps
    """
    pitch_only_pieces = []
    for p in pieces:
        notes = []
        for n in p:
            qL = n[1]
            if isinstance(qL, Fraction):
                raise ValueError("System cannot handle triplet notes.")
            for _ in range(int(qL * 4)):
                notes.append(n[0])
        if notes:
            pitch_only_pieces.append(notes)
    return pitch_only_pieces


def encode_sequences(input_sequences, encodings, output_sequences=None):
    """Encodes each item within a sequence.

    Arguments:
        input_sequences {nd numpy array} -- network inputs
        encodings {dictionary} -- encodings for network inputs/output values

    Keyword Arguments:
        output_sequences {numpy array} -- array of network outputs (default: {None})

    Returns:
        tuple -- (encoded inputs, encoded outputs)
    """
    encoded_inputs = np.zeros(input_sequences.shape)
    for j, s in enumerate(input_sequences):
        for k, n in enumerate(s):
            encoded_note = encodings[n]
            encoded_inputs[j][k] = encoded_note
    if output_sequences is not None:
        encoded_outputs = np.zeros(output_sequences.shape)
        for j, s in enumerate(output_sequences):
            encoded_outputs[j] = encodings[s]
    else:
        encoded_outputs = None
    return encoded_inputs, encoded_outputs


def one_hot_sequences(encoded_inputs, domain_size, encoded_outputs=None):
    """Convert sequences of encoded data into one-hot vectors.

    Arguments:
        encoded_inputs {nd numpy array} -- encoded input sequences
        domain_size {int} -- number of encodings

    Keyword Arguments:
        encoded_outputs {numpy array} -- encoded outputs (default: {None})

    Returns:
        tuple -- (one-hotted input sequences, one-hotted outputs)
    """
    if encoded_outputs is not None:
        one_hot_outputs = np.zeros(encoded_outputs.shape + tuple([domain_size]))
        for j, o in enumerate(encoded_outputs):
            np.put(one_hot_outputs[j], o, 1.0)
    else:
        one_hot_outputs = None
    one_hot_inputs = np.zeros(encoded_inputs.shape + tuple([domain_size]))
    for j, s in enumerate(encoded_inputs):
        for k, n in enumerate(s):
            np.put(one_hot_inputs[j][k], n, 1.0)
    return one_hot_inputs, one_hot_outputs


def encode_data(data):
    """Creates encodings (and decodings) for data.

    Arguments:
        data {list} -- list of data types

    Returns:
        tuple -- (new encodings, new decodings)
    """
    encodings = dict([(idx, datapoint) for datapoint, idx in enumerate(data)])
    decodings = dict([(datapoint, idx) for datapoint, idx in enumerate(data)])
    return encodings, decodings


def encode_multi_sequences(input_sequences, encodings, output_sequences=None):
    """Encoding function for data containing multiple labels.

    Arguments:
        input_sequences {nd numpy array} -- network inputs
        encodings {dictionary} -- encodings for network inputs/output values

    Keyword Arguments:
        output_sequences {numpy array} -- array of network outputs (default: {None})

    Returns:
        tuple -- (encoded inputs, encoded outputs)
    """
    domain_size = len(encodings)
    encoded_inputs = np.empty(np.array(input_sequences).shape + tuple([domain_size]))
    for j, s in enumerate(input_sequences):
        for k, n in enumerate(s):
            encoded_objs = [encodings[o] for o in n]
            encoded_inputs[j][k] = encoded_objs + [
                None for _ in range(domain_size - len(encoded_objs))
            ]
    if output_sequences is not None:
        encoded_outputs = np.empty(output_sequences.shape + tuple([domain_size]))
        for j, s in enumerate(output_sequences):
            encoded_o = [encodings[o] for o in s]
            encoded_outputs[j] = encoded_o + [
                None for _ in range(domain_size - len(encoded_o))
            ]
    else:
        encoded_outputs = None
    return encoded_inputs, encoded_outputs


def multi_hot_sequences(encoded_inputs, domain_size, encoded_outputs=None):
    """Convert sequences of encoded data into vectors.

    Arguments:
        encoded_inputs {nd numpy array} -- encoded input sequences
        domain_size {int} -- number of encodings

    Keyword Arguments:
        encoded_outputs {numpy array} -- encoded outputs (default: {None})

    Returns:
        tuple -- (vectorised input sequences, vectorised outputs)
    """
    if encoded_outputs is not None:
        one_hot_outputs = np.zeros(encoded_outputs.shape)
        for j, o in enumerate(encoded_outputs):
            for oo in o:
                if math.isnan(oo):
                    continue
                np.put(one_hot_outputs[j], oo, 1.0)
    else:
        one_hot_outputs = None
    one_hot_inputs = np.zeros(encoded_inputs.shape)
    for j, s in enumerate(encoded_inputs):
        for k, n in enumerate(s):
            for no in n:
                if math.isnan(no):
                    continue
                np.put(one_hot_inputs[j][k], no, 1.0)
    return one_hot_inputs, one_hot_outputs
