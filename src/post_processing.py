# import numpy as np
from music21 import (
    note,
    stream,
    analysis,
    meter,
    key,
    spanner,
    articulations,
    expressions,
    dynamics,
)
import time
import calendar


def split_qls(legal_qls, current_qls):
    """Splits notes into legal lengths.

    Arguments:
        legal_qls {set} -- set of quarter lengths allowed in grade
        current_qls {list} -- current ql of note

    Returns:
        list -- list of note lengths
    """
    splits = []
    while current_qls >= min(legal_qls):
        next_max_ql = max([l_ql for l_ql in legal_qls if l_ql <= current_qls])
        splits.append(next_max_ql)
        current_qls -= next_max_ql
    if current_qls != 0:
        splits.append(current_qls)
    return splits


def join_notes(prediction_output, legal_qls, rest_qls=[], time_signature=(4, 4)):
    """Joins identical adjacent notes.

    Arguments:
        prediction_output {dictionary} -- output from music generation process
        legal_qls {set} -- set of quarter lengths allowed in grade (note lengths)

    Keyword Arguments:
        rest_qls {list} --  list of rest lengths allowed in grade, if not used defaults to same as legal_qls (default: {[]})
        time_signature {tuple} -- time signature of piece (default: {(4, 4)})

    Returns:
        2-d list -- list of bars or piece with joined notes
    """
    if rest_qls == []:
        rest_qls = legal_qls
    print(time_signature)
    timesteps_per_bar = int((time_signature[0] * 16) / time_signature[-1])
    objs = list(prediction_output.keys())
    bars = [
        {k: v[i : i + timesteps_per_bar] for k, v in prediction_output.items()}
        for i in range(0, len(prediction_output["Note"]), timesteps_per_bar)
    ]
    bars_joined = []
    for bar in bars:
        bar_contents = []
        curr_length = 0.25
        for j in range(len(bar["Note"])):
            current_note = {i: bar[i][j] for i in objs}
            try:
                next_note = {i: bar[i][j + 1] for i in objs}
            except IndexError:
                this_note = [
                    {**current_note, **{"quarter_length": ql}}
                    for ql in split_qls(
                        legal_qls if current_note["Note"] != "Rest" else rest_qls,
                        curr_length,
                    )
                ]
                bar_contents += this_note
                continue
            if current_note == next_note:
                curr_length += 0.25
            else:
                this_note = [
                    {**current_note, **{"quarter_length": ql}}
                    for ql in split_qls(
                        legal_qls if current_note["Note"] != "Rest" else rest_qls,
                        curr_length,
                    )
                ]
                bar_contents += this_note
                curr_length = 0.25
        bars_joined.append(bar_contents)
    return bars_joined


def generate_score(
    piece,
    output_folder,
    text_expressions=[],
    file_name="",
    key_signature=None,
    time_signature="4/4",
):
    """Converts piece into sheet music and saves it as MusicXML file.

    Arguments:
        piece {[type]} -- [description]
        output_folder {string} -- folder to save pieces into

    Keyword Arguments:
        text_expressions {list} -- list of text expressions in piece (default: {[]})
        file_name {str} -- file name to save piece as (default: {""})
        key_signature {[]} -- key signature for piece, if left blank a key signature is generated (default: {None})
        time_signature {str} -- time signature of piece (default: {"4/4"})
    """
    print(piece)
    note_offset = 0
    generated_piece = []
    slurs = []
    current_slur_start = None
    current_spanners = {}
    current_dynamic = None
    piece_stream = stream.Stream()
    piece_stream.append(meter.TimeSignature(time_signature))
    for bar in piece:
        for step in bar:
            if step["Note"] == "Rest":
                note_as_note = note.Rest()
            else:
                note_as_note = note.Note(step["Note"])
            note_as_note.quarterLength = step["quarter_length"]
            note_as_note.offset = note_offset
            try:
                note_as_note.articulations = [
                    a()
                    for a in step["Articulation"]
                    if isinstance(a(), articulations.Articulation)
                ]
                note_as_note.expressions = [
                    e()
                    for e in step["Articulation"]
                    if isinstance(e(), expressions.Expression)
                ]
            except KeyError:
                pass

            piece_stream.insert(note_offset, note_as_note)
            try:
                if step["Dynamic"] != current_dynamic:
                    current_dynamic = step["Dynamic"]
                    piece_stream.insert(note_offset, dynamics.Dynamic(step["Dynamic"]))
            except KeyError:
                pass
            try:
                if current_slur_start is not None:
                    if step["Slur"] == []:
                        slurs.append(spanner.Slur([current_slur_start, note_as_note]))
                        current_slur_start = None
                else:
                    if step["Slur"] != []:
                        current_slur_start = note_as_note
            except KeyError:
                pass
            cs_to_delete = []
            for cs in current_spanners.keys():
                if cs not in step["Spanner"]:
                    piece_stream.insert(0, cs([current_spanners[cs], note_as_note]))
                    cs_to_delete.append(cs)
            for cs in cs_to_delete:
                del current_spanners[cs]
            try:
                for s in step["Spanner"]:
                    if s not in current_spanners.keys():
                        current_spanners.update({s: note_as_note})
            except KeyError:
                pass
            note_offset += step["quarter_length"]

    for i, te in enumerate(text_expressions):
        for e in te:
            piece_stream.insert(float(i) / 4, expressions.TextExpression(e))
    if key_signature is not None:
        if not isinstance(key_signature, int):
            key_signature = key_signature.sharps
        piece_stream.keySignature = key.KeySignature(key_signature)
    else:
        try:
            piece_stream.keySignature = analysis.discrete.KrumhanslSchmuckler().getSolution(
                piece_stream
            )
        except analysis.discrete.DiscreteAnalysisException:
            piece_stream.keySignature = key.KeySignature(0)

    for s in slurs:
        piece_stream.insert(0, s)

    piece_stream.write(
        "xml",
        fp="{}/{}-{}.xml".format(
            output_folder, file_name, calendar.timegm(time.gmtime())
        ),
    )
