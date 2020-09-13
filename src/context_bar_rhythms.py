def remove_anacruses(pieces):
    """Remove all.

    Arguments:
        pieces {[type]} -- [description]nces

    Returns:
        list -- list of sequences with anacrusis sequences removed
    """
    new_pieces = []
    for _, p in enumerate(pieces):
        new_p = p
        if new_p[0][-1] == "Rest":
            continue
        if new_p[0][0] == "Rest":
            continue
        new_pieces.append(new_p)
    return new_pieces


def get_unique_rhythms(bars):
    """Counts the frequencies of each unique context rhythm.

    Arguments:
        bars {list} -- list of sequences/contexts

    Returns:
        dictionary -- {rhythm:frequency}
    """
    unique_rhythms = {}
    for b in bars:
        bar_rhythm = []
        last_note = None
        unique_note = 0
        for n in b[0]:
            if n != last_note:
                unique_note += 1
            bar_rhythm.append(unique_note)
            last_note = n
        bar_rhythm = tuple([tuple(bar_rhythm), tuple(b[1])])
        if bar_rhythm not in unique_rhythms:
            unique_rhythms[bar_rhythm] = 1
        else:
            unique_rhythms[bar_rhythm] += 1
    return unique_rhythms


def get_context_bar_rhythms(input_sequences, end_sequences, bar_length):
    """Retrieves all bar-wise context-length rhythms from the dataset.

    Arguments:
        input_sequences {list} -- list containing all input sequences from the dataset
        end_sequences {list} -- list containing all end sequences from the dataset
        bar_length {int} -- number of timesteps in a bar

    Returns:
        dictionary -- all unique whole-bar context-length rhythms (inc. slur pattern) in the dataset along with their corresponding frequencies
    """
    note_input_sequences = [
        i + j for i, j in zip(input_sequences["Note"], end_sequences["Note"])
    ]
    slur_input_sequences = [
        i + j for i, j in zip(input_sequences["Slur"], end_sequences["Slur"])
    ]
    slur_input_sequences = [
        [[[0 if l == [] else 1 for l in k] for k in j] for j in i]
        for i in slur_input_sequences
    ]
    unique_input_sequences = []
    unique_slur_sequences = []
    for i, p in enumerate(note_input_sequences):
        for j, s in enumerate(p):
            if j % bar_length == 0:
                unique_input_sequences.append(s[0])
                unique_slur_sequences.append(slur_input_sequences[i][j][0])
    zipped_seqs = [
        [unique_input_sequences[i], unique_slur_sequences[i]]
        for i in range(len(unique_input_sequences))
    ]
    no_an_seqs = remove_anacruses(zipped_seqs)
    unique_rhythms = get_unique_rhythms(no_an_seqs)
    return unique_rhythms
