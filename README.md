---
bibliography:
- bibliography.bib
---

Running The System {#running-the-system .unnumbered}
==================

As mentioned in section
[\[sec:code\_management\]](#sec:code_management){reference-type="ref"
reference="sec:code_management"}, different files are used to separate
the functionality of the system. In addition to these files, there are
files containing the \"driver code\" used to run each stage on some
given data. All driver code scripts contain command line parameters
which can be used to change how the code is run - most parameters are
optional and have default values. Argument parsing is handled by the
argparse [@davis2015package] package; all optional and required
arguments for a file, along with their descriptions, can be found by
running the following code in the command line:
[`python <filename>.py –help`]{style="background-color: light-gray"}.

Inputs Required {#sec:inputs_required .unnumbered}
---------------

The inputs required in order to generate sight-reading music are: a
dataset of test pieces and a JSON file containing the requirements of
the instrument and grade. Figure
[\[code:requirements.json\]](#code:requirements.json){reference-type="ref"
reference="code:requirements.json"} shows the requirements JSON file
used to test the system; the format of each key-value pair are explained
below:

-   : These represent the allowed note lengths, each defined as a
    fraction of a quarter note (quaver).

-   : The possible time signatures that could appear in the grade.

-   : The possible key signatures that could appear in the grade.
    Positive numbers indicate the number of sharps; negative numbers
    indicate the number of flats.

-   : The allowed lengths of rests, each defined as a fraction of a
    quarter note (quaver).

-   : The range of piece lengths that could feature in the grade,
    measured in number of bars.

-   : The range of pitches allowed in the grade.

-   : The possible keys that could appear in the grade. Capitalised
    letters represent major keys, lowercase letters represent minor
    keys, and the \"-\" character represents \"flat\".

``` {.json language="json" startFrom="1"}
{
      "quarter_lengths": [1, 0.5, 2, 3, 1.5, 0.75, 0.25],
      "time_signatures": ["4/4", "2/4", "3/4", "6/8"],
      "key_signatures": [1, 2, 3, 0, -1, -2, -3, 4, -4],
      "rest_lengths": [1, 2, 0.5],
      "length": [8, 16],
      "pitch_range": ["G3", "E6"],
      "keys": [
        "D", "A", "G", "e", "C", "F", "B-", "a", "d", "g", "E-", "E", "A-", "b", "c"
      ]
    }
```

Preparing Data {#preparing-data .unnumbered}
--------------

All data preprocessing can be run using the
[`data_preparation.py`]{style="background-color: light-gray"} script.
This script takes in a folder containing the data and requirements for a
specific instrument grade, as well as a number of optional arguments,
and performs all data processing required. This includes, but is not
limited to: all preprocessing, analysing the data for seed rhythms, and
splitting the data into train, test and validation data. Figure
[\[tab:data\_prep\_params\]](#tab:data_prep_params){reference-type="ref"
reference="tab:data_prep_params"} shows all of the parameters accepted
by [`data_preparation.py`]{style="background-color: light-gray"}.

  ----------- ----------------------- ---------------------------------------------------------------------- ------------------------------- ----------
  Char Flag   Flag                    Description                                                            Default Value                   Required
  -f          --folder                Folder containing requirements (requirements.json) and data (data/).   n/a                             Yes
  None        --time\_signatures      Time signatures to create data for.                                    All available time signatures   No
  -a          --dont\_augment\_data   Don't augment data flag. -a for don't augment, nothing for augment.    False                           No
  -b          --bars\_context         Length of musical context given to the neural network.                 4                               No
  ----------- ----------------------- ---------------------------------------------------------------------- ------------------------------- ----------

Training a Model {#training-a-model .unnumbered}
----------------

A model can be trained by running the
[`driver.py`]{style="background-color: light-gray"} script. The only
required parameter is, again, the folder containing the data (and now
preprocessed data too) and the JSON file containing the grade
requirements. Optional parameters can be used to modify the default
layout of the model or to change other aspects of training. Figure
[\[tab:driver\_params\]](#tab:driver_params){reference-type="ref"
reference="tab:driver_params"} shows all of the parameters accepted by
[`driver.py`]{style="background-color: light-gray"}.

  ----------- -------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------- ----------
  Char Flag   Flag                       Description                                                                                                                                          Default Value                   Required
  -f          --folder                   Folder containing requirements (requirements.json) and data (data/).                                                                                 n/a                             Yes
  None        --features                 Which extra musical features to have. Slurs: s, Dynamics: d, Spanners: p, Articulations: a, Text expressions: t. Example format: --features sdpat.   s                               No
  -a          --dont\_augment\_data      Don't augment data flag. -a for don't augment, nothing for augment.                                                                                  False                           No
  None        --early\_stopping          Stop the model once the validation loss has reached a minimum.                                                                                       False                           No
  -u          --lstm\_units              Number of LSTM units in each layer.                                                                                                                  512                             No
  -d          --dropout                  Dropout after each LSTM layer.                                                                                                                       0.3                             No
  -l          --lstm\_layers             Number of LSTM layers in network.                                                                                                                    3                               No
  -s          --single\_loss\_function   Loss function for single-label classification features (e.g. notes).                                                                                 categorical-\_crossentropy      No
  -m          --multi\_loss\_function    Loss function for multi-label classification features (e.g. expressions).                                                                            binary-\_crossentropy           No
  -o          --optimiser                Optimiser for network.                                                                                                                               Adam                            No
  -i          --last\_layer\_inputs      Input multi-label features before final layer of network.                                                                                            False                           No
  None        --time\_signatures         Time signatures to create model for.                                                                                                                 All available time signatures   No
  None        --epochs                   Number of epochs to train for.                                                                                                                       60                              No
  ----------- -------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------- ----------

Generating Music {#generating-music .unnumbered}
----------------

Music can be generated from a trained model by running the
[`generator_driver.py`]{style="background-color: light-gray"} script
with the location of the model folder as a parameter. Figure
[\[tab:generator\_driver\_params\]](#tab:generator_driver_params){reference-type="ref"
reference="tab:generator_driver_params"} shows the parameters accepted
by [`generator_driver.py`]{style="background-color: light-gray"}. If
neither the loss\_flag or val\_loss\_flag are chosen, both are set to
true. If neither the end\_sequences\_flag or artificial\_seeds\_flag are
chosen, both are set to true.

  ----------- --------------------------- ------------------------------------------------------------------------------------------------------ --------------- ----------
  Char Flag   Flag                        Description                                                                                            Default Value   Required
  -m          --model\_folder             Folder containing model.                                                                               n/a             Yes
  -p          --no\_generated\_pieces     Number of generated pieces.                                                                            50              No
  -l          --loss\_flag                Generate music from the weights saved at the minimum loss of the network during training.              False           No
  -v          --val\_loss\_flag           Generate music from the weights saved at the minimum validation loss of the network during training.   False           No
  -e          --end\_sequences\_flag      Generate music from end sequence seeds.                                                                False           No
  -a          --artificial\_seeds\_flag   Generate music from artificial seeds.                                                                  False           No
  ----------- --------------------------- ------------------------------------------------------------------------------------------------------ --------------- ----------
