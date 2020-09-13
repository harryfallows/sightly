import os
import argparse

units = []
opt = []
s_loss = []
layers = []
dropout = []
features = "n"
time_signatures = "6/8"
epochs = 0
system_folder = "abrsm_violin_grade_5"
parser = argparse.ArgumentParser()
parser.add_argument("-n", dest="range", default=[], nargs="+")
args = parser.parse_args()
combos = []

for o in opt:
    for s in s_loss:
        for l in layers:
            for d in dropout:
                for u in units:
                    combos.append("-u {} -o {} -s {} -l {} -d {}".format(u, o, s, l, d))

for i in combos[int(args.range[0]) : int(args.range[-1])]:
    os.system(
        "python3 ../driver.py -f {} --features {} --time_signatures {} --epochs {} {}".format(
            system_folder, features, time_signatures, epochs, i
        )
    )
