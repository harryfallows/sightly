import os
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder", dest="folder", help="Folder containing models.", required=True,
)
args = parser.parse_args()
folder = args.folder

for i, file in enumerate(os.listdir(folder)):
    os.system("python3 test_driver.py -m {}".format(folder + "/" + file))
