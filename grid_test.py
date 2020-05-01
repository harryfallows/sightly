import os
import sys

model_folder = sys.argv[1]
for i, file in enumerate(os.listdir(model_folder)):
    os.system("python3 test_driver.py -m {}".format(model_folder + "/" + file))
