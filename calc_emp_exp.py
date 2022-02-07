import sys
import os
import re
import numpy as np
import MaxEnt

import random
import time

"""
MaxEnt
"""
start = time.time()


def run(train_in, output_file):
    start = time.time()
    with open(train_in, 'r', encoding='utf8') as f:
        train_lines = f.readlines()

    clf = MaxEnt.MaxEntClassifier()
    clf.train_raw = train_lines
    clf.process_train()
    clf.save_emp_exp(output_file)

    end = time.time()
    total_time = end - start
    total_mins = total_time / 60
    print("time=" + str(total_time))
    # fname = 'q2/time_'
    # with open(fname, 'w', encoding='utf8') as f:
    #     f.write("time (s) " + str(total_time) + " | time (m) " + str(total_mins))


if __name__ == "__main__":
    TEST = False
    if TEST:
        TRAIN_IN = '/Users/Karl/_UW_Compling/LING572/hw5/hw5/examples/train2.vectors.txt'
        OUTPUT_FILE = 'q3/emp_count'
    else:
        TRAIN_IN = sys.argv[1]
        OUTPUT_FILE = sys.argv[2]
    run(TRAIN_IN, OUTPUT_FILE)


