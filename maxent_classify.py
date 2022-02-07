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


def run(test_in, model_file, sys_output):
    start = time.time()
    with open(test_in, 'r', encoding='utf8') as f:
        test_lines = f.readlines()
    with open(model_file, 'r', encoding='utf8') as f:
        model_lines = f.readlines()

    clf = MaxEnt.MaxEntClassifier()
    clf.load_model(model_lines)
    clf.test_raw = test_lines
    clf.process_test()

    y_pred_ts, y_probs_ts = clf.predict(clf.X_test, save='test')

    clf.save_sys_output(sys_output)
    clf.classification_report()
    end = time.time()
    total_time = end - start
    total_mins = total_time / 60
    # print("time=" + str(total_time))
    fname = 'q2/time_'
    with open(fname, 'w', encoding='utf8') as f:
        f.write("time (s) " + str(total_time) + " | time (m) " + str(total_mins))


if __name__ == "__main__":
    TEST = False
    if TEST:
        TEST_IN = '/Users/Karl/_UW_Compling/LING572/hw5/hw5/examples/test2.vectors.txt'
        MODEL_FILE = 'q1/m1.txt'
        SYS_OUTPUT = 'q2/res'
    else:
        TEST_IN = sys.argv[1]
        MODEL_FILE = sys.argv[2]
        SYS_OUTPUT = sys.argv[3]
    run(TEST_IN, MODEL_FILE,  SYS_OUTPUT)
