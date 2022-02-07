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


def run(train_in, output_file, model_file=None):
    start = time.time()
    with open(train_in, 'r', encoding='utf8') as f:
        train_lines = f.readlines()

    clf = MaxEnt.MaxEntClassifier()
    clf.train_raw = train_lines

    if model_file is not None:
        with open(model_file, 'r', encoding='utf8') as f:
            model_lines = f.readlines()
        clf.load_model(model_lines)
        clf.process_train()
        y_tr_pred, y_tr_probs = clf.predict(instances=clf.X_train, save='train')
    else:
        clf.process_train()

    clf.calc_model_exp()
    clf.save_exp(output_file, exp_type='model')

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
        OUTPUT_FILE = 'q4/model_count2'
        MODEL_FILE = None #'q1/m1.txt'
    else:
        TRAIN_IN = sys.argv[1]
        OUTPUT_FILE = sys.argv[2]
        if len(sys.argv) < 4:
            MODEL_FILE = None
        else:
            MODEL_FILE = sys.argv[3]
    run(TRAIN_IN, OUTPUT_FILE, MODEL_FILE)


