#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def load_data_and_labels(data_file, with_labels = True, raw_data = False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    # {"Very Negative", "Negative", "Neutral", "Positive", "Very Positive"}
    x_raw = []
    with open(data_file, 'r') as f:
        x_raw = [s.strip() for s in list(f.readlines())]

    if raw_data:
        return x_raw
    else:
        x_text = []

        if with_labels:
            labels = []

            label_mappings = {
                '0': [1,0,0,0,0],
                '1': [0,1,0,0,0],
                '2': [0,0,1,0,0],
                '3': [0,0,0,1,0],
                '4': [0,0,0,0,1]
            }

            # Default label for unkonw label
            default_label = [0,0,1,0,0]

            for sample in x_raw:
                x_text.append(clean_str(sample[2:]))
                labels.append(label_mappings.get(sample[0], default_label))

            y_label = np.concatenate([labels], 0)

        else:
            for sample in x_raw:
                x_text.append(clean_str(sample))

            y_label = None

        return [x_text, y_label]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_data(lable_data_file):
    lable_examples = list(open(lable_data_file, "r", -1, "utf-8").readlines())
    lable_examples = [s.strip() for s in lable_examples]
    x_text = lable_examples
    x_text2 = [clean_str(sent) for sent in x_text]
    x_text=[sent for sent in x_text]
    return[x_text2,x_text]

