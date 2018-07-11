from __future__ import print_function

import numpy as np
import csv, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.utils.data_utils import get_file
rom keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

## NLP preprocessing helper
# dataset types:
# one file with 2 sentences each row, tab-delimited.  (parallel-data)
# 2 parallel folders, multiple files inside internal folder in each 1 sentence each row. (parallel-data)
#https://github.com/keithecarlson/Zero-Shot-Style-Transfer/tree/master/Data/Bibles/
# ASV/1Chronicles/1Chronicles10.txt
# ASV/1Chronicles/1Chronicles11.txt
# ASV/Colossians/Colossians1.txt
# XYZ/Colossians/Colossians1.txt

# internals:
#  vocab fitted only on train part (word2id, id2word)

# output:
# LM as classification: x is first N tokens . y is only one tokens N+1
#    as seq2seq: x is N tokens.  y is N tokens, with are advanced by 1.
# can be both char level or word level

# Pairs classification
# x is 2 sentences (x1,x2) , y is label (duplicate/not.  entitelitmennt/neutral/...)

# translation
# x is sentence of size N, y is sentence of different size M

#see: torchtext http://anie.me/On-Torchtext/

def load_text_pairs():
    '''
    :return: tuple of utf8 sentences style1, style2 and label. (assumes parallel corpus)
    '''
    KERAS_DATASETS_DIR = expanduser('~/.keras/datasets/')
    QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
    QUESTION_PAIRS_FILE = 'quora_duplicate_questions.tsv'
    if not exists(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE):
        get_file(QUESTION_PAIRS_FILE, QUESTION_PAIRS_FILE_URL)

    print("Processing", QUESTION_PAIRS_FILE)

    question1 = []
    question2 = []
    is_duplicate = []
    with open(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            question1.append(row['question1'])
            question2.append(row['question2'])
            is_duplicate.append(row['is_duplicate'])

    print('Question pairs: %d' % len(question1))
    for i in range(4,6):
        print (is_duplicate[i],question1[i],question2[i])
    return question1,question2,is_duplicate


def vectorize(question1,question2,is_duplicate):
    from vocab import Vocabulary

    v = Vocabulary()


    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

