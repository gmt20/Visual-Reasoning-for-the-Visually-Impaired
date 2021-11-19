import os

import json
import numpy as np
import ast

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
import nltk.corpus
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

import itertools
import re
from collections import Counter

import pickle


def prepare_answers(annotations):

    # print("\n ************************************************")
    
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]
    prepared = []

    for sample_answers in answers:

        prepared_sample_answers = []

        for answer in sample_answers:
            # lower case
            answer = answer.lower()

            # define desired replacements here
            punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}

            rep = punctuation_dict
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            answer = pattern.sub(lambda m: rep[re.escape(m.group(0))], answer)
            prepared_sample_answers.append(answer)

            # print(prepared_sample_answers)

        prepared.append(prepared_sample_answers)
        # print(prepared)
    
    return prepared


def create_answer_vocab(annotations, top_k):

    # flattens out the cleaned answers into a single list in which element is an answer
    answers = itertools.chain.from_iterable(prepare_answers(annotations)) 
    # print(answers)

    counter = Counter(answers)
    # print(counter)

    counted_ans = counter.most_common(top_k)

    # start from labels from 0

    # print(counted_ans)

    vocab = {t[0]: i for i, t in enumerate(counted_ans, start=0)}
    # print(vocab)

    return vocab 


def encode_answers(answers, answer_to_index):
    #print("\n **************Inside Fucnction ************")
    answer_vec = np.zeros(len(answer_to_index))

    #print(answers)

    i = 0

    for answer in answers:
        #print(answer)

        index = answer_to_index.get(answer)

        #print(index)
        
        if index is not None:
            answer_vec[index] += 1
    
    return answer_vec

# ---------------------------------------------------------------------------------------------------------------------------

with open("/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/updated_annotations.json",'r') as lst:
    filtered_data = ast.literal_eval(lst.read())
    print("\n No of points in filtered dataset : ", len(filtered_data))

# print(filtered_data[0:5])


answer_vocab = create_answer_vocab(filtered_data, 3000)


# vocabs = {'answer': answer_vocab}
# print("\n Filtered data vocab : ", vocabs)


# answer_to_index = vocabs['answer']
# print("\n answer to index : ", answer_to_index)

answers = prepare_answers(filtered_data)
# print("\n answers : ", answers[1])

answers = [encode_answers(a, answer_vocab) for a in answers]
# answer = encode_answers(answers[1], answer_vocab)
print(np.nonzero(answers))

# print("\n Labelled vectors shape : ", np.count_nonzero(answers[566]))

num_non_zero = 0
sum_not_ten = 0

for label_vec in answers:
    if np.count_nonzero(label_vec) == 0:
        num_non_zero += 1
    
    if np.sum(label_vec) != 10:
        sum_not_ten += 1

print("\n Number of label vectors with zeros only: ", num_non_zero)
print("\n Number of label vectors with sum != 10: ", sum_not_ten)

# print(np.nonzero(answers[5]))

data_path = '/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/'

'''f = open(data_path + 'label_vec_VizWiz.pkl', 'wb')
pickle.dump(answers, f)
f.close()'''