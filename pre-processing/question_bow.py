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

import pickle
import re


data_path = '/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/'


with open("/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/updated_annotations.json",'r') as lst:
    filtered_data = ast.literal_eval(lst.read())
    print("\n No of points in filtered dataset : ", len(filtered_data))


'''with open("/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/Annotations/train.json",'r') as train_file:
    train_data = json.load(train_file)
    print(len(train_data))'''

df_updated = pd.DataFrame(filtered_data)

list_of_questions = list(df_updated.question)
print(list_of_questions[:5])

'''
tokenized_word_in_questions = [word_tokenize(i) for i in list_of_questions]

print(tokenized_word_in_questions[0])

flattened_tokens = sum(tokenized_word_in_questions, [])
print(flattened_tokens[0])


# print(flattened_tokens[0])

stop_words = set(stopwords.words('english'))
# print(stop_words)
filtered_tokens = [w for w in flattened_tokens if not w.lower() in stop_words]  # removes stopwords and lower cases the word
print(len(filtered_tokens))


filtered_tokens = flattened_tokens


filtered_tokens = [w for w in flattened_tokens if w.isalnum()]
print(len(filtered_tokens))

unique = list(set(filtered_tokens))

print(len(unique))

print(unique)

'''
t  = Tokenizer()

for i in range(len(list_of_questions)):
    # print(list_of_questions[i])
    list_of_questions[i] = re.sub("[^A-Za-z +]", "", list_of_questions[i])
    # print(list_of_questions[i])


# t.fit_on_texts(list_of_questions)
t.fit_on_texts(list_of_questions)


encoded_docs = t.texts_to_matrix(list_of_questions, mode = 'count')

print("\n The document count", t.document_count)
# print("\n BOW vector : ", encoded_docs[4])

print(encoded_docs.shape)
print(encoded_docs[0])

'''f = open(data_path + 'question_BOW_VizWiz.pkl', 'wb')
pickle.dump([encoded_docs], f)
f.close()'''