import copy
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torch.nn.functional as F

import sys
import pickle as pk

import tensorflow as tf

import pickle, ast
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split


class VizWiz_VQA_Dataset(Dataset):
    
    def __init__(self, data_root):

        self.data_root = data_root
        self.question_BOW_arr = None
        self.label_vec_arr = None
        self.img_features_arr = None

        # Load necessary data from .pkl files
        self.question_BOW_arr, self.label_vec_arr, self.img_features_arr = self.load_questions_BOW_answer_labels()
        
        # Concatenate the image features and question BOW
        self.merged_features = self.concat_img_features_question_BOW(self.question_BOW_arr, self.img_features_arr)

    
    def load_questions_BOW_answer_labels(self):

        # -------------------------------------------------------------------------------------------------------------------
        # Load image features ---> Filtered out required features ---> Convert to numpy array

        with open(self.data_root + 'question_BOW_VizWiz.pkl', 'rb') as f:
            training_questions = pickle.load(f)

        # print("\n Training Questions Shape: ", training_questions[0].shape)

        question_BOW_arr = training_questions[0]

        print("\n Training Questions Shape: ", question_BOW_arr.shape)
        print("\n Training Questions Type: ", type(question_BOW_arr))


        # -------------------------------------------------------------------------------------------------------------------
        # Load label_vectors ---> Convert to numpy array

        with open(self.data_root + 'label_vec_VizWiz.pkl', 'rb') as f:
            answers_train = pickle.load(f)

        # print("\n Total label vectors: ", len(answers_train))
        # print("\n Shape of label vector: ", answers_train[0].shape)

        label_vec_arr = np.zeros((len(answers_train), answers_train[0].shape[0]))

        i = 0
        for vec in answers_train:
            label_vec_arr[i] = answers_train[i]
            i += 1

        print("\n Label Vector Array Shape: ", label_vec_arr.shape)
        print("\n Label Vector Array Type: ", type(label_vec_arr))


        # -------------------------------------------------------------------------------------------------------------------
        # Load image features and filtered dataset .json file ---> Filtered out required features ---> Convert to numpy array

        with open("/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/updated_annotations.json",'r') as lst:
            filtered_data = ast.literal_eval(lst.read())

        # print("\n No of points in filtered dataset : ", len(filtered_data))
        # print(filtered_data[0]['image'])

        names_list = list()
        for names in filtered_data:
            names_list.append(names['image'])

        with open(self.data_root + 'VizWiz_files/VizWiz_Inception_normalised_max.pkl', 'rb') as f:
            img_features, img_name = pickle.load(f)

        # print("\n Features list size: ", len(img_features))
        # print("\n Img name size: ", len(img_name))

        all_features = pd.DataFrame(img_name, columns = ['image'])
        all_features["img_features"] = img_features

        i = 0
        for name in all_features['image']:
            all_features['image'][i] = name.replace('/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/train_VizWiz/train/', '')
            i += 1

        # print("\n Image dataset dataframe : ", all_features['image'][0])

        filtered_features_df = all_features[all_features['image'].isin(names_list)]
        # print(filtered_features_df.iloc[:,1])

        filtered_features = filtered_features_df.iloc[:,1]
        filtered_features = np.asarray(filtered_features)
        # filtered_features = tf.convert_to_tensor(filtered_features)

        for row in range(filtered_features.shape[0]):
            filtered_features[row] = filtered_features[row].numpy()

        img_features_arr = np.zeros((filtered_features.shape[0], filtered_features[0].shape[0]))
        # print(features_arr.shape)

        for i in range(img_features_arr.shape[0]):
            img_features_arr[i] = filtered_features[i]

        print("\n Image Features Array Shape: ", img_features_arr.shape)
        print("\n Image Features Array Type: ", type(img_features_arr))

        return question_BOW_arr, label_vec_arr, img_features_arr


    def concat_img_features_question_BOW(self, question_BOW_arr, img_features_arr):

        merged_features = np.hstack((img_features_arr, question_BOW_arr))

        print("\n Merged Array Shape: ", merged_features.shape)

        return merged_features
    

    def __len__(self):

        return len(self.merged_features)

    
    def __getitem__(self, idx):
        
        X = self.merged_features[idx]
        y = self.label_vec_arr[idx]
        
        return X, y