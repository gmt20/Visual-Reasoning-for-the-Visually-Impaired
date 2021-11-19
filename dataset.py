'''
Team 17 - CS 7641 ML Project Fall 2021
"Visual Reasoning for the Visually Impaired"
Authors: Angana Borah, Devshree Bharatia, Dimitri Adhikary, Megha Thukral, Yusuf Ali

Script Summary: Generate VizWiz VQA dataset which will be used in the training pipeline
'''

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
        '''
        Init function which calls other class functions to generate the dataset 
        '''

        self.data_root = data_root              # path to directory which contains the question_BOW_VizWiz.pkl, label_vec_VizWiz.pkl, updated_annotations.json, VizWiz_Inception_normalised_max.pkl
        self.question_BOW_arr = None            # numpy matrix which stores question_BOW vectors ---> Dim: (13108,3042) | Each question BOW is a (3042,) vector
        self.label_vec_arr = None               # numpy matrix which stores label_answer_vectors ---> Dim: (13108,3000) | Each label_vec is a (3000,) vector
        self.img_features_arr = None            # numpy matrix  which stores CNN features for each image ---> Dim: (13108, 2048) | Each CNN feature is a (2048,) vector

        # Load necessary data from .pkl files ---> store data from each .pkl file into respective variables
        self.question_BOW_arr, self.label_vec_arr, self.img_features_arr = self.load_questions_BOW_answer_labels()
        
        # Concatenate the image features and question BOWs
        self.merged_features = self.concat_img_features_question_BOW(self.question_BOW_arr, self.img_features_arr)

    
    def load_questions_BOW_answer_labels(self):
        '''
        This function does three things which are the following:
            1. Get data from "question_BOW_VizWiz.pkl" file and store it into "self.question_BOW_arr" as numpy_matrix
            2. Get data from "label_vec_VizWiz.pkl" and store it into "self.label_vec_arr" as numpy matrix
            3. Get data from "VizWiz_Inception_normalised_max.pkl" and select only necessary features based on the "updated_annotations.json" file ---> store the filtered image featrues as numpy matrix
        
        Args: 
            None
        Return: 
            question_BOW_arr: numpy matrix of size (13108,3042) 
            label_vec_arr: numpy matrix of size (13108,3000)
            img_features_arr: numpy matrix of size (13108, 2048)
        '''

        # -------------------------------------------------------------------------------------------------------------------
        # 1. Load question_BOW ---> Convert to numpy array

        with open(self.data_root + '/question_BOW_VizWiz.pkl', 'rb') as f:
            training_questions = pickle.load(f)

        # print("\n Training Questions Shape: ", training_questions[0].shape)

        question_BOW_arr = training_questions[0]

        print("\n Training Questions Shape: ", question_BOW_arr.shape)
        print("\n Training Questions Type: ", type(question_BOW_arr))


        # -------------------------------------------------------------------------------------------------------------------
        # 2. Load label_vectors ---> Convert to numpy array

        with open(self.data_root + '/label_vec_VizWiz.pkl', 'rb') as f:
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
        # 3. Load image features and the updated_annotations.json file ---> Filter out required features based on selected data points ---> Convert to numpy array

        with open(self.data_root + '/updated_annotations.json', 'r') as lst:
            filtered_data = ast.literal_eval(lst.read())                                    # Read the .json file which contains the filtered data points

        names_list = list()
        for names in filtered_data:
            names_list.append(names['image'])                                               # Append names of all filtered images to the names_list

        with open(self.data_root + '/VizWiz_Inception_normalised_max.pkl', 'rb') as f:
            img_features, img_name = pickle.load(f)                                         # Load generated CNN features and corresponding names for all images in dataset (23890 in number)

        all_features = pd.DataFrame(img_name, columns = ['image'])                          # Create pandas dataframe which will constitute all image features and corresponding image_names in the dataset
        all_features["img_features"] = img_features                                         # Create a column whcih contains all the image features

        i = 0
        for name in all_features['image']:                                                  # This loop does some formatting to change the image names from the entire absolute path name to just the image name | For ex: 
            all_features['image'][i] = name.replace('/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/train_VizWiz/train/', '')    # For ex: "/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/train_VizWiz/train/VizWiz_train_00000001.jpg" is replaced with "VizWiz_train_00000001.jpg"
            i += 1                                                                          # We need to do this because "names_list" contains only the image_names (from .json file) adn not the entire absolute paths ---> hence, we cant compare the name values directly to filter out the required data points

        filtered_features_df = all_features[all_features['image'].isin(names_list)]         # Filter oout the necessary data points based on the image_names present in the names_list

        filtered_features = filtered_features_df.iloc[:,1]                                  # Store the filtered image_features into a new variable 
        filtered_features = np.asarray(filtered_features)                                   # Convert the above dataframe which contains teh filtered image_features into a numpy matrix ---> We will get a single numpy array, each row of which is a TF Tensor type of image feature (this is because I used TF to geenrate CNN image features (kmeans_COCO.py) - I know this is bad coding but I will change the pipeline in the future)

        for row in range(filtered_features.shape[0]):   
            filtered_features[row] = filtered_features[row].numpy()                         # Convert each TF Tensor image feature (row) into numpy type row

        img_features_arr = np.zeros((filtered_features.shape[0], filtered_features[0].shape[0]))    # Initialise the final img_features_arr which will hold the image features

        for i in range(img_features_arr.shape[0]):
            img_features_arr[i] = filtered_features[i]                                      # Store each numpy image feature from "filtered_features" into the img_features_arr

        print("\n Image Features Array Shape: ", img_features_arr.shape)
        print("\n Image Features Array Type: ", type(img_features_arr))

        # -------------------------------------------------------------------------------------------------------------------

        return question_BOW_arr, label_vec_arr, img_features_arr


    def concat_img_features_question_BOW(self, question_BOW_arr, img_features_arr):
        '''
        Stack the CNN image features and question_BOW array to get the final input array for the training pipeline

        Args: 
            question_BOW_arr: numpy matrix of size (13108,3042) 
            img_features_arr: numpy matrix of size (13108,2048) 
        Return: 
            merged_features: numpy matrix of size (13108,5090) 
        '''

        merged_features = np.hstack((img_features_arr, question_BOW_arr))

        print("\n Merged Array Shape: ", merged_features.shape)

        return merged_features
    

    def __len__(self):
        '''
        Derived function from torch.utils.data.Dataset which returns length of the generated dataset 
        For ex: train_set = VizWiz_VQA_Dataset(data_path)
                We can get the length of the dataset by calling len(train_set)

        Args: 
            None
        Return: 
            Number of data points in the dataset
        '''

        return len(self.merged_features)

    
    def __getitem__(self, idx):
        '''
        Derived function from torch.utils.data.Dataset which returns a particular data point based on the index value
        For ex: train_set = VizWiz_VQA_Dataset(data_path)
                We can get the n-th element of train_Dataset by train_set[n]
                input merged_features for n-th data point = train_set[n][0]
                label_vector for n-th data point = train_set[n][1]

        Args: 
            idx: int value
        Return: 
            X: numpy array of size: (5090,)
            y: numpy array of size: (3000,)
        '''
        
        X = self.merged_features[idx]
        y = self.label_vec_arr[idx]
        
        return X, y