'''
Team 17 - CS 7641 ML Project Fall 2021
"Visual Reasoning for the Visually Impaired"
Authors: Angana Borah, Devshree Bharatia, Dimitri Adhikary, Megha Thukral, Yusuf Ali
'''

#!/usr/bin/env python

#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import os
from tqdm import tqdm
import pandas as pd

import shutil 
import pickle

import argparse

from numba import cuda 

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


print("Successfully Imported All Libraries in Conda Environment !")

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

NUM_IMAGES = 0
BATCH_SIZE = 16


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224)) # (299,299) for Inception
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path



if __name__ == "__main__":

    data_path = '/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/'
    '''
    input_shape = (299, 299)
    
    num = 0

    train_data_folder = data_path + 'train_VizWiz/train/'

    image_path = list()

    for img_file in os.listdir(train_data_folder):
        image_path.append(train_data_folder + img_file)

    train_image_paths = image_path

    img_name_vector = []

    for image_path in train_image_paths:
        img_name_vector.extend([image_path])

    image_model = tf.keras.applications.ResNet50(include_top=False,
                                                weights='imagenet', pooling='max')
    
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    encode_train = sorted(set(img_name_vector))

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

    features_list = []
    img_name = []

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        # batch_features = tf.reshape(batch_features,
        #                             (batch_features.shape[0], -1, batch_features.shape[3]))

        # print("\n Batch feature size:", batch_features.shape)
        # print("\n Path: ", path)
        # print("\n *************************")
        
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            #np.save(path_of_feature, bf.numpy())

            # print("\n Individual batch feature size: ", bf.shape)
            # print("\n Path: ", p.numpy().decode("utf-8"))

            # print("\n ----------------------------------------------")

            bf = bf / np.linalg.norm(bf)

            print(bf.shape)

            features_list.append(bf)
            img_name.append(path_of_feature)
    
    print("\n Features list size: ", len(features_list))
    print("\n Img name size: ", len(img_name))

    img_features = features_list
    '''
    flag = 1
    
    if flag == 0:
        f = open(data_path + 'VizWiz_ResNet50_norm_max.pkl', 'wb')
        pickle.dump([img_features,img_name], f)
        f.close()

    if flag == 1:    
        with open(data_path + 'VizWiz_files/VizWiz_ResNet50_norm_avg.pkl', 'rb') as f:
                img_features, img_name = pickle.load(f)

        print("\n Features list size: ", len(img_features))
        print("\n Img name size: ", len(img_name))

    loss_values = []
    
    for K in [1, 2, 3, 5, 10, 20, 40, 60, 300, 1000]:
    
        # K = 60
        
        clusters = KMeans(K, random_state = 40)
        print("\n Running KMeans for K = ", K, " ... ")

        clusters.fit(img_features)
        print("\n Kmeans Completed ")

        # Convert the generated clusters into a pandas dataframe which consists of two columns ---> image_name and cluster_id
        image_cluster = pd.DataFrame(img_name, columns = ['image'])
        image_cluster["clusterid"] = clusters.labels_
        
        # Calculate K-Means algorithm obejctive function value (sum of squared distances between data points and centroids)
        print("\n Calculating loss... ")
        image_cluster_loss = clusters.inertia_
        print("\n Loss for K = ", K, " : ", image_cluster_loss)

        print("\n ***************************************** \n")

        loss_values.append(image_cluster_loss)

    print("\n Loss Values: ", loss_values)

    '''
    # Create clustered_data directory in the root VizWiz folder which will contain the images segregated into their respective clusters
    print("\n Creating clustered_data directory... ", len(img_features))
    for i in range(K):
        path_cluster_folder = data_path + 'clustered_data_COCO/' + str(i)

        if not os.path.exists(path_cluster_folder):
            os.makedirs(path_cluster_folder)

        else:
            shutil.rmtree(path_cluster_folder)                      # Removes all the subdirectories if the clustered_data directory already exists
            os.makedirs(path_cluster_folder)
    print("\n clustered_data directory created ", len(img_features))
    
    # Copy the clustered images from the trainign data folder into the respective cluster folders in the clustered_data directory 
    print("\n Copying images into respective cluster folders... ", len(img_features))
    
    idx = 1
    
    for i in tqdm(range(len(img_features))):
        # print("\n Saved image ", idx)
        source = image_cluster['image'][i]
        destination = os.path.join(data_path, 'clustered_data_COCO/', str(image_cluster['clusterid'][i]))
        shutil.copy(source, destination)

        # idx += 1
    
    print("\n Copied images into respective cluster folders... ", len(img_features))
    '''
    
    device = cuda.get_current_device()
    device.reset()