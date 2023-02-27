# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from frameextractor import frameExtractor
from sklearn.metrics.pairwise import cosine_similarity
from handshape_feature_extractor import HandShapeFeatureExtractor

## import the handfeature extractor class


# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video


def extract_frames_and_features(videos_path, frames_path):

    print("extracting frames...")
    frames_path.mkdir(exist_ok=True)

    for i, f in enumerate(sorted(videos_path.iterdir())):
        # assuming there are only video files and nothing else in dir
        print(str(f), str(frames_path), i)
        frameExtractor(str(f), str(frames_path), i)

    print("extracting features...")

    features = []
    for f in sorted(frames_path.iterdir()):
        # assuming there are only extracted frames and nothing else in dir
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        feature = HandShapeFeatureExtractor.get_instance().extract_feature(img)
        features.append(feature)

    return features


def extract_labels(videos_path):

    def extract_label(filename):

        for i in range(10):
            if f'num{i}' in filename:
                return i

        if 'fanDown' in filename: return 10
        if 'fanOn' in filename: return 11
        if 'fanOff' in filename: return 12
        if 'fanUp' in filename: return 13
        if 'lightsOff' in filename: return 14
        if 'lightsOn' in filename: return 15
        if 'setThermo' in filename: return 16

        raise ValueError(f'Unexpected file name: {filename}')


    return [extract_label(str(f)) for f in sorted(videos_path.iterdir())]



training_videos_path = Path('traindata')
training_frames_path = Path('trainframes')


print("Working on training set...")

train_features = extract_frames_and_features(training_videos_path, training_frames_path)
train_labels = extract_labels(training_videos_path)




# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

test_videos_path = Path('test')
testing_frames_path = Path('testframes')

print("Working on test set...")

test_features = extract_frames_and_features(test_videos_path, testing_frames_path)


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

print("Calculating Cosine Similarity...")


test_features=np.array(test_features)
train_features=np.array(train_features)
train_labels=np.array(train_labels)
test_features = test_features.reshape(test_features.shape[0], test_features.shape[2])
train_features = train_features.reshape(train_features.shape[0], train_features.shape[2])

similarities = cosine_similarity(test_features, train_features)
most_similar_index = np.argmax(similarities, axis=1)
predicted_labels = train_labels[most_similar_index]


video_names = iter(sorted(test_videos_path.iterdir()))
with open('Results.csv', 'w') as f:
    for label in predicted_labels:
        f.write(f"{str(label)}\n")
        print(f"perdicted labe;: {label} file name: {next(video_names)}")


