from math import sqrt
from collections import Counter
import numpy as np

# Calculate the Euclidean distance between two vectors
def euclidean_dist(row1, row2):
    dist = 0.0
    for i in range(len(row1) - 1):
        dist += (row1[i] - row2[i]) ** 2
    return sqrt(dist)

# Locate the most similar k neighbors
def knn(train, test_row, k):
    distances = list()
    for train_row in train:
        dist = euclidean_dist(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_class(train, test_row, k):
    neighbors = knn(train, test_row, k)
    output_values = [row[-1] for row in neighbors]
    vote_counts = Counter(output_values)
    prediction = vote_counts.most_common(1)[0][0]
    return prediction

# Calculate Accuracy
def accuracy(predictions, test_class):
    correct = sum(
        1 for pred, true_label in zip(predictions, test_class) if pred == true_label
    )
    return correct / float(len(test_class)) * 100

# Function to predict class of a test instance
def predict_class_fm(X_train, test_instance, y_train, k):
    distances = [euclidean_dist(test_instance, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
    return most_common

# Function to calculate accuracy
def accuracy_fm(predictions, y_test):
    correct = sum(predictions == y_test)
    total = len(y_test)
    return correct / total
