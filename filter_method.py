# Luisa Rosa
# HW 3 - Data Mining
# 03/18/2024

import arff
import math
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

# Function to get the Pearson Correlation Calculation using the pseudocode given
def pcc(x, y):
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    mean_x = 0
    mean_y = 0
    N = len(x)

    for i in range(N):
        sum_sq_x += x[i] * x[i]
        sum_sq_y += y[i] * y[i]
        sum_coproduct += x[i] * y[i]
        mean_x += x[i]
        mean_y += y[i]

    mean_x /= N
    mean_y /= N

    pop_sd_x = math.sqrt((sum_sq_x / N) - (mean_x * mean_x))
    pop_sd_y = math.sqrt((sum_sq_y / N) - (mean_y * mean_y))
    cov_x_y = (sum_coproduct / N) - (mean_x * mean_y)

    correlation = cov_x_y / (pop_sd_x * pop_sd_y) if (pop_sd_x * pop_sd_y) != 0 else 0
    return correlation

# Function to perform LOOCV and return the accuracy
def loocv_accuracy(features, labels, k):
    loocv = LeaveOneOut()
    correct = 0
    total = 0
    for train_index, test_index in loocv.split(features):
        X_train, X_test = np.array(features)[train_index], np.array(features)[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
        
        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Predict
        prediction = knn.predict(X_test)
        
        # Check accuracy
        if prediction == y_test:
            correct += 1
        total += 1
        
    return correct / total

# Load ARFF file
with open("veh-prime.arff", "r") as f:
    data = arff.load(f)

# Access data and attributes
attributes = data["attributes"]
data_points = data["data"]

# Convert labels
for point in data_points:
    if point[-1] == "noncar":
        point[-1] = 0
    elif point[-1] == "car":
        point[-1] = 1

# Extract features and class labels
features = [point[:-1] for point in data_points]
labels = [point[-1] for point in data_points]

# Calculate PCC for each feature
abs_corr = []
normal_corr = []
for feature in zip(*features):
    correlation = pcc(feature, labels)
    normal_corr.append(correlation)
    abs_corr.append(abs(correlation))

# QUESTION 2.A)
# Sort features in descending order based on their absolute correlation coefficients
sorted_features = sorted(
    zip(range(len(abs_corr)), abs_corr, normal_corr), key=lambda x: x[1], reverse=True
)

# Print sorted features with their absolute correlation coefficients
print("Features from highest |r| to lowest and their |r| values:")
for idx, abs_correlation, normal_correlation in sorted_features:
    print(f"Feature {idx :<5} |r| = {abs_correlation :<25} r = {normal_correlation}")

print(
    "Answer: By sorting based on |r| values, you can identify which features have the strongest association with the class label, regardless of whether the relationship is positive or negative. This helps in feature selection or identifying important predictors in a dataset."
)

# QUESTION 2.B)
# Fix k = 7 for all runs of Leave-One-Out Cross-Validation
k = 7

# Values of m to try
m_values = list(range(1, len(sorted_features) + 1))

best_accuracy = 0
best_m = None

# Perform LOOCV for different values of m
for m in m_values:
    print(f'For m value {m}: ')
    
    # Select top m features based on their absolute correlation coefficients
    top_m_features = [feature[0] for feature in sorted_features[:m]]
    print(top_m_features)
    
    # Restrict the dataset to only those m features
    restricted_features = [[point[i] for i in top_m_features] for point in features]
    
    # Perform LOOCV
    accuracy = loocv_accuracy(restricted_features, labels, k)
    print(accuracy)
    
    # Update best accuracy and corresponding m value
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_m = m

print(f"Optimal value of m: {best_m}")
print(f"Highest LOOCV classification accuracy: {best_accuracy * 100:.5f}%")


