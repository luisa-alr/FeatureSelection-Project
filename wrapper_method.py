# Luisa Rosa
# HW 3 - Data Mining
# 03/18/2024

import arff
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

# Function to perform LOOCV and return the accuracy
def loocv_accuracy(features, labels, k):
    loocv = LeaveOneOut()
    correct = 0
    total = 0
    for train_index, test_index in loocv.split(features):
        X_train, X_test = (
            np.array(features)[train_index],
            np.array(features)[test_index],
        )
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
X = np.array([point[:-1] for point in data_points])
y = np.array([point[-1] for point in data_points])

# Define k value for KNN
k = 7

## Question 3:
# Define variables and arrays to perform SFS
selected_features = []
selected_idx = []
final_acc = 0
step = 0

# Sequential Forward Selection
f_list = []
remaining_flist = []
f_list = [attributes[i][0] for i in range(len(attributes) - 1)]
remaining_flist = list(range(len(attributes) - 1))

print(f"Step 0: ")
print(f"Selected feature subset is {selected_features}")

while len(remaining_flist) > 0:
    step += 1
    print(f"Step {step}: ")
    tmp_acc_list = []
    for i in range(len(remaining_flist)):
        tmp_flist = selected_idx + [remaining_flist[i]] # Combine selected features with the current feature
        # print(tmp_flist)
        tmp_X = X[:, tmp_flist] # Select columns from the dataset corresponding to the selected features
        tmp_acc_list.append(loocv_accuracy(tmp_X, y, k)) # Compute accuracy using LOOCV
    # print("Features    = ", [f_list[idx] for idx in remaining_flist])
    # print("Accuracies  = ",tmp_acc_list )

    ## Question 3.a):
    max_acc = max(tmp_acc_list)
    max_idx = tmp_acc_list.index(max_acc)
    max_feature = remaining_flist[max_idx]

    print(f"Maximum Accuracy achieved is {max_acc}%, with feature f{max_feature}")

    # Update selected feature subset if accuracy improves
    if max_acc >= final_acc:
        selected_features.append(f_list[max_feature])
        selected_idx.append(max_feature)
        remaining_flist.remove(max_feature)
        final_acc = max_acc
        print("New Selected feature subset is ", selected_features)
    else:
        print(
            "Accuracy is not increased from the previous feature set. Breaking out of loop.\n\n"
        )
        break

## Question 3.b):
print("Final Selected Feature set is ,", selected_features)
print("Final Accuracy with above feature set is ", final_acc)

