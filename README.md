## Data Mining - Ensemble Learning - Feature Selection Methods

## Luisa Rosa - Spring 2024

Feature selection is used to remove irrelevant or correlated features in order to improve classification performance. In this project you can compare 2 different feature selection methods: the Filter Method which doesn’t make use of cross-validation and the Wrapper Method which does.

## Instructions:

- Download all files (4 Python programs and 1 .arff dataset)
- To see the Feature Selection Methods, run the respective python program
  - Run fm_from_scratch.py to see the Filter Method applied without any libraries
  - Run filter_method.py to understand the Filter Method
  - Run wrapper_method.py to understand the Wrapper Method
  - Run majority_vote.py to understand Ensemble Learning Majority Vote

---

## Question 2: Filter Method

Make the class labels numeric (set “noncar”=0 and “car”=1) and calculate the Pearson Correlation Coefficient (PCC) of each feature with the numeric class label. The PCC value is commonly referred to as r.

(a) List the features from highest |r| (the absolute value of r) to lowest, along with their |r| values. Why would one be interested in the absolute value of r rather than the raw value?

(b) Select the features that have the highest m values of |r|, and run LOOCV on the dataset restricted to only those m features. Which value of m gives the highest LOOCV classification accuracy, and what is the value of this optimal accuracy?

### Solution:

1. Step 1 - Convert labels and extract features and class labels into their own variables.

2. Step 2 - Calculate PCC for each feature.

3. Step 3 - Sort features in descending order based on their absolute correlation coefficients.

4. Step 4 - Print sorted features with their absolute correlation coefficients.

5. Step 5 - Fix k = 7 for all runs of Leave-One-Out Cross-Validation.

6. Step 6 - Perform Leave-One-Out-Cross-Validation for different values of m.
   - Select top m features based on their absolute correlation coefficients
   - Restrict the dataset to only those m features
   - Perform LOOCV
   - Update best accuracy and corresponding m value

---

## Question 3: Wrapper Method

Starting with the empty set of features, use a greedy approach to add the single feature that improves performance by the largest amount when added to the feature set. This is Sequential Forward Selection. Define performance as the LOOCV classification accuracy of the KNN classifier using only the features in the selection set (including the “candidate” feature). Stop adding features only when there is no candidate that when added to the selection set increases the LOOCV accuracy.

(a) Show the set of selected features at each step, as it grows from size zero to its final size (increasing in size by exactly one feature at each step)

(b) What is the LOOCV accuracy over the final set of selected features?

### Solution:

1. Step 1 - Convert labels and extract features and class labels into their own variables.

2. Step 2 - Fix k = 7 for all runs of Leave-One-Out Cross-Validation.

3. Step 3 - Define variables and arrays to perform Sequential Forward Selection.

4. Step 4 - Loop through the feature list:

   - Combine selected features with the current feature
   - Select columns from the dataset corresponding to the selected features
   - Compute accuracy using LOOCV
   - Update selected feature subset if accuracy improves

5. Step 5 - Print the Selected Feature Subset in each iteration.

   - identifying the feature selected and the maximum accuracy achieved in that iteration

6. Step 6 - Determine the Final Features Selected and the final accuracy with the best feature set.

---

## Question 4: Ensemble Learning - Majority Vote

Suppose we need to build a predictive model for a binary classification task. We have 25 students in our class. Each of us worked independently and everyone is able to builda model with 60% accuracy.

a) If we take 3 models and build a majority vote classifier C3, what would be theaccuracy of our new classifier C3? Show your work.

b) If we take 5 models and build a majority vote classifier C5, what would be theaccuracy of our new classifier C5? Show your work.

c) If we take all 25 models and build a majority vote classifier C25, what would bethe accuracy of our new classifier C25? Show your work.

d) The performance you obtained for C25 is too good to be true. What’s the as-sumption in your calculations that often does not hold in reality?

e) What would be the answer to (c) if everyone’s model only has 45% accuracy? Show your work.
