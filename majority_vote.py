from math import comb

def mv_accuracy(n, p):
    total_accuracy = 0
    for k in range(n // 2 + 1): 
        accuracy = comb(n, k) * ((1 - p)**k) * (p ** (n - k))
        total_accuracy += accuracy
    return total_accuracy

# 4.a)
n_models = 3
accuracy_per_model = 0.6
acc = mv_accuracy(n_models, accuracy_per_model)
print(f"4.a) Accuracy being {accuracy_per_model} for {n_models} models: {acc}\n")

# 4.b)
n_models = 5
accuracy_per_model = 0.6
acc = mv_accuracy(n_models, accuracy_per_model)
print(f"4.b) Accuracy being {accuracy_per_model} for {n_models} models: {acc}\n")

# 4.c)
n_models = 25
accuracy_per_model = 0.6
acc = mv_accuracy(n_models, accuracy_per_model)
print(f"4.c) Accuracy being {accuracy_per_model} for {n_models} models: {acc}\n")

# 4.d)
print("4.d) In the majority vote ensemble method, it is assumed that each classifier in the ensemble makes predictions independently of the others. However, in practice, this assumption may not hold true. If the classifiers in the ensemble are trained on similar data or share common features, they may end up making correlated predictions. In such cases, the ensemble's performance may not be as good as expected based on the assumption of independence.\n")

# 4.e)
n_models = 25
accuracy_per_model = 0.45
acc = mv_accuracy(n_models, accuracy_per_model)
print(f"4.e) Accuracy being {accuracy_per_model} for {n_models} models: {acc}")

