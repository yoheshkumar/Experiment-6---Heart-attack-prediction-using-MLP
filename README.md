# Experiment-6---Heart-attack-prediction-using-MLP

## Aim:
To construct a  Multi-Layer Perceptron to predict heart attack using Python

## Algorithm:

1. Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.
2. Load the heart disease dataset from a file using pd.read_csv().
3. Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).
4. Split the dataset into training and testing sets using train_test_split().
5. Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.
6. Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.
7. Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.
8. Make predictions on the testing set using mlp.predict(X_test).
9. Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().
10. Print the accuracy of the model.
11. Plot the error convergence during training using plt.plot() and plt.show().

## Program:
```
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset (assuming it's stored in a file)
data = pd.read_csv('heart.csv')

# Separate features and labels
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
training_loss = mlp.fit(X_train, y_train).loss_curve_

# Make predictions on the testing set
y_pred = mlp.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the error convergence
plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.show()
```
## Output:

![image](https://github.com/SarankumarJ/Experiment-6---Heart-attack-prediction-using-MLP/assets/94778101/83ebae9c-b9c4-4192-be70-5ce59f56f02a)

## Result:
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
