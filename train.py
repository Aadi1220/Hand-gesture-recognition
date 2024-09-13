import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Read the dataset
df = pd.read_csv('landmarks_medium.csv')

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df.drop('class_id', axis=1), df['class_id'],
                                                    test_size=0.30, random_state=1)

# Train the logistic regression model
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

# Make predictions on the testing set
prediction = logmodel.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)

# Saving the model using pickle
filename = 'gesture_model.sav'
pickle.dump(logmodel, open(filename, 'wb'))
