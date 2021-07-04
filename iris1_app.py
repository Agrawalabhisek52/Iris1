import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(x_train, y_train)

@st.cache
def prediction(model,s_l,s_w,p_l,p_w):
	pred = model.predict([[s_l,s_w,p_l,p_w]])
	pred = pred[0]
	if pred == 0:
		return 'Iris-setosa'
	elif pred == 1:
		return 'Iris-virginica'
	else :
		return 'Iris-versicolor'

st.sidebar.title("Sizes")
# Add 4 sliders and store the value returned by them in 4 separate variables. 
sepell_length = st.sidebar.slider("Sepell Length",5.5,15.5)
sepell_width = st.sidebar.slider("Sepell Width",6.6,16.6)
petell_length = st.sidebar.slider("Petell Length",7.7,17.7)
petell_width = st.sidebar.slider("Petell Width",8.8,18.8)
# The 'float()' function converts the 'numpy.float' values to Python float values.

# Add a select box in the sidebar with the 'Classifier' label.
# Also pass 3 options as a tuple ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier').
# Store the current value of this slider in the 'classifier' variable.
classifier = st.sidebar.selectbox("Classifiers",('SVM','RFC','LR'))
# When the 'Predict' button is clicked, check which classifier is chosen and call the 'prediction()' function.
# Store the predicted value in the 'species_type' variable accuracy score of the model in the 'score' variable. 
# Print the values of 'species_type' and 'score' variables using the 'st.text()' function.
rfc_model = RandomForestClassifier().fit(x_train,y_train)
lr_model = LogisticRegression().fit(x_train,y_train)

if st.sidebar.button("predict"):
  if classifier == 'SVM':
    name = prediction(svc_model,sepell_length,sepell_width,petell_length,petell_width)
    score = svc_model.score(x_train,y_train)
  elif classifier == 'RFC':
    name = prediction(rfc_model,sepell_length,sepell_width,petell_length,petell_width)
    score = rfc_model.score(x_train,y_train)
  else :
    name = prediction(lr_model,sepell_length,sepell_width,petell_length,petell_width)
    score = lr_model.score(x_train,y_train)

  st.write("Species -:",name,'\n\nScore -:',score)