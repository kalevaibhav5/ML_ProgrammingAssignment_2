# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------- Part-1: Import and Data Preprocessing for ANN --------

# Importing the dataset
dataset = pd.read_csv('./Bank_Predictions.csv')

# Get the Features and Output into X and Y
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Encode the categorial variables Location and Gender
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

transformer = make_column_transformer(
    (OneHotEncoder(), ['Location','Gender']),
    remainder='passthrough')

transformed = transformer.fit_transform(pd.DataFrame(X))
X = pd.DataFrame(
    transformed,
    columns=transformer.get_feature_names()
)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ------- Part-2: Build the ANN --------

# import keras library and packages
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(32,input_shape=(13,),name='Input_Layer',activation='relu'))
classifier.add(Dense(32,name='Hidden_Layer_1',activation='relu'))

# Adding second hidden layer

classifier.add(Dense(32,name='Hidden_Layer_2',activation='relu'))

# Adding output layer

classifier.add(Dense(1,name='Output_Layer',activation='sigmoid'))

classifier.summary()

# Compiling the ANN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

# Fitting the ANN to the training set

classifier.fit(X_train, y_train, epochs=100,batch_size=10)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

# Normalize the predicted output

y_pred = np.where(y_pred > 0.5, 1, 0)


# Making the confusion Matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.show()





