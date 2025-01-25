import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

dataset = pd.read_csv('./spambase.csv')

X = dataset.drop('spam',axis=1) # features
y = dataset['spam'] # label

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)

model = LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
confusion_mat = confusion_matrix(y_test,y_pred)
classification_rep = classification_report(y_test,y_pred)

print(f'Accuracy: {accuracy*100}%')

print("Confusion matrix: ")
print(confusion_mat)

print('Classification Report:')
print(classification_rep)