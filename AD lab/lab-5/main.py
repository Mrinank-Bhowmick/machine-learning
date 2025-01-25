import pandas as pd
from sklearn.metrics import classification_report,f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt



iris = pd.read_csv("iris.csv")
X = iris.drop(['Species',"Id"],axis="columns")
y=iris["Species"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model=DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test,y_pred)
print("Report: ")
print(report)


plot_tree(model,filled=True,feature_names=X.columns,class_names=model.classes_)
plt.show()
