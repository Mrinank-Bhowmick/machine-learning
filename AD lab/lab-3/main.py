# Implement Multiple Linear Regression to predict student performance based on study hours, class attendance and 
# assignment scores. (a) Calculate matrix like MSE & R2 Score to assess performance.
# (b) Create a scatter plot comparing the actual vs predicted values 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('StudentPerformanceFactors.csv')

# Drop all columns except Hours_Studied, Attendance, Previous_Scores, Exam_Score
df = df[['Hours_Studied', 'Attendance', 'Previous_Scores', 'Exam_Score']]

# Prepare features and target variable
X = df[['Hours_Studied', 'Attendance', 'Previous_Scores']]
y = df['Exam_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores')
plt.show()