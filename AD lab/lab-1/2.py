# 
import pandas as pd
import seaborn as sns

# Load the Iris dataset
iris = sns.load_dataset('iris')

print("Sample of the Iris dataset:")
print(iris.head())

# Display summary of the dataset
print("\nDataset Information:")
print(iris.info())

print("\nSummary Statistics:")
print(iris.describe())
