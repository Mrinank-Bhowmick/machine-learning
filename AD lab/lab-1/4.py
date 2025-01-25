import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load sample dataset (Iris dataset)
iris = sns.load_dataset('iris')

# Scatter plot to visualize the relationship between sepal length and sepal width
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=iris,
    x='sepal_length',
    y='sepal_width',
    hue='species',  # Color points by species
    style='species',  # Different marker styles for each species
    palette='deep',
    s=100  # Marker size
)

# Add titles and labels
plt.title('Sepal Length vs Sepal Width by Species', fontsize=16)
plt.xlabel('Sepal Length (cm)', fontsize=14)
plt.ylabel('Sepal Width (cm)', fontsize=14)

# Show the plot
plt.legend(title='Species', loc='upper right')
plt.grid(alpha=0.3)
plt.show()
