import seaborn as sns
import matplotlib.pyplot as plt

# Load sample dataset (Iris dataset)
iris = sns.load_dataset('iris')

# Compute correlation matrix for numerical columns only
correlation_matrix = iris.select_dtypes(include=['float64', 'int64']).corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,        # Show correlation values in the cells
    fmt=".2f",         # Format the annotations to two decimal places
    cmap="coolwarm",   # Color map
    linewidths=0.5,    # Add lines between cells
    cbar=True          # Display color bar
)

# Add title
plt.title('Feature Correlation Heatmap', fontsize=16)

# Show plot
plt.show()
