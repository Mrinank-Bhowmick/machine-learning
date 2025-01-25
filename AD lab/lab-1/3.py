# plot the distribution using matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
data = np.random.normal(loc=50, scale=10, size=1000)  # Normal distribution with mean=50, std=10

# Plot the distribution
plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

# Add titles and labels
plt.title('Distribution of Data', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)

# Show the plot
plt.show()
