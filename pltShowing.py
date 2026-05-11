import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\Legion\Desktop\MachineLearningProject\data\creditcard.csv")

# Compute the correlation matrix
corr = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(16,12))

# Draw the heatmap with seaborn
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, cbar=True)

plt.title('Correlation Heatmap of Credit Card Transactions', fontsize=18)
plt.show()
