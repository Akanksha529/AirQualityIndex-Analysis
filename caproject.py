import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_excel(r"C:\Users\akank\Downloads\Cleaned_Air_Quality_Data.xlsx")
#basic explore
df.describe()
# Overview of dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#unique pollutant
df['pollutant_id'].unique()
print("Unique countries:", df['country'].unique())
print("Unique pollutants:", df['pollutant_id'].unique())
#(EDA)
# Correlation matrix
correlation = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(correlation)

# Covariance matrix
covariance = df.cov(numeric_only=True)
print("\nCovariance Matrix:")
print(covariance)

# Heatmap of correlation
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Outlier detection using IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers

# Example for detecting outliers in 'pollutant_avg'
outliers = detect_outliers_iqr(df, 'pollutant_avg')
print("\nDetected Outliers in 'pollutant_avg':")
print(outliers)

# Group by state and calculate average AQI
top_states = df.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False).head(5).reset_index()

# Plotting using correct column names and DataFrame
plt.figure(figsize=(10, 6))
sns.barplot(data=top_states, x='pollutant_avg', y='state', hue='state', palette='Reds_r', legend=False)
plt.xlabel("Average Pollutant Level (AQI)")
plt.ylabel("State")
plt.title("Top 5 States with Highest Average AQI Levels")
plt.tight_layout()
plt.show()


# all graphs

#distribution of pollutants
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='pollutant_id', palette='Set2',hue='pollutant_id',legend=False)
plt.title("Distribution of Pollutants")
plt.xlabel("Pollutant Type")
plt.ylabel("Count")
plt.show()


#  Plot - Average pollutant levels(bar plot)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='pollutant_id', y='pollutant_avg',hue='pollutant_id', palette='coolwarm',legend=True)
plt.title("Average Pollutant Levels")
plt.xlabel("Pollutant Type")
plt.ylabel("Average Value")
plt.tight_layout()
plt.show()

# Step 7: Plot - Pollutants in a specific city (e.g., Gaya)
gaya_data = df[df['city'].str.lower() == 'gaya']
plt.figure(figsize=(10, 6))
sns.barplot(data=gaya_data, x='pollutant_id', y='pollutant_avg', hue='pollutant_id',palette='magma',legend=False)
plt.title("Pollutant Levels in Gaya")
plt.xlabel("Pollutant")
plt.ylabel("Average Level")
plt.tight_layout()
plt.show()


#boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='pollutant_id', y='pollutant_avg', hue='pollutant_id', palette='Set2', dodge=False)
plt.legend([],[], frameon=False)  # Removes duplicate legend
plt.title('Distribution of Average Pollution Levels by Pollutant', fontsize=14, weight='bold')
plt.xlabel('Pollutant Type')
plt.ylabel('Average Pollution Level')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Create the pairplot
pollutant_data = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']]
sns.pairplot(pollutant_data, kind='scatter', plot_kws={'alpha': 0.7, 's': 80}, diag_kws={'color': 'skyblue'})
plt.suptitle("Pairplot of Pollutant Levels", fontsize=14, y=0.02)
plt.show()

#heatmap
numerical_df = df[['pollutant_min', 'pollutant_max', 'pollutant_avg', 'latitude', 'longitude']]
# Compute the correlation matrix
corr_matrix = numerical_df.corr()
# Plot the correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black')
plt.title(" Correlation Heatmap of Numerical Features", fontsize=14)
plt.tight_layout()
plt.show()

# Create histogram for pollutant_avg
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='pollutant_avg', bins=15, kde=True, color='skyblue', edgecolor='black')
plt.title(" Distribution of Pollutant Average Values", fontsize=14)
plt.xlabel("Pollutant Average")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#lineplot
df['last_update'] = pd.to_datetime(df['last_update'])
# Sort by time (optional but cleaner)
df = df.sort_values(by='last_update')
# Set the figure size
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='last_update', y='pollutant_avg', hue='pollutant_id', marker='D', style='pollutant_id')
plt.title("Pollutant Trends Over Time with Markers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

#scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='latitude', y='pollutant_avg', hue='pollutant_id', palette='viridis')
plt.title("Pollutant Average vs Latitude")
plt.xlabel("Latitude")
plt.ylabel("Pollutant Average")
plt.legend(title="Pollutant Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
