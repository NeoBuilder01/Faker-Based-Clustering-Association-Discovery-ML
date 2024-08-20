# Faker-Based-Clustering-Association-Discovery-ML


Synthetic Data Analysis with PCA, KMeans, and Association Rules

Project Overview:
This project demonstrates the application of various data analysis techniques on synthetic datasets generated using the Faker library. It integrates Principal Component Analysis (PCA), KMeans clustering, and association rule mining (Apriori algorithm) to provide insights into customer behavior and product associations.

Key Features:
Synthetic Data Generation: Utilizes the Faker library to create realistic synthetic data for product transactions and customer interactions.

Principal Component Analysis (PCA): Applies PCA for dimensionality reduction, helping to visualize and interpret high-dimensional data in a three-dimensional space.

KMeans Clustering: Implements KMeans clustering to segment customers into distinct clusters based on their transaction data. Visualizes the clustering results at different stages of the algorithm's execution.

Association Rule Mining (Apriori Algorithm): Analyzes product associations using the Apriori algorithm to discover frequent itemsets and generate association rules. Visualizes the top association rules based on support, confidence, and lift metrics.

How It Works
Data Generation:

Synthetic data is created to simulate product transactions among customers using the Faker library.
The data includes customer IDs and the products they purchase, structured into a DataFrame.
Data Analysis:

PCA: Reduces the dimensionality of the data and projects it into a 3D space for visualization.
KMeans Clustering: Clusters customers into groups based on their purchase patterns. Intermediate and final clustering results are visualized.
Apriori Algorithm: Finds association rules from the transaction data to identify patterns and relationships between products.
Visualization:

PCA Visualization: Displays the data points in 3D space after applying PCA.
KMeans Visualization: Shows the clustering results at different stages and the final clustering.
Apriori Rules Visualization: Provides a 3D scatter plot of the association rules based on support, confidence, and lift.
