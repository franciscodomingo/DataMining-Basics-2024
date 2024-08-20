# Data Mining #

In this repository, there is Python code that processes an agriculture dataset, and the following tasks are performed:

1. **Categorical variables encoding**: Convert the categorical variables into numerical format using techniques like one-hot encoding or label encoding to prepare them for machine learning algorithms.

2. **Data exploration**: Perform exploratory data analysis (EDA) by generating various visualizations, such as histograms, box plots, and scatter plots, to understand the distribution, relationships, and patterns in the data.

3. **Outlier removal**: Identify and remove outliers that can skew the results of the analysis or modeling. This is done using methods like IQR (Interquartile Range) or Z-score to ensure the data is clean and reliable.

4. **Dataset scaling**: Apply feature scaling techniques such as MinMax Scaler and Standard Scaler to normalize the range of independent variables. This step ensures that all features contribute equally to the model and improves the performance of machine learning algorithms.

5. **Correlation analysis**: Examine the relationships between variables using correlation matrices and heatmaps. This helps in understanding the strength and direction of the associations between features, which is critical for feature selection and multicollinearity checks.

6. **PCA (Principal Component Analysis)**: Implement PCA to reduce the dimensionality of the dataset while retaining most of the variance in the data. PCA helps in simplifying the dataset by transforming the original variables into a new set of uncorrelated components (principal components).

7. **Cumulative variance plotting**: Create graphs that show the cumulative variance explained by each principal component. This helps in determining how many components are necessary to capture the majority of the variance in the data.

8. **Elbow method**: Use the elbow method to determine the optimal number of principal components by identifying the point where the explained variance starts to plateau. This provides a balance between dimensionality reduction and information retention.

9. **Variance, Kaiser criterion, and elbow plot analysis**: Compare and analyze the results from the cumulative variance, the Kaiser criterion (eigenvalues greater than 1), and the elbow plot to confirm the number of principal components to retain for further analysis.

---

To visualize high-dimensional data, the following steps are taken:

10. **ISOMAP plotting**: Implement ISOMAP (Isometric Mapping) to create 2D and 3D visualizations of the high-dimensional data. ISOMAP is a nonlinear dimensionality reduction technique that preserves the global structure of the data in the lower-dimensional space.

11. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Apply t-SNE to visualize the dataset in a lower-dimensional space, focusing on preserving the local structure and relationships between data points. t-SNE is particularly useful for visualizing clusters and patterns in complex datasets.

12. **UMAP (Uniform Manifold Approximation and Projection)**: Perform UMAP to project the data into a lower-dimensional space while preserving both local and global structure. UMAP is known for its speed and ability to maintain the true topology of the data.

13. **K-Means clustering**: Implement K-Means clustering to group the data into distinct clusters based on feature similarity. This unsupervised learning technique helps in identifying patterns and segmenting the data into meaningful groups.

14. **Hierarchical clustering**: Conduct hierarchical clustering, a method that builds a tree-like structure (dendrogram) of clusters. This technique is useful for understanding the hierarchy and relationships between clusters at different levels of granularity.

15. **GAP analysis**: Perform GAP analysis to evaluate the clustering performance and determine the optimal number of clusters by comparing the within-cluster dispersion with that of a reference null distribution.



## Python Libraries ##

pandas, to create dataframes.

matplotlib to create graphs.

seaborn to create greater graphs.

SKLearn to scale and generate TSNE, isomap and PCA.

umap to create umap.