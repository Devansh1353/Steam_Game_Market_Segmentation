**Steam Games Clustering and Anomaly Detection**
Problem Statement:

Steam hosts a large collection of games with information such as price, positive reviews, negative reviews, and average playtime. Analyzing such a large amount of game data manually is difficult and time-consuming. This project uses unsupervised learning algorithms to discover hidden patterns within the dataset by grouping similar games into clusters and identifying unusual games. Detecting clusters helps users understand different categories of games and player behavior, while anomaly detection helps identify games with unique or unexpected characteristics.

Project Overview:

In this project, a Steam Games dataset was analyzed using various unsupervised learning techniques. The dataset was first cleaned and preprocessed by removing unnecessary columns and scaling numerical features using StandardScaler. K-Means Clustering was then applied to group similar games based on their characteristics. The Elbow Method was used to determine the optimal number of clusters, while Silhouette Score and Davies-Bouldin Index were used to evaluate clustering performance.

Hierarchical Clustering and Dendrogram visualization were implemented to study relationships between games in a hierarchical manner. DBSCAN was used as a density-based clustering technique to identify naturally occurring groups and noise points. Principal Component Analysis (PCA) was applied to reduce dimensionality and visualize clusters effectively in two dimensions. Finally, Isolation Forest was used to detect anomalous games that significantly differ from the majority of the dataset.

Libraries Used

NumPy was used for numerical computations and array operations.

Pandas was used for loading, cleaning, manipulating, and analyzing the Steam Games dataset.

Matplotlib was used to generate visualizations such as cluster plots, Elbow Method graphs, PCA plots, and anomaly detection visualizations.

Scikit-Learn was used to implement machine learning algorithms including K-Means Clustering, Hierarchical Clustering, DBSCAN, PCA, Isolation Forest, StandardScaler, Silhouette Score, and Davies-Bouldin Index.

SciPy was used for generating and visualizing the dendrogram required for Hierarchical Clustering.

Algorithms Implemented
K-Means Clustering
Elbow Method
Silhouette Score
Davies-Bouldin Index
Hierarchical Clustering
Dendrogram Visualization
DBSCAN Clustering
Principal Component Analysis (PCA)
Isolation Forest
Results

Made By:-
1)Dhruv Sharma
2)Devansh Sheth

The project successfully identified clusters of similar Steam games, visualized relationships between games using PCA, evaluated cluster quality using clustering metrics, and detected anomalous games through Isolation Forest. These techniques provide valuable insights into game characteristics and player engagement patterns without requiring predefined labels.# Steam_Game_Market_Segmentation
