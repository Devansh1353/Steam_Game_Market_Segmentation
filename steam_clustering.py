# INSTALL REQUIRED PACKAGES
# pip install numpy pandas matplotlib scikit-learn scipy


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage


print("\nLoading Dataset...\n")

df = pd.read_csv("steam_games_sample_50_rows.csv")

print(df.head())



print("\nRemoving Unnecessary Columns...\n")

df = df.drop(
    ['appid', 'name', 'recommendations', 'achievements'],
    axis=1
)

print(df.head())



print("\nApplying StandardScaler...\n")

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df)




print("\nApplying K-Means...\n")

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

label = kmeans.fit_predict(df_scaled)

df['Kmeans'] = label

print(df.head())


plt.figure(figsize=(10, 6))

plt.scatter(
    df_scaled[:, 0],
    df_scaled[:, 1],
    c=label,
    cmap='viridis',
    s=50
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='X',
    s=300,
    color='red',
    label='Centroids'
)

plt.title("K-Means Clustering (K=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.show()




print("\nApplying Elbow Method...\n")

inertias = []

for k in range(1, 11):

    km = KMeans(
        n_clusters=k,
        random_state=42
    )

    km.fit(df_scaled)

    inertias.append(km.inertia_)

print("\nInertias:")
print(inertias)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertias,
    marker='o',
    linestyle='--'
)

plt.title("The Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.show()



print("\nEvaluating Clusters...\n")

sil_score = silhouette_score(
    df_scaled,
    label
)

db_score = davies_bouldin_score(
    df_scaled,
    label
)

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Score: {db_score:.3f}")



print("\nApplying Hierarchical Clustering...\n")

linked = linkage(
    df_scaled,
    method='ward'
)

plt.figure(figsize=(12, 6))

dendrogram(linked)

plt.title("Dendrogram")
plt.xlabel("Games")
plt.ylabel("Distance")

plt.show()

hc = AgglomerativeClustering(
    n_clusters=3
)

hc_labels = hc.fit_predict(df_scaled)

df['Hierarchical'] = hc_labels

print(df.head())



print("\nApplying DBSCAN...\n")

dbscan = DBSCAN(
    eps=1.5,
    min_samples=3
)

dbscan_labels = dbscan.fit_predict(
    df_scaled
)

df['DBSCAN'] = dbscan_labels

print("Unique Clusters:")
print(np.unique(dbscan_labels))

plt.figure(figsize=(10, 6))

plt.scatter(
    df_scaled[:, 0],
    df_scaled[:, 1],
    c=dbscan_labels,
    cmap='plasma',
    s=50
)

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()




print("\nApplying PCA...\n")

pca = PCA(
    n_components=2
)

pca_data = pca.fit_transform(
    df_scaled
)

print("Shape after PCA:", pca_data.shape)

plt.figure(figsize=(10, 6))

plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=label,
    cmap='viridis',
    s=50
)

plt.title("PCA Visualization of K-Means Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar()

plt.show()




print("\nApplying Isolation Forest...\n")

iso = IsolationForest(
    contamination=0.10,
    random_state=42
)

anomaly_labels = iso.fit_predict(
    df_scaled
)

df['Anomaly'] = anomaly_labels

print("\nAnomaly Counts:")
print(df['Anomaly'].value_counts())

print("\nAnomalous Games:")
print(df[df['Anomaly'] == -1])

plt.figure(figsize=(10, 6))

plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=anomaly_labels,
    cmap='coolwarm',
    s=60
)

plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()




print("\nFinal Comparison\n")

print(f"K-Means Silhouette Score: {sil_score:.3f}")
print(f"K-Means Davies-Bouldin Score: {db_score:.3f}")

print(f"Hierarchical Clusters: {len(np.unique(hc_labels))}")

print(f"DBSCAN Clusters: {len(np.unique(dbscan_labels))}")

print(f"Anomalies Detected: {(anomaly_labels == -1).sum()}")


print("\nConclusion")

print("1. K-Means successfully grouped Steam games into clusters.")
print("2. Elbow Method helped determine the optimal K value.")
print("3. Silhouette Score and Davies-Bouldin Index evaluated cluster quality.")
print("4. Hierarchical Clustering produced a dendrogram showing cluster hierarchy.")
print("5. DBSCAN identified density-based clusters and noise points.")
print("6. PCA reduced dimensionality for visualization.")
print("7. Isolation Forest detected anomalous Steam games.")
print("8. Multiple unsupervised learning techniques were successfully applied.")

print("\nProject Completed Successfully.")