import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import gzip
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
columns = [
    'karl_id', 'host_name', 'model_name', 'hardware_make', 'karl_last_seen',
    'auth_username', 'serial_number', 'group_id', 'tenant_id', 'platform',
    'metric_category', 'measure_name', 'time', 'p90_processor_time',
    'avg_processor_time', 'max_cpu_usage', 'p90_memory_utilization',
    'avg_memory_utilization', 'max_memory_usage', 'p10_battery_health',
    'avg_battery_health', 'cpu_count', 'memory_count', 'memory_size_gb',
    'driver_vendor', 'os', 'wifi_mac_add', 'driver_version', 'driver_date',
    'os_version', 'driver', 'agent_id', 'performance_status', 'device_status',
    'max_battery_temperature', 'avg_battery_temperature', 'p90_battery_temperature',
    'avg_cpu_temp', 'p90_cpu_temp', 'avg_battery_discharge', 'p90_battery_discharge',
    'avg_boot_time', 'p90_boot_time', 'uptime_days', 'total_app_crash'
]

chunk_size = 100000
sample_data = []
with gzip.open('/Users/rolfson/Downloads/Capstone/000.gz', 'rt') as f:
    for i, chunk in enumerate(pd.read_csv(f, sep='|', names=columns, chunksize=chunk_size)):
        sample_data.append(chunk)
        if i >= 4:
            break

df = pd.concat(sample_data, ignore_index=True)

numeric_cols = ['avg_processor_time', 'max_cpu_usage', 'avg_memory_utilization',
                'max_memory_usage', 'avg_battery_health', 'cpu_count', 'memory_size_gb',
                'avg_cpu_temp', 'avg_boot_time', 'p90_boot_time', 'uptime_days', 'total_app_crash']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

host_stats = df.groupby('host_name').agg({
    'total_app_crash': 'mean',
    'avg_processor_time': 'mean',
    'avg_memory_utilization': 'mean',
    'avg_cpu_temp': 'mean',
    'avg_boot_time': 'mean',
    'uptime_days': 'mean',
    'max_cpu_usage': 'mean',
    'max_memory_usage': 'mean'
}).dropna()

print(f"Analyzing {len(host_stats)} hosts")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(host_stats)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n=== CLUSTERING COMPARISON ===\n")

fig = plt.figure(figsize=(20, 16))

print("1. K-Means Clustering (k=4)")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_db = davies_bouldin_score(X_scaled, kmeans_labels)
print(f"   Silhouette Score: {kmeans_silhouette:.4f} (higher is better)")
print(f"   Davies-Bouldin Score: {kmeans_db:.4f} (lower is better)")

plt.subplot(3, 3, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'K-Means (k=4)\nSilhouette: {kmeans_silhouette:.3f}')

host_stats['kmeans_cluster'] = kmeans_labels
for i in range(4):
    cluster_data = host_stats[host_stats['kmeans_cluster'] == i]
    print(f"   Cluster {i}: {len(cluster_data)} hosts, avg crashes: {cluster_data['total_app_crash'].mean():.0f}")

print("\n2. K-Means Clustering (k=5)")
kmeans5 = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans5_labels = kmeans5.fit_predict(X_scaled)
kmeans5_silhouette = silhouette_score(X_scaled, kmeans5_labels)
kmeans5_db = davies_bouldin_score(X_scaled, kmeans5_labels)
print(f"   Silhouette Score: {kmeans5_silhouette:.4f}")
print(f"   Davies-Bouldin Score: {kmeans5_db:.4f}")

plt.subplot(3, 3, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans5_labels, cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'K-Means (k=5)\nSilhouette: {kmeans5_silhouette:.3f}')

host_stats['kmeans5_cluster'] = kmeans5_labels
for i in range(5):
    cluster_data = host_stats[host_stats['kmeans5_cluster'] == i]
    print(f"   Cluster {i}: {len(cluster_data)} hosts, avg crashes: {cluster_data['total_app_crash'].mean():.0f}")

print("\n3. DBSCAN (Density-Based)")
dbscan = DBSCAN(eps=1.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"   Clusters found: {n_clusters_dbscan}")
print(f"   Noise points: {n_noise}")
if n_clusters_dbscan > 1:
    mask = dbscan_labels != -1
    dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
    print(f"   Silhouette Score: {dbscan_silhouette:.4f}")

plt.subplot(3, 3, 3)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'DBSCAN\nClusters: {n_clusters_dbscan}, Noise: {n_noise}')

host_stats['dbscan_cluster'] = dbscan_labels
for i in range(-1, n_clusters_dbscan):
    cluster_data = host_stats[host_stats['dbscan_cluster'] == i]
    label = "Noise" if i == -1 else f"Cluster {i}"
    print(f"   {label}: {len(cluster_data)} hosts, avg crashes: {cluster_data['total_app_crash'].mean():.0f}")

print("\n4. Hierarchical Clustering (Ward)")
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
hierarchical_db = davies_bouldin_score(X_scaled, hierarchical_labels)
print(f"   Silhouette Score: {hierarchical_silhouette:.4f}")
print(f"   Davies-Bouldin Score: {hierarchical_db:.4f}")

plt.subplot(3, 3, 4)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Hierarchical (Ward)\nSilhouette: {hierarchical_silhouette:.3f}')

host_stats['hierarchical_cluster'] = hierarchical_labels
for i in range(4):
    cluster_data = host_stats[host_stats['hierarchical_cluster'] == i]
    print(f"   Cluster {i}: {len(cluster_data)} hosts, avg crashes: {cluster_data['total_app_crash'].mean():.0f}")

print("\n5. Gaussian Mixture Model (GMM)")
gmm = GaussianMixture(n_components=4, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
gmm_db = davies_bouldin_score(X_scaled, gmm_labels)
print(f"   Silhouette Score: {gmm_silhouette:.4f}")
print(f"   Davies-Bouldin Score: {gmm_db:.4f}")
print(f"   BIC: {gmm.bic(X_scaled):.2f} (lower is better)")

plt.subplot(3, 3, 5)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Gaussian Mixture\nSilhouette: {gmm_silhouette:.3f}')

host_stats['gmm_cluster'] = gmm_labels
for i in range(4):
    cluster_data = host_stats[host_stats['gmm_cluster'] == i]
    print(f"   Cluster {i}: {len(cluster_data)} hosts, avg crashes: {cluster_data['total_app_crash'].mean():.0f}")

print("\n6. K-Means with Optimal k (Elbow Method)")
inertias = []
silhouettes = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

plt.subplot(3, 3, 6)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

plt.subplot(3, 3, 7)
plt.plot(k_range, silhouettes, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(True)

optimal_k = k_range[silhouettes.index(max(silhouettes))]
print(f"   Optimal k by Silhouette: {optimal_k}")

print("\n7. Crash-Based Segmentation")
crash_percentiles = [0, 50, 75, 90, 95, 100]
crash_bins = host_stats['total_app_crash'].quantile([p/100 for p in crash_percentiles]).values
crash_bins = pd.Series(crash_bins).drop_duplicates().values
crash_labels = [f'Segment {i+1}' for i in range(len(crash_bins)-1)]
host_stats['crash_segment'] = pd.cut(host_stats['total_app_crash'], bins=crash_bins, labels=crash_labels, include_lowest=True, duplicates='drop')

plt.subplot(3, 3, 8)
segment_counts = host_stats['crash_segment'].value_counts().sort_index()
colors = ['green', 'yellow', 'orange', 'red', 'darkred'][:len(segment_counts)]
plt.bar(range(len(segment_counts)), segment_counts.values, color=colors)
plt.xticks(range(len(segment_counts)), segment_counts.index, rotation=45)
plt.ylabel('Number of Hosts')
plt.title('Crash-Based Segmentation')

for label in crash_labels:
    segment_data = host_stats[host_stats['crash_segment'] == label]
    if len(segment_data) > 0:
        print(f"   {label}: {len(segment_data)} hosts, avg crashes: {segment_data['total_app_crash'].mean():.0f}")

print("\n8. Feature Comparison Across Methods")
comparison_data = {
    'Method': ['K-Means (k=4)', 'K-Means (k=5)', 'Hierarchical', 'GMM'],
    'Silhouette': [kmeans_silhouette, kmeans5_silhouette, hierarchical_silhouette, gmm_silhouette],
    'Davies-Bouldin': [kmeans_db, kmeans5_db, hierarchical_db, gmm_db]
}
comparison_df = pd.DataFrame(comparison_data)

plt.subplot(3, 3, 9)
x = np.arange(len(comparison_df))
width = 0.35
plt.bar(x - width/2, comparison_df['Silhouette'], width, label='Silhouette', alpha=0.8)
plt.bar(x + width/2, comparison_df['Davies-Bouldin'], width, label='Davies-Bouldin', alpha=0.8)
plt.xlabel('Method')
plt.ylabel('Score')
plt.title('Clustering Quality Comparison')
plt.xticks(x, comparison_df['Method'], rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.savefig('advanced_clustering_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: advanced_clustering_comparison.png")

print("\n=== EXTREME HOSTS BY METHOD ===")
for method in ['kmeans_cluster', 'hierarchical_cluster', 'gmm_cluster']:
    print(f"\n{method.replace('_', ' ').title()}:")
    cluster_means = host_stats.groupby(method)['total_app_crash'].mean().sort_values(ascending=False)
    extreme_cluster = cluster_means.index[0]
    extreme_hosts = host_stats[host_stats[method] == extreme_cluster].sort_values('total_app_crash', ascending=False).head(5)
    print(f"  Extreme cluster: {extreme_cluster} (avg: {cluster_means.iloc[0]:.0f} crashes)")
    print(f"  Top 5 hosts:")
    for idx, row in extreme_hosts.iterrows():
        print(f"    {idx}: {row['total_app_crash']:.0f} crashes")

print("\n=== RECOMMENDATION ===")
best_method = comparison_df.loc[comparison_df['Silhouette'].idxmax(), 'Method']
print(f"Best clustering method: {best_method}")
print(f"  Highest Silhouette Score: {comparison_df['Silhouette'].max():.4f}")
print(f"  Lowest Davies-Bouldin Score: {comparison_df['Davies-Bouldin'].min():.4f}")

print("\nAnalysis complete!")
