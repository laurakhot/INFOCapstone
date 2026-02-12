import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
print(f"Loaded {len(df)} records")

print("\nConverting numeric columns...")
numeric_cols = [
    'p90_processor_time', 'avg_processor_time', 'max_cpu_usage',
    'p90_memory_utilization', 'avg_memory_utilization', 'max_memory_usage',
    'p10_battery_health', 'avg_battery_health', 'cpu_count', 'memory_count',
    'memory_size_gb', 'max_battery_temperature', 'avg_battery_temperature',
    'p90_battery_temperature', 'avg_cpu_temp', 'p90_cpu_temp',
    'avg_battery_discharge', 'p90_battery_discharge', 'avg_boot_time',
    'p90_boot_time', 'uptime_days', 'total_app_crash'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['time'] = pd.to_datetime(df['time'], errors='coerce')

print("\nData shape:", df.shape)
print("\nBasic statistics for total_app_crash:")
print(df['total_app_crash'].describe())

print("\nAggregating by hostname...")
host_agg = df.groupby('host_name').agg({
    'total_app_crash': ['sum', 'mean', 'max', 'count'],
    'avg_processor_time': 'mean',
    'avg_memory_utilization': 'mean',
    'avg_battery_health': 'mean',
    'avg_cpu_temp': 'mean',
    'avg_boot_time': 'mean',
    'uptime_days': 'mean',
    'max_cpu_usage': 'mean',
    'max_memory_usage': 'mean'
}).reset_index()

host_agg.columns = ['_'.join(col).strip('_') for col in host_agg.columns.values]
host_agg = host_agg.dropna()

print(f"Aggregated to {len(host_agg)} unique hosts")

fig = plt.figure(figsize=(20, 12))

print("\nCreating visualizations...")

plt.subplot(3, 4, 1)
crash_dist = host_agg['total_app_crash_sum'].value_counts().head(20)
plt.bar(range(len(crash_dist)), crash_dist.values)
plt.xlabel('Crash Count')
plt.ylabel('Number of Hosts')
plt.title('Distribution of Total Crashes per Host')
plt.xticks(range(len(crash_dist)), crash_dist.index, rotation=45)

plt.subplot(3, 4, 2)
plt.scatter(host_agg['avg_processor_time_mean'], host_agg['total_app_crash_sum'], alpha=0.5)
plt.xlabel('Avg CPU Usage (%)')
plt.ylabel('Total Crashes')
plt.title('CPU Usage vs Crashes')

plt.subplot(3, 4, 3)
plt.scatter(host_agg['avg_memory_utilization_mean'], host_agg['total_app_crash_sum'], alpha=0.5)
plt.xlabel('Avg Memory Usage (%)')
plt.ylabel('Total Crashes')
plt.title('Memory Usage vs Crashes')

plt.subplot(3, 4, 4)
plt.scatter(host_agg['avg_boot_time_mean'], host_agg['total_app_crash_sum'], alpha=0.5)
plt.xlabel('Avg Boot Time (s)')
plt.ylabel('Total Crashes')
plt.title('Boot Time vs Crashes')

plt.subplot(3, 4, 5)
plt.scatter(host_agg['uptime_days_mean'], host_agg['total_app_crash_sum'], alpha=0.5)
plt.xlabel('Avg Uptime (days)')
plt.ylabel('Total Crashes')
plt.title('Uptime vs Crashes')

plt.subplot(3, 4, 6)
plt.scatter(host_agg['avg_battery_health_mean'], host_agg['total_app_crash_sum'], alpha=0.5)
plt.xlabel('Avg Battery Health (%)')
plt.ylabel('Total Crashes')
plt.title('Battery Health vs Crashes')

print("\nPerforming clustering analysis...")
features_for_clustering = [
    'avg_processor_time_mean', 'avg_memory_utilization_mean',
    'avg_boot_time_mean', 'uptime_days_mean', 'max_cpu_usage_mean',
    'max_memory_usage_mean', 'total_app_crash_sum'
]

cluster_data = host_agg[features_for_clustering].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.subplot(3, 4, 7)
scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('Clusters (PCA)')
plt.colorbar(scatter, label='Cluster')

plt.subplot(3, 4, 8)
crash_values = cluster_data['total_app_crash_sum'].values
scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=crash_values, cmap='Reds', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('Crash Intensity (PCA)')
plt.colorbar(scatter, label='Total Crashes')

cluster_data_with_labels = cluster_data.copy()
cluster_data_with_labels['cluster'] = clusters

plt.subplot(3, 4, 9)
cluster_crash_avg = cluster_data_with_labels.groupby('cluster')['total_app_crash_sum'].mean()
plt.bar(cluster_crash_avg.index, cluster_crash_avg.values, color=['red', 'orange', 'yellow', 'green'])
plt.xlabel('Cluster')
plt.ylabel('Avg Total Crashes')
plt.title('Average Crashes by Cluster')

plt.subplot(3, 4, 10)
high_crash_hosts = host_agg.nlargest(20, 'total_app_crash_sum')
plt.barh(range(len(high_crash_hosts)), high_crash_hosts['total_app_crash_sum'])
plt.ylabel('Host Rank')
plt.xlabel('Total Crashes')
plt.title('Top 20 Hosts by Crash Count')

plt.subplot(3, 4, 11)
bins = [0, 10, 50, 100, 500, float('inf')]
labels = ['0-10', '11-50', '51-100', '101-500', '500+']
host_agg['crash_category'] = pd.cut(host_agg['total_app_crash_sum'], bins=bins, labels=labels)
crash_cat_counts = host_agg['crash_category'].value_counts().sort_index()
plt.bar(range(len(crash_cat_counts)), crash_cat_counts.values, color=['green', 'yellow', 'orange', 'red', 'darkred'])
plt.xticks(range(len(crash_cat_counts)), crash_cat_counts.index, rotation=45)
plt.ylabel('Number of Hosts')
plt.title('Host Distribution by Crash Severity')

plt.subplot(3, 4, 12)
correlation_features = ['avg_processor_time_mean', 'avg_memory_utilization_mean', 
                        'avg_boot_time_mean', 'total_app_crash_sum']
corr_data = host_agg[correlation_features].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('crash_analysis_overview.png', dpi=300, bbox_inches='tight')
print("\nSaved: crash_analysis_overview.png")

print("\nCluster Analysis Summary:")
for i in range(4):
    cluster_subset = cluster_data_with_labels[cluster_data_with_labels['cluster'] == i]
    print(f"\nCluster {i}:")
    print(f"  Size: {len(cluster_subset)} hosts")
    print(f"  Avg Crashes: {cluster_subset['total_app_crash_sum'].mean():.2f}")
    print(f"  Avg CPU: {cluster_subset['avg_processor_time_mean'].mean():.2f}%")
    print(f"  Avg Memory: {cluster_subset['avg_memory_utilization_mean'].mean():.2f}%")
    print(f"  Avg Boot Time: {cluster_subset['avg_boot_time_mean'].mean():.2f}s")

print("\nHigh Risk Indicators:")
high_crash_threshold = host_agg['total_app_crash_sum'].quantile(0.9)
high_crash_hosts = host_agg[host_agg['total_app_crash_sum'] >= high_crash_threshold]
print(f"\nHosts with >{high_crash_threshold:.0f} crashes: {len(high_crash_hosts)}")
print(f"Avg CPU: {high_crash_hosts['avg_processor_time_mean'].mean():.2f}%")
print(f"Avg Memory: {high_crash_hosts['avg_memory_utilization_mean'].mean():.2f}%")
print(f"Avg Boot Time: {high_crash_hosts['avg_boot_time_mean'].mean():.2f}s")

print("\nAnalysis complete!")
