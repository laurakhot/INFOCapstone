import pandas as pd
import gzip

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
    'total_app_crash': ['mean', 'sum', 'count', 'max'],
    'avg_processor_time': 'mean',
    'avg_memory_utilization': 'mean',
    'avg_cpu_temp': 'mean',
    'avg_boot_time': 'mean',
    'uptime_days': 'mean'
}).round(2)

host_stats.columns = ['avg_crashes', 'total_crashes', 'record_count', 'max_crashes',
                      'avg_cpu', 'avg_memory', 'avg_temp', 'avg_boot', 'avg_uptime']

extreme_hosts = host_stats[host_stats['avg_crashes'] > 7000].sort_values('avg_crashes', ascending=False)

print("EXTREME HIGH-CRASH HOSTS (avg > 7000 crashes)")
print("=" * 120)
print(extreme_hosts.to_string())
print("\n")

top_20 = host_stats.sort_values('avg_crashes', ascending=False).head(20)
print("TOP 20 HOSTS BY AVERAGE CRASHES")
print("=" * 120)
print(top_20.to_string())
