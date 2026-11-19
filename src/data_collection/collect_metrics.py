
import csv
import requests
from datetime import datetime
from pathlib import Path

PROMETHEUS_URL = "http://localhost:9090"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "infrastructure_metrics.csv"

ML_OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "metrics.csv"

def get_metric(query, default=0.0):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=15
    )

    response.raise_for_status()
    data = response.json()

    if data["status"] != "success":
        raise RuntimeError("Prometheus query failed")

    results = data["data"]["result"]

    if not results:
        return default

    return float(results[0]["value"][1])


def collect_metrics():
    print("Collecting CloudPulse 2.0 metrics...")

    # Existing container metrics
    cpu_query = '''
    100 * sum(rate(
        container_cpu_usage_seconds_total{id="/", job="cadvisor"}[1m]
    ))
    '''

    memory_query = '''
    container_memory_usage_bytes{id="/", job="cadvisor"}
    '''

    cpu_percent = get_metric(cpu_query)
    memory_bytes = get_metric(memory_query)
    memory_mb = memory_bytes / (1024 * 1024)

    # New host CPU metric
    host_cpu_query = '''
    100 * (1 - avg(rate(
        node_cpu_seconds_total{mode="idle"}[1m]
    )))
    '''

    host_cpu_percent = get_metric(host_cpu_query)

    # New host memory metrics
    total_memory_query = '''
    node_memory_MemTotal_bytes
    '''

    available_memory_query = '''
    node_memory_MemAvailable_bytes
    '''

    total_memory = get_metric(total_memory_query)
    available_memory = get_metric(available_memory_query)

    host_memory_used_mb = (total_memory - available_memory) / (1024 * 1024)

    host_memory_percent = (
        ((total_memory - available_memory) / total_memory) * 100
        if total_memory > 0 else 0
    )

    # New disk usage metric
    disk_query = '''
    100 * (1 -
        sum(node_filesystem_avail_bytes{mountpoint="/", fstype!~"tmpfs|overlay"})
        /
        sum(node_filesystem_size_bytes{mountpoint="/", fstype!~"tmpfs|overlay"})
    )
    '''

    disk_percent = get_metric(disk_query)

    # New network metrics
    network_receive_query = '''
    sum(rate(node_network_receive_bytes_total{device!~"lo"}[1m]))
    '''

    network_transmit_query = '''
    sum(rate(node_network_transmit_bytes_total{device!~"lo"}[1m]))
    '''

    network_receive = get_metric(network_receive_query)
    network_transmit = get_metric(network_transmit_query)

    timestamp = datetime.now().isoformat(timespec="seconds")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0

    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "cpu_percent",
                "memory_bytes",
                "memory_mb",
                "host_cpu_percent",
                "host_memory_used_mb",
                "host_memory_percent",
                "disk_percent",
                "network_receive_bytes_per_sec",
                "network_transmit_bytes_per_sec"
            ])

        writer.writerow([
            timestamp,
            round(cpu_percent, 2),
            int(memory_bytes),
            round(memory_mb, 2),
            round(host_cpu_percent, 2),
            round(host_memory_used_mb, 2),
            round(host_memory_percent, 2),
            round(disk_percent, 2),
            round(network_receive, 2),
            round(network_transmit, 2)
        ])

            # Save original 4-column metrics for the ML pipeline
    ml_file_exists = (
        ML_OUTPUT_FILE.exists() and ML_OUTPUT_FILE.stat().st_size > 0
    )

    with open(ML_OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not ml_file_exists:
            writer.writerow([
                "timestamp",
                "cpu_percent",
                "memory_bytes",
                "memory_mb"
            ])

        writer.writerow([
            timestamp,
            round(cpu_percent, 2),
            int(memory_bytes),
            round(memory_mb, 2)
        ])

    print("Metrics collected successfully!")
    print(f"Container CPU: {cpu_percent:.2f}%")
    print(f"Container Memory: {memory_mb:.2f} MB")
    print(f"Host CPU: {host_cpu_percent:.2f}%")
    print(f"Host Memory: {host_memory_percent:.2f}%")
    print(f"Disk Usage: {disk_percent:.2f}%")
    print(f"Network Receive: {network_receive:.2f} bytes/sec")
    print(f"Network Transmit: {network_transmit:.2f} bytes/sec")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    collect_metrics()
