
def check_resource_health(cpu_percent, memory_mb):
    alerts = []
    recommendations = []

    # CPU monitoring
    if cpu_percent >= 90:
        alerts.append("CRITICAL: CPU usage is extremely high!")
        recommendations.append(
            "Reduce workload or increase CPU capacity."
        )
    elif cpu_percent >= 70:
        alerts.append("WARNING: CPU usage is high.")
        recommendations.append(
            "Monitor running processes and optimize CPU usage."
        )

    # Memory monitoring
    if memory_mb >= 1536:
        alerts.append("CRITICAL: Memory usage is extremely high!")
        recommendations.append(
            "Close unnecessary processes or increase memory capacity."
        )
    elif memory_mb >= 1024:
        alerts.append("WARNING: Memory usage is high.")
        recommendations.append(
            "Check memory-consuming processes and optimize memory usage."
        )

    if not alerts:
        alerts.append("HEALTHY: All resource levels are normal.")
        recommendations.append(
            "Continue monitoring resource usage."
        )

    return {
        "status": (
            "CRITICAL" if any("CRITICAL" in alert for alert in alerts)
            else "WARNING" if any("WARNING" in alert for alert in alerts)
            else "HEALTHY"
        ),
        "alerts": alerts,
        "recommendations": recommendations
    }
