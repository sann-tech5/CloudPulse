
from alerts import check_resource_health

# Test 1: Normal resource usage
print("\n--- TEST 1: NORMAL ---")
print(check_resource_health(30, 500))

# Test 2: High CPU usage
print("\n--- TEST 2: HIGH CPU ---")
print(check_resource_health(75, 500))

# Test 3: Critical memory usage
print("\n--- TEST 3: CRITICAL MEMORY ---")
print(check_resource_health(40, 1600))

# Test 4: Both CPU and memory critical
print("\n--- TEST 4: BOTH CRITICAL ---")
print(check_resource_health(95, 1800))
