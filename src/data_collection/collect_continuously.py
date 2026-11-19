
import time
from collect_metrics import collect_metrics

print("CloudPulse continuous monitoring started!")
print("Collecting metrics every 10 seconds.")
print("Press CTRL + C to stop.\n")

try:
    while True:
        try:
            collect_metrics()
            print("-" * 40)

        except Exception as error:
            print(f"Temporary collection error: {error}")
            print("Will try again in 10 seconds...")

        time.sleep(10)

except KeyboardInterrupt:
    print("\nCloudPulse monitoring stopped.")
