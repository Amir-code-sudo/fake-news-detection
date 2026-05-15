import subprocess
import sys

print("Starting server on http://127.0.0.1:5000")
print("Press Ctrl+C to stop\n")

try:
    subprocess.check_call([sys.executable, "app.py"])
except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    print(f"Error: {e}")
