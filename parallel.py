import threading
import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name])

# Create threads
thread1 = threading.Thread(target=run_script, args=("pose1.py",))
thread2 = threading.Thread(target=run_script, args=("pose2.py",))

# Start threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()
