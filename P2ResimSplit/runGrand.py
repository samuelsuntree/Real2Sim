import csv
import subprocess
import os
import time
from datetime import datetime

######################################
# Function to run the Julia pipeline
######################################
def run_swimmer_pipeline(a1, a2, a3, log_file_path):
    """
    Call Swimmer_Pipeline.jl with three parameters (video index a1,
    crop index a2, and a3 for duration).
    Save console outputs and possible errors/timeout info to log_file_path.
    """
    cmd = [
        "julia",                # Replace with the absolute path if needed
        "Swimmer_Pipeline.jl",  # Ensure the path to your Julia script is correct
        str(a1),
        str(a2),
        str(a3)
    ]
    
    timeout_seconds = 900  # 15 minutes
    start_time = datetime.now()
    
    # Write initial info to the log file
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"[{start_time}] INFO: Starting Julia script.\n")
        log_file.write(f"[{start_time}] CMD: {' '.join(cmd)}\n")

    try:
        # Run the Julia script, capture stdout/stderr, enforce timeout
        result = subprocess.run(
            cmd,
            check=True,            # Raise CalledProcessError if returncode != 0
            timeout=timeout_seconds,
            capture_output=True,   # Capture STDOUT/STDERR
            text=True              # Decode to text instead of bytes
        )
        
        # If we get here, script ended normally (returncode=0, no timeout)
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write("\n[STDOUT]\n")
            log_file.write(result.stdout)
            log_file.write("\n[STDERR]\n")
            log_file.write(result.stderr)
            
            end_time = datetime.now()
            elapsed_sec = (end_time - start_time).total_seconds()
            log_file.write(f"\n[{end_time}] INFO: Finished successfully. Elapsed={elapsed_sec:.2f} seconds.\n")

    except subprocess.TimeoutExpired as e:
        # Julia script took longer than timeout_seconds
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write("\n[ERROR] The process exceeded the time limit (15 minutes).\n")
            log_file.write(f"[ERROR] Partial STDOUT/STDERR:\n{e.output}\n")  # May be partial or None
            end_time = datetime.now()
            elapsed_sec = (end_time - start_time).total_seconds()
            log_file.write(f"[{end_time}] ERROR: Timeout. Elapsed={elapsed_sec:.2f} seconds.\n")

    except subprocess.CalledProcessError as e:
        # Julia script returned a non-zero exit code
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write("\n[ERROR] Julia script exited with a non-zero status.\n")
            log_file.write(f"[ERROR] Return code: {e.returncode}\n")
            log_file.write(f"[ERROR] STDOUT:\n{e.stdout}\n")
            log_file.write(f"[ERROR] STDERR:\n{e.stderr}\n")
            end_time = datetime.now()
            elapsed_sec = (end_time - start_time).total_seconds()
            log_file.write(f"[{end_time}] ERROR: Process failed. Elapsed={elapsed_sec:.2f} seconds.\n")

    except Exception as e:
        # Catch any other unexpected errors
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write("\n[ERROR] An unexpected exception occurred.\n")
            log_file.write(f"[ERROR] {str(e)}\n")
            end_time = datetime.now()
            elapsed_sec = (end_time - start_time).total_seconds()
            log_file.write(f"[{end_time}] ERROR: Process failed due to exception. Elapsed={elapsed_sec:.2f} seconds.\n")


######################################
# Main batch processing logic
######################################
def main():
    # Directory containing the CSV files with durations
    csv_dir = r"C:\Users\10521\Documents\GitHub\Real2Sim\P1VideoCapture\input_files\durations"
    
    # Define the directory for individual log files
    logs_dir = r"C:\Users\10521\Documents\GitHub\Real2Sim\P2ResimSplit\logResim"
    # Ensure the directory exists
    os.makedirs(logs_dir, exist_ok=True)

    # Process 40 video files
    for j in range(1, 41):
        timestampFilePath = os.path.join(csv_dir, f"duration{j}.csv")
        
        # Read durations from CSV, skip header
        durations = []
        with open(timestampFilePath, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header row
            for row in reader:
                if not row:
                    continue
                duration_val = float(row[0]) * 2.5
                durations.append(duration_val)
        
        print(f"[INFO] Video {j}: read {len(durations)} durations -> {durations}")
        
        # For each crop i, run the Julia pipeline
        for i, duration in enumerate(durations, start=1):
            # Create a unique log file name for each run
            start_time = datetime.now()
            start_time_str = start_time.strftime('%Y%m%d_%H%M%S')
            log_file_name = f"log_j{j}_crop{i}_{start_time_str}.txt"

            # Put the log file into the logs_dir
            log_file_path = os.path.join(logs_dir, log_file_name)
            
            print(f"[INFO] Launching Julia: video={j}, crop={i}, duration={duration:.2f}")
            print(f"[INFO] Log file: {log_file_path}")

            # Call the runner
            run_swimmer_pipeline(j, i, duration, log_file_path)

            print(f"[INFO] Finished iteration. Check '{log_file_path}' for details.\n")
    
    print("[INFO] All videos processed successfully.")


######################################
# Program entry
######################################
if __name__ == "__main__":
    main()
