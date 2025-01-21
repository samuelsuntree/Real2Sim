import csv
import subprocess
import os
import sys
import time
from datetime import datetime

######################################
# Global debug switch:
#   True  => print all Julia output in real-time to console
#   False => only write detailed logs; console stays brief
######################################
debug_mode = False

def run_swimmer_pipeline(a1, a2, a3, log_file_path):
    """
    Launch the Julia script (Swimmer_Pipeline.jl) with parameters (a1, a2, a3).
    We capture stdout in real-time:
      - Always write all lines to the log file
      - If debug_mode=True, also print every line to the console
    If the process exceeds timeout_seconds, kill it and record an error.
    """

    cmd = [
        "julia",                # Adjust path if needed
        "Swimmer_Pipeline.jl",  
        str(a1),
        str(a2),
        str(a3)
    ]

    timeout_seconds = 900  # 15 minutes
    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"[{start_time_str}] INFO: Starting Julia script.\n")
        log_file.write(f"[{start_time_str}] CMD: {' '.join(cmd)}\n\n")

        # Start the subprocess in line-buffered mode
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,        
            stderr=subprocess.STDOUT,      
            text=True,                     
            bufsize=1                      
        )

        start_loop_time = time.time()

        # Read lines until process ends or times out
        while True:
            line = process.stdout.readline()
            if line:
                # Always write to the log
                log_file.write(line)
                log_file.flush()
                # Only print if debug_mode is True
                if debug_mode:
                    print(line, end='')

            # Check if the process has exited
            if process.poll() is not None:
                # Read any leftover lines after exit
                leftover = process.stdout.read()
                if leftover:
                    log_file.write(leftover)
                    log_file.flush()
                    if debug_mode:
                        print(leftover, end='')
                break

            # Check for timeout
            elapsed = time.time() - start_loop_time
            if elapsed > timeout_seconds:
                process.kill()
                msg = f"[ERROR] Timeout after {elapsed:.2f} seconds.\n"
                log_file.write(msg)
                if debug_mode:
                    print(msg, end='')
                break

        retcode = process.returncode
        end_time = datetime.now()
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        elapsed_sec = (end_time - start_time).total_seconds()

        if retcode is None:
            log_file.write(f"[{end_time_str}] ERROR: Process killed. Elapsed={elapsed_sec:.2f} s.\n")
            if debug_mode:
                print(f"[ERROR] Process killed. Elapsed={elapsed_sec:.2f} s.")
        elif retcode == 0:
            log_file.write(f"\n[{end_time_str}] INFO: Finished successfully. Elapsed={elapsed_sec:.2f} s.\n")
            if debug_mode:
                print(f"[INFO] Finished successfully. Elapsed={elapsed_sec:.2f} s.")
        else:
            log_file.write(f"\n[{end_time_str}] ERROR: Non-zero exit code ({retcode}). Elapsed={elapsed_sec:.2f} s.\n")
            if debug_mode:
                print(f"[ERROR] Non-zero exit code ({retcode}). Elapsed={elapsed_sec:.2f} s.")


######################################
# Main batch processing logic
######################################
def main():
    # Directory containing your CSV files
    csv_dir = r"C:\Users\10521\Documents\GitHub\Real2Sim\P1VideoCapture\input_files\durations"

    for j in range(1, 41):
        timestampFilePath = os.path.join(csv_dir, f"duration{j}.csv")
        
        # Read durations
        durations = []
        with open(timestampFilePath, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header
            for row in reader:
                if not row:
                    continue
                duration_val = float(row[0]) * 2.5
                durations.append(duration_val)
        
        if debug_mode:
            print(f"[INFO] Video {j}: read {len(durations)} durations -> {durations}")
        
        for i, duration in enumerate(durations, start=1):
            run_start = datetime.now()
            run_start_str = run_start.strftime('%Y%m%d_%H%M%S')
            log_file_name = f"log_j{j}_crop{i}_{run_start_str}.txt"
            
            if debug_mode:
                print(f"\n[INFO] Launching Julia: video={j}, crop={i}, duration={duration:.2f}")
                print(f"[INFO] Log file -> {log_file_name}")
            
            run_swimmer_pipeline(j, i, duration, log_file_name)
            
            if debug_mode:
                print(f"[INFO] Finished iteration for j={j}, i={i}. See '{log_file_name}' for details.\n")

    print("[INFO] All videos processed successfully.")


if __name__ == "__main__":
    main()
