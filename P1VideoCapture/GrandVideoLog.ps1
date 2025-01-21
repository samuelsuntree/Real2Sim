# Set the path to the specific Python environment
$pythonExePath = "C:/Users/10521/.conda/envs/Thomas/python.exe"

# Set the path to the Python script
$pythonScriptPath = "F:\\edge_consistency_v1\\main.py"

#$logFilePath = "F:\\edge_consistency_v1\\process_log.txt"
# Set the path to the log folder
$logFolderPath = "F:\\edge_consistency_v1\\output_files\\logs"

# Ensure the log folder exists
if (-not (Test-Path $logFolderPath)) {
    New-Item -ItemType Directory -Path $logFolderPath
}

# Generate the log file path with a timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFilePath = "$logFolderPath\\process_log_$timestamp.txt"


# Loop for processing 40 video files
for ($j = 1; $j -le 40; $j++) {

    # Set the path to the corresponding timestamp file
    $timestampFilePath = "F:\\edge_consistency_v1\\input_files\\durations\\duration$j.csv"

    # Read the content of the CSV file and manually parse it
    $csvLines = Get-Content -Path $timestampFilePath

    # Initialize a counter
    $i = 1

    # Create an array to store all durations
    $durations = @()

    # Skip the header row and manually parse each line
    foreach ($line in $csvLines[1..$($csvLines.Length-1)]) {
        $columns = $line.Trim() -split ","
        $duration = $columns[0].Trim()  # Trim whitespace and get the Duration column

        # Convert to float and store it
        $durationFloat = [double]$duration
        $durations += $durationFloat

        # Print and check if the values read are correct
        Write-Host "Video number $i for video $j has duration $durationFloat seconds"

        # Increment the counter
        $i++
    }

    # After confirming the durations are correct, start executing the Python script
    $i = 1
    foreach ($duration in $durations) {
        $startTime = Get-Date

        Write-Host "Processing video $j, crop $i with duration $duration seconds"
        Add-Content -Path $logFilePath -Value "[$startTime] Starting process for video $j, crop $i with duration $duration seconds"

        # Call the Python script in the specific Python environment and pass the parameters
        & $pythonExePath $pythonScriptPath $j $i $duration

        # Record the end time and the elapsed time
        $endTime = Get-Date
        $elapsedTime = $endTime - $startTime
        Write-Host "Finished processing video $j, crop $i. Duration: $($elapsedTime.TotalSeconds) seconds"
        Add-Content -Path $logFilePath -Value "[$endTime] Finished processing video $j, crop $i. Duration: $($elapsedTime.TotalSeconds) seconds"

        # Increment the counter to process the next crop
        $i++
    }

    Write-Host "All crops for video $j processed successfully."
}

Write-Host "All videos processed successfully."
