# 设置特定Python环境的路径
$pythonExePath = "C:/Users/10521/.conda/envs/Thomas/python.exe"

# 设置Python脚本的路径
$pythonScriptPath = "F:\edge_consistency_v1\main.py"

#$logFilePath = "F:\edge_consistency_v1\process_log.txt"
# 设置日志文件夹路径
$logFolderPath = "F:\edge_consistency_v1\output_files\logs"

# 确保日志文件夹存在
if (-not (Test-Path $logFolderPath)) {
    New-Item -ItemType Directory -Path $logFolderPath
}

# 生成包含时间戳的日志文件路径
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFilePath = "$logFolderPath\process_log_$timestamp.txt"


# 处理40个视频文件的循环
for ($j = 1; $j -le 40; $j++) {

    # 设置对应的时间戳文件的路径
    $timestampFilePath = "F:\edge_consistency_v1\input_files\durations\duration$j.csv"

    # 读取CSV文件内容并手动解析
    $csvLines = Get-Content -Path $timestampFilePath

    # 初始化计数器
    $i = 1

    # 创建一个数组，用于存储所有的 duration
    $durations = @()

    # 跳过标题行并手动解析每一行
    foreach ($line in $csvLines[1..$($csvLines.Length-1)]) {
        $columns = $line.Trim() -split ","
        $duration = $columns[0].Trim()  # 去除空白字符并获取 Duration 列

        # 转换为浮点数并存储
        $durationFloat = [double]$duration
        $durations += $durationFloat

        # 打印并检查读取的值是否正确
        Write-Host "Video number $i for video $j has duration $durationFloat seconds"

        # 递增计数器
        $i++
    }

    # 确认 durations 读取无误后，开始执行 Python 脚本
    $i = 1
    foreach ($duration in $durations) {
        $startTime = Get-Date

        Write-Host "Processing video $j, crop $i with duration $duration seconds"
        Add-Content -Path $logFilePath -Value "[$startTime] Starting process for video $j, crop $i with duration $duration seconds"

        # 调用特定Python环境下的Python脚本并传递参数
        & $pythonExePath $pythonScriptPath $j $i $duration

        # 记录结束时间和持续时间
        $endTime = Get-Date
        $elapsedTime = $endTime - $startTime
        Write-Host "Finished processing video $j, crop $i. Duration: $($elapsedTime.TotalSeconds) seconds"
        Add-Content -Path $logFilePath -Value "[$endTime] Finished processing video $j, crop $i. Duration: $($elapsedTime.TotalSeconds) seconds"

        # 递增计数器，处理下一个 crop
        $i++
    }

    Write-Host "All crops for video $j processed successfully."
}

Write-Host "All videos processed successfully."
