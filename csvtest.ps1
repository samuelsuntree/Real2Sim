# 读取CSV文件内容并处理
$csvLines = Get-Content -Path "F:\edge_consistency_v1\input_files\durations\duration2.csv"

# 跳过标题行并手动解析
foreach ($line in $csvLines[1..$($csvLines.Length-1)]) {
    $columns = $line.Trim() -split ","
    $duration = $columns[0].Trim()  # 去除空白字符并获取 Duration 列
    Write-Host "Duration: $duration seconds"
}
