# �����ض�Python������·��
$pythonExePath = "C:/Users/10521/.conda/envs/Thomas/python.exe"

# ����Python�ű���·��
$pythonScriptPath = "F:\edge_consistency_v1\main.py"

# ����ʱ����ļ���·��
$timestampFilePath = "F:\edge_consistency_v1\input_files\durations\duration2.csv"

# ��ȡCSV�ļ����ݲ��ֶ�����
$csvLines = Get-Content -Path $timestampFilePath

# ��ʼ��������
$i = 1

# ����һ�����飬���ڴ洢���е� duration
$durations = @()

# ���������в��ֶ�����ÿһ��
foreach ($line in $csvLines[1..$($csvLines.Length-1)]) {
    $columns = $line.Trim() -split ","
    $duration = $columns[0].Trim()  # ȥ���հ��ַ�����ȡ Duration ��

    # ת��Ϊ���������洢
    $durationFloat = [double]$duration
    $durations += $durationFloat

    # ��ӡ������ȡ��ֵ�Ƿ���ȷ
    Write-Host "Video number $i has duration $durationFloat seconds"

    # ����������
    $i++
}

# ȷ�� durations ��ȡ����󣬿�ʼִ�� Python �ű�
$i = 1
foreach ($duration in $durations) {
    Write-Host "Processing video number $i with duration $duration seconds"

    # �����ض�Python�����µ�Python�ű������ݲ���
    & $pythonExePath $pythonScriptPath $i $duration

    # ������������������һ����Ƶ
    $i++
}

Write-Host "All videos processed successfully."
