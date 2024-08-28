import os
import csv

# 定义一个函数，将 "hh:mm:ss" 格式的时间转换为以秒为单位的浮点数
def time_str_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# 定义输入和输出文件夹
input_folder = 'F:/edge_consistency_v1/input_files/timestamps'
output_folder = 'F:/edge_consistency_v1/input_files/durations'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 处理 1 到 40 的所有 timestamps 文件
for i in range(1, 41):
    input_file = os.path.join(input_folder, f'timestamps{i}.csv')
    output_file = os.path.join(output_folder, f'duration{i}.csv')
    
    durations = []
    
    # 读取 timestamps 文件并计算持续时间
    with open(input_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            start_time = time_str_to_seconds(row[0])  # 将时间字符串转换为秒
            end_time = time_str_to_seconds(row[1])    # 将时间字符串转换为秒
            duration = end_time - start_time
            durations.append([duration])
    
    # 将持续时间写入新的 duration 文件
    with open(output_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Duration'])  # 写入标题
        csv_writer.writerows(durations)   # 写入每一行的持续时间

    print(f"Processed {input_file}, output saved to {output_file}")
