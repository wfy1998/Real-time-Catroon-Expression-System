import torch
import pandas as pd
csv_filename = 'data/wfy4/wfy4_2D_data.csv'
# 从 CSV 文件中读取数据和标签
df_read = pd.read_csv(csv_filename)

# 获取读取的数据和标签
read_data_list = df_read['data'].apply(eval).tolist()
read_label_list = df_read['label'].tolist()

# 转换读取的数据为 Torch 张量
read_data = torch.tensor(read_data_list)
read_label = torch.tensor(read_label_list)  # 假设只有一个标签

# 打印读取的数据和标签
print("读取的数据:", read_data.shape)
print("读取的标签:", read_label)