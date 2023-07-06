import matplotlib.pyplot as plt
import time

# 初始化时间、数据数组
times = []
data1 = []
data2 = []

# 在每次循环中添加数据
start_time = time.time()  # 获取开始时间
for i in range(10):
    # 获取当前时间戳和随机数作为数据
    timestamp = int((time.time() - start_time) * 1000)  # 将时间戳转换为毫秒
    value1 = i + 1
    value2 = (i + 1) * 2
    
    # 将时间和数据添加到数组中
    times.append(timestamp)
    data1.append(value1)
    data2.append(value2)
    
    # 等待一段时间
    time.sleep(0.1)

# 绘制随时间变化的图表
plt.plot(times, data1, label='Data 1')
plt.plot(times, data2, label='Data 2')

plt.xlabel('时间（毫秒）')
plt.ylabel('数值')
plt.title('随时间变化的数据')
plt.legend()  # 显示图例
plt.show()
