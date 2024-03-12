import time
import threading
import numpy as np
import queue
import csv
import concurrent.futures
import os


JOB_NUM = 99  # 发送请求的个数

# 在opt-1.3B上的实验数据 单位: ms
x = [1, 4, 16, 64, 256, 512, 1024]
first_time = [5.88, 5.93, 6.57, 8.04, 23.8, 43.9, 98.5]
next_time = [5.13, 5.11, 5.16, 5.22, 5.52, 5.72, 5.82]

# 通过实验数据拟合每次迭代推理时间
z1 = np.polyfit(x, first_time, 1)
p1 = np.poly1d(z1)

z2 = np.polyfit(x, next_time, 1)
p2 = np.poly1d(z2)
#print(p1(90))
#######################################################
import matplotlib.pyplot as plt
'''
with open('example.txt', 'r') as file:
    lines = file.readlines()
tuple_list = [eval(line.strip()) for line in lines]
sorted_tuple_list = sorted(tuple_list, key=lambda x: x[0])
ids, values = zip(*sorted_tuple_list)
plt.bar(ids, values, color='blue')
plt.xlabel('ID')
plt.ylabel('JCT')
plt.title('所有任务的JCT')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择适用于你的系统的中文字体
ave = 60.2017146578 # 你想要的阈值
plt.axhline(y=ave, color='red', linestyle='--', label='平均值')
plt.show()
'''
'''
rate=[1,2,4,8,16]
ave=[41.4,61.1,73.4,80.2,83.9]
plt.plot(rate,ave)
plt.xlabel('arrival_rate')
plt.ylabel('Average_JCT')

plt.show()
'''
queue_num=[4,8,12,16]
ave=[62.17,62.25,63.28,64.00]
plt.plot(queue_num,ave)
plt.xlabel('queue_num')
plt.ylabel('Average_JCT')
plt.ylim(0, 100)  # 根据实际情况调整纵轴的最小值和最大值
plt.show()