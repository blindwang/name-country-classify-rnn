from utils import plot_loss_acc

# 绘制loss和acc曲线
# experiment_id = "H0Ibe0U2RE6aWDPSMNXe3g"  # base
# plot_loss_acc(experiment_id, "compare_base", "compare_base.csv", "new_compare_base.png")

# https://tensorboard.dev/experiment/iGVlJWlJTPSlTbtjiEN21Q
# experiment_id = "iGVlJWlJTPSlTbtjiEN21Q"  # lanent_lr
# plot_loss_acc(experiment_id, "lenet", "compare_lanent_lr.csv", "compare_lanent_lr.png")

# experiment_id = "QkfOqRRsRKmdS5sXwVtXSQ"
# plot_loss_acc(experiment_id, "lenet", "compare_lanent_optim.csv", "compare_lanent_optim.png")

# https://tensorboard.dev/experiment/dK5Qh2aoQvOXmIN3EuvgLQ
experiment_id = "dK5Qh2aoQvOXmIN3EuvgLQ"  # lenet optim
plot_loss_acc(experiment_id, "t", "compare_lenet_optim.csv", "compare_lenet_optim.png")

# https://tensorboard.dev/experiment/OHY2cAbrRTeuzjzuNulHMQ/
experiment_id = "OHY2cAbrRTeuzjzuNulHMQ" # alexnet lr
plot_loss_acc(experiment_id, "t", "compare_alexnet_optim.csv", "compare_alexnet_optim.png")

# import matplotlib.pyplot as plt
#
# import os
# import fnmatch
# import pandas as pd
# import re
# import seaborn as sns
# import numpy as np
#
# plt.style.use('ggplot')
#
# # 定义要读取的文件夹路径
# folder_path = 'tmp'
#
# # 读取文件夹下的所有.csv文件
# csv_files = []
# for root, dirnames, filenames in os.walk(folder_path):
#     for filename in fnmatch.filter(filenames, '*.csv'):
#         csv_files.append(os.path.join(root, filename))
#
# # 筛选出文件名中包含train的文件
# train_files = []
# dfs = {}
# for file in csv_files:
#     if 'test' in os.path.basename(file):
#         train_files.append(file)
#         optim_name = re.search('optim_(.*)_test', file).group(1)
#         dfs[optim_name] = pd.read_csv(file)
#
# colors = ['blue', 'green', 'yellow', 'pink']
# ind = 0
# cmap = plt.cm.get_cmap('viridis')
# for label, df in dfs.items():
#     y = df['Value']
#     y_smoothed = df['Value'].rolling(window=100).mean()
#     x = df['Step']
#     color = cmap((ind+1)/len(dfs))
#     plt.plot(x, y, color=color)
#     # plt.plot(x, y_smoothed, color=color, label=label)
#     ind+=1
#
# # 设置图表标题和坐标轴标签
# plt.title('Accuracy Comparison')
# plt.xlabel('Step')
# plt.ylabel('Accuracy')
#
# # 显示图例
# plt.legend()
# plt.savefig('result/compare_base')
# # 显示图表
# plt.show()



