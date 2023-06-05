根据任务，分为比较优化器（`compare_optim.py`）、更改网络架构（`run.py`）、模型层数比较（`compare_layer_nums.py`）、比较学习率（`compare_lr.py`）、比较batch_size（`compare_batchsize.py`）

代码结构说明

- 构建dataloader使用`dataloader.py`，其中还会用到`data.py`转换字符的编码格式。

- 构建模型使用`model.py`，包含RNN、BiRNN、GRU、BiGRU、LSTM、BiLSTM

- 训练和验证使用`train.py`和`evaluate.py`

- 深度学习参数调整的一些技巧（带有warmup的学习率调节器）使用`optmizer.py`

- 早停策略使用`earlystop.py`

- 数据可视化使用的函数放在`utils.py`
  数据上传至tensorboard dev中，最后用`plot.py`完成符合要求的对比图

其他文件夹的说明

- 数据保存在`/data`文件夹

- 训练记录保存在`/logs`文件夹

- 可视化的折线图保存在`/result`文件夹

- 训练过程中保存的模型权重在`/models_weight`文件夹

- 从tensorboard下载的数据保存在`/tmp`文件夹