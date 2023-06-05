import torch.nn as nn
import torch
import torch.optim as optim
import random
import math
import numpy as np
import string

from train import train
from evaluate import evaluate
from dataloader import build_dataloader
from model import RNN, BiRNN, GRU, BiGRU, LSTM
from utils import bulid_tensorboard_writer
from earlystop import EarlyStopping
from optimizer import get_scheduler

"""随机种子"""
seed = 2023

# Python随机数生成器的种子
random.seed(seed)

# Numpy随机数生成器的种子
np.random.seed(seed)

# Pytorch随机数生成器的种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 判断是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""网络"""
models_ls = ["birnn", "bigru", "lstm"]

"""定义batch_size列表"""
batch_size = [8, 16]

"""默认超参数"""
num_epochs = 40
learning_rate = 0.1

for model_name in models_ls:
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters)
    n_hidden = 128
    layer_num = 2
    for batchsize in batch_size:        
        """读取数据"""
        trainloader, testloader, trainset, testset = build_dataloader(batchsize)
        n_categories = trainset.n_categories
        if model_name == 'lstm':
            model = LSTM(n_letters, n_hidden, n_categories, num_layers=layer_num)
        elif model_name == 'birnn':
            model = BiRNN(n_letters, n_hidden, n_categories, num_layers=layer_num)
        elif model_name == 'bigru':
            model = BiGRU(n_letters, n_hidden, n_categories, num_layers=layer_num)

        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = get_scheduler(optimizer,
                                  3 * math.ceil(len(trainset) / batchsize),
                                  num_epochs * math.ceil(len(trainset) / batchsize))
        earlystop = EarlyStopping(patience=5, model_name=model_name + f'_batch_size_{batchsize}')

        print(f"Training {model_name}... w/ batch_size={batchsize}")

        """tensorboard writer"""
        train_summary_writer, test_summary_writer = bulid_tensorboard_writer(f"compare_batchsize/{model_name}",
                                                                             f'batch_size_{batchsize}')

        # 训练模型
        for epoch in range(num_epochs):
            train(trainloader, model, criterion, optimizer, epoch, device, train_summary_writer, scheduler)
            acc = evaluate(testloader, model, epoch, device, test_summary_writer)
            earlystop(-acc, model)
            if earlystop.early_stop:
                break

        print('Finished Training')
