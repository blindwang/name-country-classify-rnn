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
# models_ls = {"resnet18": ResNet(), "vgg": VGG(), "alexnet": AlexNet(), "googlenet": GoogleNet(), "lenet": LeNet5()}

models_ls = ["birnn", "bigru", "bilstm"]

"""定义优化器列表"""
lr_ls = [0.005, 0.05, 0.075, 0.1]

"""默认超参数"""
batch_size = 8
num_epochs = 40

for model_name in models_ls[-1:]:
    """读取数据"""
    trainloader, testloader, trainset, testset = build_dataloader(batch_size)

    n_categories = trainset.n_categories
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters)
    n_hidden = 128

    for learning_rate in lr_ls:
        if model_name == 'gru':
            model = GRU(n_letters, n_hidden, n_categories)
        elif model_name == 'lstm':
            model = LSTM(n_letters, n_hidden, n_categories)
        elif model_name == 'birnn':
            model = BiRNN(n_letters, n_hidden, n_categories)
        elif model_name == 'bigru':
            model = BiGRU(n_letters, n_hidden, n_categories)
        else:
            model = RNN(n_letters, n_hidden, n_categories)

        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = get_scheduler(optimizer,
                                  3 * math.ceil(len(trainset) / batch_size),
                                  num_epochs * math.ceil(len(trainset) / batch_size))
        earlystop = EarlyStopping(patience=5, model_name=model_name + f'_lr_{learning_rate}')

        print(f"Training {model_name}... w/ lr={learning_rate}")

        """tensorboard writer"""
        train_summary_writer, test_summary_writer = bulid_tensorboard_writer(f"compare_lr/{model_name}",
                                                                             f'lr_{learning_rate}')

        # 训练模型
        for epoch in range(num_epochs):
            train(trainloader, model, criterion, optimizer, epoch, device, train_summary_writer, scheduler)
            acc = evaluate(testloader, model, epoch, device, test_summary_writer)
            earlystop(-acc, model)
            if earlystop.early_stop:
                break

        print('Finished Training')
