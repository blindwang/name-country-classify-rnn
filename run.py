import torch.nn as nn
import torch
import torch.optim as optim
import random
import numpy as np
import math
import warnings
import string
warnings.filterwarnings("ignore")

from train import train
from evaluate import evaluate
from dataloader import build_dataloader
from model import RNN, BiRNN, GRU, BiGRU, LSTM, BiLSTM
from utils import bulid_tensorboard_writer
from optimizer import get_scheduler
from earlystop import EarlyStopping

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

models_ls = ["rnn", "birnn", "gru", "bigru", "lstm", "bilstm"]

# 判断是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

for model_name in models_ls[-2:]:
    print(f"Training {model_name} model...")

    """默认超参数"""
    batch_size = 8
    learning_rate = 0.075
    num_epochs = 40

    """tensorboard writer"""
    train_summary_writer, test_summary_writer = bulid_tensorboard_writer("compare_base", model_name)

    """默认dataloader"""
    trainloader, testloader, trainset, testset = build_dataloader(batch_size)

    n_categories = trainset.n_categories
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters)
    n_hidden = 128
    """定义默认网络"""

    if model_name == 'gru':
        model = GRU(n_letters, n_hidden, n_categories)
    elif model_name == 'lstm':
        model = LSTM(n_letters, n_hidden, n_categories)
    elif model_name == 'birnn':
        model = BiRNN(n_letters, n_hidden, n_categories)
    elif model_name == 'bigru':
        model = BiGRU(n_letters, n_hidden, n_categories)
    elif model_name == 'bilstm':
        model = BiLSTM(n_letters, n_hidden, n_categories)
    else:
        model = RNN(n_letters, n_hidden, n_categories)

    model.to(device)
    
    """定义默认损失函数和优化器"""
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(optimizer,
                              3 * math.ceil(len(trainset) / batch_size),
                              num_epochs * math.ceil(len(trainset) / batch_size))

    """设置早停策略"""
    earlystop = EarlyStopping(patience=5, model_name=model_name)

    # 训练模型
    for epoch in range(num_epochs):
        train(trainloader, model, criterion, optimizer, epoch, device, train_summary_writer, scheduler)
        acc = evaluate(testloader, model, epoch, device, test_summary_writer)
        earlystop(-acc, model)
        if earlystop.early_stop:
            break

    print('Finished Training')
