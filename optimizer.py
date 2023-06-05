import transformers
import torch
import re


def _get_resnet_name_to_layer(pt_model):
    """获取resnet中参数的分层，从0开始，层数越高表示越顶层，全连是最顶层（5），返回name: layer的字典"""
    name_to_layer = {}
    for k, v in pt_model.named_parameters():
        if k.startswith('conv1.') or k.startswith('bn1.'):
            name_to_layer[k] = 0
        elif k.startswith('layer1.'):
            name_to_layer[k] = 1
        elif k.startswith('layer2.'):
            name_to_layer[k] = 2
        elif k.startswith('layer3.'):
            name_to_layer[k] = 3
        elif k.startswith('layer4.'):
            name_to_layer[k] = 4
        elif k.startswith('fc.'):
            name_to_layer[k] = 5
        else:
            print("ERROR")
    return name_to_layer, 5


def _get_vgg_name_to_layer(pt_model):
    """获取vgg中参数的分层，从0开始，层数越高表示越顶层，全连和辅助分类器是最顶层（5），返回name: layer的字典"""
    name_to_layer = {}
    for k, v in pt_model.named_parameters():
        if v.shape[0] == 64:
            name_to_layer[k] = 0
        elif v.shape[0] == 128:
            name_to_layer[k] = 1
        elif v.shape[0] == 256:
            name_to_layer[k] = 2
        elif v.shape[0] == 512:
            name_to_layer[k] = 3
        elif k.startswith('classifier'):
            name_to_layer[k] = 4
        else:
            print("ERROR")
    return name_to_layer, 4


def _get_alexnet_name_to_layer(pt_model):
    """获取alexnet中参数的分层，从0开始，层数越高表示越顶层，全连和辅助分类器是最顶层（5），返回name: layer的字典"""
    name_to_layer = {}
    for k, v in pt_model.named_parameters():
        if k.startswith('features'):
            name_to_layer[k] = 0
        elif k.startswith('classifier'):
            name_to_layer[k] = 1
        else:
            print("ERROR")
    return name_to_layer, 1

def _get_lenet_name_to_layer(pt_model):
    """获取lenet中参数的分层，从0开始，层数越高表示越顶层，全连和辅助分类器是最顶层（5），返回name: layer的字典"""
    name_to_layer = {}
    for k, v in pt_model.named_parameters():
        if k.startswith('conv'):
            name_to_layer[k] = 0
        elif k.startswith('fc'):
            name_to_layer[k] = 1
        else:
            print("ERROR")
    return name_to_layer, 1


def _get_google_name_to_layer(pt_model):
    """获取vgg中参数的分层，从0开始，层数越高表示越顶层，全连是最顶层（5），返回name: layer的字典"""
    name_to_layer = {}
    for k, v in pt_model.named_parameters():
        if k.startswith('conv1.'):
            name_to_layer[k] = 0
        elif k.startswith('conv2.') or k.startswith('conv3.'):
            name_to_layer[k] = 1
        elif k.startswith('inception3'):
            name_to_layer[k] = 2
        elif k.startswith('inception4'):
            name_to_layer[k] = 3
        elif k.startswith('inception5'):
            name_to_layer[k] = 4
        elif k.startswith('fc.'):
            name_to_layer[k] = 5
        elif k.startswith('aux'):
            name_to_layer[k] = 5
        else:
            print("ERROR")
    return name_to_layer, 5


def _get_name_to_layer(pt_model, model_name):
    if model_name.startswith('google'):
        return _get_google_name_to_layer(pt_model)
    elif model_name.startswith('resnet'):
        return _get_resnet_name_to_layer(pt_model)
    elif model_name.startswith('vgg'):
        return _get_vgg_name_to_layer(pt_model)
    elif model_name.startswith('alexnet'):
        return _get_alexnet_name_to_layer(pt_model)
    elif model_name.startswith('lenet'):
        return _get_lenet_name_to_layer(pt_model)
    else:
        raise ValueError


def _get_resnet_no_decay_param_names(pt_model):
    """获取resnet中不需要weight_decay的参数的名字"""
    no_decay_param_names = []
    for k, v in pt_model.named_parameters():
        if re.search("\.?bn[0-9][\.]", k) is not None:  # 如果正则表达式匹配到了batchnorm
            no_decay_param_names.append(k)
            # print(k)
        elif k.endswith('.bias'):
            no_decay_param_names.append(k)
            # print(k)
    return set(no_decay_param_names)


def get_optim(pt_model, model_name, optim_name="adam", lr=1e-5, weight_decay=0.01,
              filter_bias_and_bn=True, lr_decay_factor=None):
    """获取优化器"""
    param_groups = []
    # 计算不同层的学习率，高一层是当前层的lr_decay_factor倍
    if lr_decay_factor is not None:
        name_to_layer, num_layers = _get_name_to_layer(pt_model, model_name=model_name)
        layer_scales = list(lr_decay_factor ** (num_layers - i) for i in range(num_layers + 1))
    # 过滤不需要weight decay的参数
    if filter_bias_and_bn is True and weight_decay != 0.0:
        no_decay_param_names = _get_resnet_no_decay_param_names(pt_model)
        for k, v in pt_model.named_parameters():
            if k in no_decay_param_names:
                param_groups.append({'params': v, 'lr': lr * layer_scales[name_to_layer[k]]})
            else:
                param_groups.append(
                    {'params': v, 'lr': lr * layer_scales[name_to_layer[k]], 'weight_decay': weight_decay})
    else:
        for k, v in pt_model.named_parameters():
            param_groups.append({'params': v, 'lr': lr * layer_scales[name_to_layer[k]], 'weight_decay': weight_decay})
    # 构建optimizer
    if optim_name.lower() == 'adam':
        opt = torch.optim.AdamW(param_groups, lr=lr)
    elif optim_name.lower() == 'sgd':
        opt = torch.optim.SGD(param_groups, lr=lr)
    elif optim_name.lower() == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr=lr)
    elif optim_name.lower() == 'adadelta':
        opt = torch.optim.Adadelta(param_groups, lr=lr)
    elif optim_name.lower() == 'adagrad':
        opt = torch.optim.Adagrad(param_groups, lr=lr)

    return opt


def get_scheduler(optim, num_warmup_steps, num_training_steps):
    # num_warmup_steps是warm up阶段的步数
    # num_training_steps是训练总共需要的步数
    return transformers.get_linear_schedule_with_warmup(optim, num_warmup_steps, num_training_steps)
