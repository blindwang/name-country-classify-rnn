# import tensorflow as tf
from tqdm import tqdm
import torch


def evaluate(testloader, net, epoch, device, test_summary_writer):
    with tqdm(testloader, desc=f'Evaluate epoch {epoch}') as tbar:
        # 在测试集上测试准确率
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # 在tqdm中展示loss

                tbar.set_postfix(eval_acc=f'{100 * correct / total:.2f}%')
                # 更新进度条
                tbar.update()
        acc = 100 * correct / total
        # with test_summary_writer.as_default():
        #     tf.summary.scalar('accuracy', acc, step=epoch)
        test_summary_writer.add_scalar('accuracy', acc, epoch)

    return acc
