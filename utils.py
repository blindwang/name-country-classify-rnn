import os

# import tensorflow as tf
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorboard as tb

from tensorboardX import SummaryWriter


def bulid_tensorboard_writer(dir_type, model_type):
    # tensorboard writer
    log_dir = Path("logs") / dir_type / model_type
    train_log_dir = log_dir / "train"
    test_log_dir = log_dir / "test"

    # Clear any logs from previous runs
    shutil.rmtree(train_log_dir, ignore_errors=True)
    shutil.rmtree(test_log_dir, ignore_errors=True)

    # train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    # test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))

    train_summary_writer = SummaryWriter(logdir=str(train_log_dir))
    test_summary_writer = SummaryWriter(logdir=str(test_log_dir))
    return train_summary_writer, test_summary_writer


def plot_loss_acc(experiment_id, flag, csv_path, pic_path):
    csv_path = Path('tmp') / csv_path
    os.makedirs('tmp', exist_ok=True)

    if csv_path.exists():
        dfw = pd.read_csv(csv_path)
    else:
        experiment_id = experiment_id
        experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
        dfw = experiment.get_scalars()
        dfw.to_csv(csv_path, index=False)

    dfw_roundtrip = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(dfw_roundtrip, dfw)

    # Filter the DataFrame to only validation data, which is what the subsequent
    # analyses and visualization will be focused on.
    # print(dfw)
    dfw_acc = dfw[(dfw['run'].str.startswith(flag)) & (dfw['tag'] == "accuracy")]
    dfw_loss = dfw[(dfw['run'].str.startswith(flag)) & (dfw['tag'] == "loss")]
    # print(dfw)
    dfw_loss['value'] = dfw_loss['value'].rolling(window=100).mean()
    # Get the optimizer value for each row of the validation DataFrame.
    optimizer_acc = dfw_acc.run.apply(lambda run: run.split(",")[0].split('\\')[1])
    optimizer_loss = dfw_loss.run.apply(lambda run: run.split(",")[0].split('\\')[1])
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=dfw_acc, x="step", y="value",
                 hue=optimizer_acc, ci='sd').set_title("accuracy")
    plt.subplot(1, 2, 2)
    # print(dfw_loss.shape)
    # dfw_loss.dropna(axis=0, inplace=True)
    # print(dfw_loss.shape)
    sns.lineplot(data=dfw_loss, x="step", y="value",
                 hue=optimizer_loss, ci='sd').set_title("loss")

    pic_path = Path('result') / pic_path
    os.makedirs('result', exist_ok=True)
    plt.savefig(pic_path)
    plt.show()
