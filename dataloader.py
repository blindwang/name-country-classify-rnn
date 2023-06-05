import torch
from data import *
import random


def collate(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = torch.stack(targets)
    return inputs, targets


class NameDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train=True):
        self.categories = []
        self.lines = []
        self.n_categories = 0

        for filename in findFiles(root_dir + '/*.txt'):
            category = filename.split('/')[-1].split('.')[0]
            self.categories.append(category)
            lines = readLines(filename)

            if is_train:
                lines = lines[:int(0.8 * len(lines))]
            else:
                lines = lines[int(0.8 * len(lines)):]

            self.lines += [(line, self.n_categories) for line in lines]
            self.n_categories += 1

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line, category_idx = self.lines[idx]
        category = torch.tensor(category_idx, dtype=torch.long)
        line_tensor = lineToTensor(line)
        return line_tensor, category


def build_dataloader(batch_size):
    train_dataset = NameDataset('data/names/', is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_dataset = NameDataset('data/names/', is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, test_loader, train_dataset, test_loader
