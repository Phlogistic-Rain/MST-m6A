import torch
import torch.utils.data as Data
import pandas as pd

def window_str(input_str, size):
    mid = len(input_str) // 2

    size = min(size, len(input_str))

    left_size = size // 2
    right_size = size - left_size

    truncated_str = input_str[mid - left_size: mid + right_size]

    return truncated_str


def data_process(file, window_size):
    df = pd.read_csv(file, sep='\t', header=0)

    data = []
    labels = []
    for idx, row in df.iterrows():
        s = row['text']
        label = row['label']

        s = window_str(s, window_size)
        label = int(label)

        data.append(s)
        labels.append(label)

    # print(data)
    # print(labels)

    # return data, torch.cuda.LongTensor(labels)
    return data, torch.cuda.LongTensor(labels)


class MyDataSet(Data.Dataset):
    def __init__(self, data, label, mutation=False, mutation_rate=None):
        self.data = data
        self.label = label
        self.mutation_rate = mutation_rate
        self.mutation = mutation

        if mutation and not mutation_rate:
            raise ValueError('未提供 mutation rate')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        d = self.data[idx]

        if self.mutation:
            d = mutate_dna(d, self.mutation_rate)

        return d, self.label[idx]
