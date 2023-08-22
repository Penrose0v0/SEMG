import torch
from torch.utils.data import Dataset
import csv
import os
from math import sqrt

def normalize(x, ave=-1.0979379441417457e-05, var=4.726647396007718e-08):
    return (x - ave) / sqrt(var)

def load_data(csv_dir, window_len=188, offset=20):
    count = 0
    samples = []
    labels = []
    csv_files = os.listdir(csv_dir)
    for csv_file in csv_files:
        # Filter wrong files
        if ".csv" not in csv_file:
            continue

        # Reading CSV files
        print(f"Reading {csv_file}... ", end='')
        gesture = eval(csv_file[: csv_file.find('_')])
        participant = eval(csv_file[csv_file.find('_') + 1: csv_file.find('.')])

        with open(csv_dir + '/' + csv_file, 'r') as file:
            reader = csv.reader(file)
            rows = [[eval(data) for data in row] for row in reader]

            for i in range(window_len, len(rows), offset):
                time_last = rows[i][0]
                time_first = rows[i-window_len][0]
                if time_last - time_first > window_len + 100:
                    continue

                sample = [[normalize(data) for data in row[1: -2]] for row in rows[i-window_len: i]]
                sample = torch.tensor(sample)
                label = torch.tensor([0] * 7, dtype=torch.float16)
                label[gesture-1] += 1

                samples.append(sample)
                labels.append(label)
        print("Over! ")

        count += 1
        # if count == 10:
        #     break

    print(f"Complete! Total: {count}\n")
    return samples, labels


class SEMG_Dataset(Dataset):
    def __init__(self, channels, labels):
        self.channels = channels
        self.labels = labels

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, item):
        return self.channels[item], self.labels[item]
