import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import os
from math import sqrt

# Set hyper-parameters
learning_rate = 0.0001
batch_size = 512
epochs = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device\n")

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
        if count == 10:
            break

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


class SEMG(nn.Module):
    def __init__(self):
        super(SEMG, self).__init__()
        self.conv1_1 = nn.Conv1d(8, 64, 3)
        self.conv1_2 = nn.Conv1d(64, 64, 3)
        self.pooling = nn.MaxPool1d(3)

        self.conv2_1 = nn.Conv1d(64, 128, 3)
        self.conv2_2 = nn.Conv1d(128, 128, 3)

        self.dropout = nn.Dropout(0.5)
        self.gru_1 = nn.GRU(128, 150, batch_first=True)
        self.gru_2 = nn.GRU(150, 150, batch_first=True)

        self.fc = nn.Linear(150 * 57, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 8, 188)
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pooling(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))

        x = x.transpose(1, 2)
        x, _ = self.gru_1(self.dropout(x))
        x = self.tanh(x)
        x, _ = self.gru_2(self.dropout(x))
        x = self.tanh(x)
        x = self.dropout(x)

        x = x.reshape(batch_size, -1)
        x = self.softmax(self.fc(x))

        return x


# Create an instance of the neural network
model = SEMG()
model.to(device)

# Load dataset
print("Loading training set")
sample_train, label_train = load_data("EMG_data/train")
train_data = SEMG_Dataset(sample_train, label_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

print("Loading testing set")
sample_test, label_test = load_data("EMG_data/test")
test_data = SEMG_Dataset(sample_test, label_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch, count=5):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward + Backward + Update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % count == count - 1:
            print(f"Epoch {epoch + 1: <4d} Batch {batch_idx + 1: <4d} Loss = {running_loss/count:.4f}")
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            sample, labels = data
            sample, labels = sample.to(device), labels.to(device)
            predict = model(sample)
            _, result = torch.max(predict.data, dim=1)
            total += labels.size(0)
            answer = torch.max(labels.data, dim=1)[1]
            correct += (result == answer).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy = {accuracy:.4f}%")
    return accuracy


if __name__ == "__main__":
    # model.load_state_dict(torch.load("weights/weight.pth"))
    max_accuracy = 0
    for epoch in range(epochs):
        # Train + Test
        train(epoch)
        current_accuracy = test()

        # Save model
        torch.save(model.state_dict(), f"./weights/epoch_{epoch + 1}.pth")
        if current_accuracy > max_accuracy:
            torch.save(model.state_dict(), "./weights/best.pth")
            print("Update the best model")
            max_accuracy = current_accuracy
        print()
