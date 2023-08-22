import torch.nn as nn

class SEMG(nn.Module):
    def __init__(self):
        super(SEMG, self).__init__()
        self.conv1_1 = nn.Conv1d(8, 64, kernel_size=3)
        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=3)
        self.pooling = nn.MaxPool1d(3)

        self.conv2_1 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=3)

        self.dropout = nn.Dropout(0.5)
        self.gru_1 = nn.GRU(128, 150, batch_first=True)
        self.gru_2 = nn.GRU(150, 150, batch_first=True)

        self.fc = nn.Linear(150 * 57, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 8, 188)  # batch_size * 8 * 186
        x = self.relu(self.conv1_1(x))  # batch_size * 64 * 186
        x = self.relu(self.conv1_2(x))  # batch_size * 64 * 184
        x = self.pooling(x)  # batch_size * 64 * 61

        x = self.relu(self.conv2_1(x))  # batch_size * 128 * 59
        x = self.relu(self.conv2_2(x))  # batch_size * 128 * 57

        x = x.transpose(1, 2)   # batch_size * 57 * 128
        x, _ = self.gru_1(self.dropout(x))  # batch_size * 57 * 150
        x = self.tanh(x)
        x, _ = self.gru_2(self.dropout(x))  # batch_size * 57 * 150
        x = self.tanh(x)
        x = self.dropout(x)

        x = x.reshape(batch_size, -1)
        x = self.softmax(self.fc(x))

        return x
