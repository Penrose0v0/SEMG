import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SEMG
from dataloader import SEMG_Dataset, load_data

# Set hyper-parameters
learning_rate = 0.0001
batch_size = 512
epochs = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device\n")

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


def train(epoch_num, count=25):
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
            print(f"Epoch {epoch_num + 1: <4d} Batch {batch_idx + 1: <4d} Loss = {running_loss / count:.4f}")
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
    # model.load_state_dict(torch.load("weights/best.pth"))
    max_accuracy = 0
    for epoch in range(epochs):
        # Train + Test
        train(epoch)
        current_accuracy = test()

        # Save model
        torch.save(model.state_dict(), f"./weights/epoch/epoch_{epoch + 1}.pth")
        if current_accuracy > max_accuracy:
            torch.save(model.state_dict(), "./weights/best.pth")
            print("Update the best model")
            max_accuracy = current_accuracy
        print()
    print(f"Training finished! Max Accuracy: {max_accuracy:.4f}")
