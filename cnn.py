import csv
import tqdm
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from torchsummary import summary  # pip install torch-summary
from torch.utils.data import DataLoader

from utils import get_device, accuracy
from data import MyDataset
from config import DIGITS_MAT_PATH

class EarlyStopping:
    def __init__(self, patience=30, min_delta=0):
        """
        params patience : early stop only if epoches of no improvement >= patience.
        params min_delta: an absolute change of less than min_delta, will count as no improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = float("inf")
        self.cnt = 0
        self.flag = False

    def __call__(self, loss):
        if (self.min_loss - loss) < self.min_delta:
            self.cnt += 1
        else:
            self.min_loss = loss
            self.cnt = 0
        if self.cnt >= self.patience:
            self.flag = True


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(12544, 10)

        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dense(10, activation='softmax'))

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


def train(model, train_dataset, test_dataset, epochs=200, batch_size=32, learning_rate=0.01,
          device="cpu", loss_func=None, optimizer=None, early_stopping=None, num_workers=0):
    print("Starting training...")
    # Prepare
    if not loss_func:
        loss_func = nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not early_stopping:
        early_stopping = EarlyStopping(patience=30, min_delta=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    log = []
    model.to(device)
    model.train()

    # Data
    train_data = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # test_data = DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Training
    for epoch in range(epochs):
        print("="*10, "Epoch", epoch+1, "="*10)
        # Some metrics
        total_loss = 0
        train_correct_cnt = 0
        train_total_cnt = 0
        # Train each batch
        for img, label in tqdm.tqdm(train_data):
            img = img.to(device)
            label = label.to(device)
            label = torch.flatten(label)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
            pred = torch.argmax(output, dim=1)
            train_correct_cnt += torch.sum(pred == label)
            train_total_cnt += img.shape[0]
        scheduler.step(total_loss)
        print("Total loss:", total_loss)
        train_acc = float(train_correct_cnt/train_total_cnt)
        print("Training accuracy:", train_acc)

        # Testing
        test_acc = test(model, test_dataset, device=device, batch_size=batch_size, num_workers=num_workers)
        print("Testing accuracy:", test_acc)
        model.train()

        # Log
        log.append([epoch+1, total_loss, train_acc, test_acc])

        # Early stopping
        early_stopping(total_loss)
        if early_stopping.flag:
            print(f"Early stop at epoch {epoch}.")
            break

    # Save final model & log
    print("Saving final model...")
    t = datetime.now()
    torch.save(model.state_dict(), f"checkpoints/cnn-{t}.pt")
    with open(f"checkpoints/cnn-{t}.csv", "w+") as f:
        f.write("epoch,total_loss,train_acc,test_acc\n")
        for data in log:
            data_s = [str(num) for num in data]
            f.write(",".join(data_s)+"\n")


def test(model, dataset, device="cpu", batch_size=32, num_workers=0):
    test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    test_correct_cnt = 0
    test_total_cnt = 0
    with torch.no_grad():
        for img, label in test_data:
            img = img.to(device)
            label = label.to(device)
            label = torch.flatten(label)
            output = model(img)
            pred = torch.argmax(output, dim=1)
            test_correct_cnt += torch.sum(pred == label)
            test_total_cnt += img.shape[0]
    test_acc = float(test_correct_cnt/test_total_cnt)
    return test_acc


if __name__ == "__main__":
    # Dataset
    train_dataset = MyDataset(DIGITS_MAT_PATH, "mat", True, 0)
    test_dataset = MyDataset(DIGITS_MAT_PATH, "mat", False, 0)
    # Parameter
    epochs = 500
    batch_size = 32
    device = get_device()
    learning_rate = 0.01
    num_workers = 0
    # Model
    model = CNN()
    summary(model, (1, 28, 28))
    # Training
    train(model, train_dataset, test_dataset, epochs=epochs, batch_size=batch_size,
          learning_rate=learning_rate, device=device, num_workers=num_workers)
    