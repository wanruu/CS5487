import os
import tqdm
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from torchsummary import summary  # pip install torch-summary
from torch.utils.data import DataLoader

from utils import get_device
from data import MyDataset
from config import DIGITS_MAT_PATH
from cnn_models import EarlyStopping
from cnn_models import CNN_1 as CNN


def train(model, train_dataset, test_dataset, epochs=200, batch_size=32, learning_rate=0.01,
          device="cpu", loss_func=None, optimizer=None, early_stopping=None, num_workers=0, log=[]):
    print("Starting training...")
    # Prepare
    if not loss_func:
        loss_func = nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not early_stopping:
        early_stopping = EarlyStopping(patience=30, min_delta=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
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
    epochs = 50
    batch_size = 32
    device = get_device()
    learning_rate = 0.01
    num_workers = 0

    # Model
    model = CNN()
    # summary(model, (1, 28, 28))

    # Training
    log = []
    train(model, train_dataset, test_dataset, epochs=epochs, batch_size=batch_size,
          learning_rate=learning_rate, device=device, num_workers=num_workers, log=log)

    # Save final model & log
    print("Saving final model...")
    path = f"checkpoints/{datetime.now()}"
    os.mkdir(path)
    torch.save(model.state_dict(), f"{path}/model.pt")
    with open(f"{path}/result.csv", "w+") as f:
        f.write("epoch,total_loss,train_acc,test_acc\n")
        for data in log:
            data_s = [str(num) for num in data]
            f.write(",".join(data_s)+"\n")
    with open(f"{path}/model-info.txt", "w+") as f:
        model_stats = summary(model, (1, 28, 28))
        f.write(str(model_stats)+"\n")
        f.write(str(model))
    