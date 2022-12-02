import os
import tqdm
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from torchsummary import summary  # pip install torch-summary
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping

from utils import get_device
from data import MyDataset
from config import DIGITS_MAT_PATH
from cnn_models import CNN_1, CNN_2, CNN_3



def train(model, train_dataset, val_dataset, test_dataset, epochs=200, batch_size=32, learning_rate=0.01,
          device="cpu", num_workers=0, log=[]):
    print("Starting training...")
    
    # Prepare
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    min_valid_loss = np.inf
    model.to(device)

    # Data
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    # Training
    for epoch in range(epochs):
        print("="*10, "Epoch", epoch+1, "="*10)

        # ======
        # Train
        # ======
        model.train()
        train_loss = 0
        train_correct_cnt = 0
        train_total_cnt = 0
        for img, label in tqdm.tqdm(train_data):
            img = img.to(device)
            label = label.to(device)
            label = torch.flatten(label)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            train_correct_cnt += torch.sum(pred == label)
            train_total_cnt += img.shape[0]
        scheduler.step(train_loss)
        train_acc = float(train_correct_cnt/train_total_cnt)
        

        # ===========
        # validation
        # ===========
        model.eval()
        valid_loss = 0
        for img, label in val_data:
            img = img.to(device)
            label = label.to(device)
            label = torch.flatten(label)
            output = model(img)
            loss = loss_func(output, label)
            valid_loss += loss.item()
        
        # ===========
        # test
        # ===========
        test_acc = test(model, test_dataset, device=device, batch_size=batch_size, num_workers=num_workers)

        print(f"train_loss: {train_loss}, valid_loss: {valid_loss}")
        print(f"train_acc: {train_acc}, test_acc: {test_acc}")
        log.append([epoch+1, train_loss, valid_loss, train_acc, test_acc])

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t no Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            # torch.save(model.state_dict(), 'saved_model.pt')

        # early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
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


def test_multi(models, dataset, device="cpu", batch_size=32, num_workers=0):
    test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_correct_cnt = 0
    test_total_cnt = 0
    with torch.no_grad():
        for img, label in test_data:
            img = img.to(device)
            label = label.to(device)
            label = torch.flatten(label)
            
            output = 0
            for model in models:
                model.eval()
                output += model(img)
            pred = torch.argmax(output, dim=1)

            test_correct_cnt += torch.sum(pred == label)
            test_total_cnt += img.shape[0]
    test_acc = float(test_correct_cnt/test_total_cnt)
    return test_acc



if __name__ == "__main__":
    # Dataset
    total_train_dataset = MyDataset(DIGITS_MAT_PATH, "mat", True, 1)
    test_dataset = MyDataset(DIGITS_MAT_PATH, "mat", False, 1)

    # k-folder validation 
    k = 5
    split_sizes = [int(len(total_train_dataset)/k) for _ in range(k-1)]
    split_sizes.append(len(total_train_dataset)-sum(split_sizes))
    train_val_datasets = torch.utils.data.random_split(total_train_dataset, lengths=split_sizes)

    # Parameter
    epochs = 200
    batch_size = 32
    device = get_device()
    learning_rate = 0.01
    num_workers = 0

    path = f"cross_valid-trial2/{datetime.now()}"
    os.mkdir(path)

    models = []

    for trial_idx in range(k):
        print("="*20, "Trial", trial_idx, "="*20)

        # dataset
        train_datasets = train_val_datasets[:trial_idx] + train_val_datasets[trial_idx+1:]
        train_dataset = train_datasets[0]
        for idx in range(1, len(train_datasets)):
            train_dataset += train_datasets[idx]
        val_dataset = train_val_datasets[trial_idx]

        # Model
        model = CNN_3(dropout=0.3)
        # Training
        log = []
        train(model, train_dataset, val_dataset, test_dataset, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, device=device, num_workers=num_workers, log=log)
        # Extract result
        acc = log[-1][-1]
        models.append(model)

        # Save final model & log
        print("Saving final model...")
        torch.save(model.state_dict(), f"{path}/{trial_idx}-model.pt")
        with open(f"{path}/{trial_idx}-result.csv", "w+") as f:
            f.write("epoch,train_loss,valid_loss,train_acc,test_acc\n")
            for data in log:
                data_s = [str(num) for num in data]
                f.write(",".join(data_s)+"\n")
        with open(f"{path}/{trial_idx}-model-info.txt", "w+") as f:
            model_stats = summary(model, (1, 28, 28))
            f.write(str(model_stats)+"\n")
            f.write(str(model))
        print(acc)
    

    test_acc = test_multi(models, test_dataset)
    print(test_acc)
    