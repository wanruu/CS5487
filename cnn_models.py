import torch.nn as nn

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-5):
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


class CNN_1(nn.Module):
    def __init__(self, dropout=0.2) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(5408, 10)

    def forward(self, input):
        out = self.conv1(input)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class CNN_2(nn.Module):
    def __init__(self, dropout=0.5) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1600, 10)    # kernel = 5
        # self.fc = nn.Linear(3136, 10)  # kernel = 3
        # self.fc = nn.Linear(1024, 10)  # kernel = 7

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class CNN_3(nn.Module):
    def __init__(self, dropout=0.5) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 10)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    from torchsummary import summary 
    model = CNN_2()
    summary(model, (1, 28, 28))