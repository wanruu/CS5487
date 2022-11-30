import torch
import scipy.io
from torch.utils.data import Dataset

# For pytorch.
class MyDataset(Dataset):
    def __init__(self, filepath, filetype="mat", train=True, cross_idx=0):
        # Read data from mat file.
        # Extract all data.
        if filetype == "mat":
            mat_data = scipy.io.loadmat(filepath)
            labels = mat_data["digits_labels"]  # (1,4000)
            imgs = mat_data["digits_vec"]  # (784,4000)
            trainset = mat_data["trainset"][cross_idx] - 1  # (2,2000) -> (2000,)
            testset = mat_data["testset"][cross_idx] - 1  # (2,2000) -> (2000,)
        if train:
            self.imgs = imgs[:,trainset]  # (784,2000)
            self.labels = labels[:,trainset]  # (1,2000)
        else:
            self.imgs = imgs[:,testset]  # (784,2000)
            self.labels = labels[:,testset]  # (1,2000)

    def __getitem__(self, index):
        img = torch.Tensor(self.imgs[:,index].reshape(-1, 28, 28))
        label = torch.LongTensor(self.labels[:,index])
        return img, label

    def __len__(self):
        return self.labels.shape[1]


# Sample
if __name__ == "__main__":
    from config import DIGITS_MAT_PATH
    train_cross_1 = MyDataset(DIGITS_MAT_PATH, "mat", True, 0)
    test_cross_1 = MyDataset(DIGITS_MAT_PATH, "mat", False, 0)
    train_cross_2 = MyDataset(DIGITS_MAT_PATH, "mat", True, 1)
    test_cross_2 = MyDataset(DIGITS_MAT_PATH, "mat", False, 1)
    print(train_cross_1[0])