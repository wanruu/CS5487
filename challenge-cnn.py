import torch
import scipy.io
from torch.utils.data import Dataset
from cnn_models import CNN_1, CNN_2, CNN_3
from cnn_cross import test_multi

class ChallDataset(Dataset):
    def __init__(self, filepath):
        mat_data = scipy.io.loadmat(filepath)
        self.imgs = mat_data["cdigits_vec"]  # (784,150)
        self.labels = mat_data["cdigits_labels"]  # (1,150)
    def __getitem__(self, index):
        img = torch.Tensor(self.imgs[:,index].reshape(-1, 28, 28))
        label = torch.LongTensor(self.labels[:,index])
        return img, label
    def __len__(self):
        return self.labels.shape[1]

test_dataset = ChallDataset("../challenge/cdigits.mat")

trial1 = "cross_valid/2022-12-02 08:21:29.424266/"
trial2 = "cross_valid-trial2/2022-12-02 12:51:21.340284/"


models = []
for k in range(5):
    model = CNN_3()
    state_dict = torch.load(f"{trial2}/{k}-model.pt")
    model.load_state_dict(state_dict)
    models.append(model)

acc = test_multi(models, test_dataset)
print(acc)