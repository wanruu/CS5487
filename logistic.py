import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

from config import DIGITS_MAT_PATH
from data import MyDataset
from utils import accuracy

# import cv2

train_dataset = MyDataset(DIGITS_MAT_PATH, "mat", True, 0)
test_dataset = MyDataset(DIGITS_MAT_PATH, "mat", False, 0)
X_train = train_dataset.imgs.T
y_train = train_dataset.labels.T
X_test = test_dataset.imgs.T
y_test = test_dataset.labels.T

# train_data = DataLoader(train_dataset, batch_size=1,
#                         shuffle=False, num_workers=0)


# for img, label in train_data:
#     # plt.figure(figsize=(20,4))
#     # plt.imshow(np.reshape(img, (28,28)), cmap=plt.cm.gray)
#     print(img.reshape(28, 28).shape)
#     cv2.imshow("", np.array(img.reshape(28, 28)))
#     cv2.waitKey(0)
#     break

clf = LogisticRegression(fit_intercept=True,
                         multi_class='auto',
                         penalty='l2',  # ridge regression
                         solver='saga',
                         max_iter=10000,
                         C=50)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print(accuracy(y_test,y_predict))