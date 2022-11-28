import numpy as np

def accuracy(predict, actual) -> float:
    predict = np.array(predict).flatten()
    actual = np.array(actual).flatten()
    result = predict == actual
    return np.sum(result) / predict.shape[0]  # [0,1]
