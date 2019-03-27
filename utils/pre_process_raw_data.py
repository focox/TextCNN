import os
import numpy as np


def read_raw_data(filepath, sep='$'):
    if not os.path.exists(filepath):
        print(filepath, 'Not Found.')
    with open(filepath) as f:
        data = f.read()
    data = data.split('\n')
    data = [i.split(sep) for i in data if i]
    return np.array(data)


def raw2label(data):
    label = data[:, -1]
    label = label.tolist()
    category = list(set(label))
    label = [category.index(i) for i in label]
    label = np.array(label)
    return np.column_stack((data, label)), category


# def


if __name__ == '__main__':
    filepath = 'external_data.csv'
    data = read_raw_data(filepath)
    print(data)
    data, label = raw2label(data)
    print(data)
