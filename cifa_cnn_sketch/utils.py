"""
Load data:

load and preprocess data/

sample:
sample the data in an IID way.

"""

from torchvision import datasets, transforms
import numpy as np

def load_mnist(path):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST(path, train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(path, train=False, download=True, transform=trans_mnist)
    return dataset_train, dataset_test

def load_cifar(path_cifar):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.8, saturation=0.05, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset_train = datasets.CIFAR10(path_cifar, train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10(path_cifar, train=False, download=True, transform=transform_test)
    return dataset_train, dataset_test


# randomly shuffle tha training data and dispense training data to differenent users
    # Generate random indices and random signs
    # Args:
    #    dataset: training data
    #    num_users: number of users
    # Return:
    #    data_split: a list of indices for users
def sample_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    data_split = []
    idx = np.random.permutation(len(dataset))
    for i in range(num_users):
        start = i * num_items
        end = start + num_items
        data_split.append(idx[start:end])
    return data_split


if __name__ == '__main__':
    train_data, test_data = load_mnist('./data/mnist')
