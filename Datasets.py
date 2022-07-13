import torch
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib import pyplot as plt
from torchvision.transforms.functional import rotate


class MNISTBoolLabel(Dataset):
    def __init__(self, inner_dataset, id):
        super(MNISTBoolLabel, self).__init__()
        self.inner_dataset = inner_dataset
        self.id = id

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, idx):
        x, y = self.inner_dataset.__getitem__(idx)
        return x, y - self.id * 2


class MNISTSplitDataset:
    def __init__(self, train) -> None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

        self.global_dataset = datasets.MNIST(root="./data", train=train,
                                             download=True, transform=transform)

    def get_dataset(self, id):
        if id == -1:
            return self.global_dataset
        all_labels = self.global_dataset.targets
        wanted_labels = [2 * id, 2 * id + 1]
        indices = None
        for wanted_label in wanted_labels:
            if indices is None:
                indices = (all_labels == wanted_label)
            else:
                indices = torch.logical_or(indices, (all_labels == wanted_label))

        indices = indices.nonzero().reshape(-1)
        dataset = MNISTBoolLabel(torch.utils.data.Subset(self.global_dataset, indices), id)
        return dataset


class MNISTRotateImage(Dataset):
    def __init__(self, inner_dataset, degree):
        super(MNISTRotateImage, self).__init__()
        self.inner_dataset = inner_dataset
        self.degree = degree

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, idx):
        x, y = self.inner_dataset.__getitem__(idx)
        return torchvision.transforms.functional.rotate(x, self.degree), y


class MNISTRotationDataset:
    def __init__(self, train, number_of_dataset) -> None:
        self.count = number_of_dataset
        self.train = train

    def get_dataset(self, id):
        if id == -1:
            transform = torchvision.transforms.Compose([torchvision.transforms.RandomRotation(180),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                        ])
            return datasets.MNIST(root="./data", train=self.train,
                                  download=True, transform=transform)
        else:
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
            return MNISTRotateImage(datasets.MNIST(root="./data", train=self.train,
                                                   download=True, transform=transform), (360 / self.count) * id)
