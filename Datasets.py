import torch
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib import pyplot as plt


class MNIST_bool_label(Dataset):
    def __init__(self, inner_dataset, id):
        super(MNIST_bool_label, self).__init__()
        self.inner_dataset = inner_dataset
        self.id = id

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, idx):
        x, y = self.inner_dataset.__getitem__(idx)
        return x, y - self.id * 2


class MNISTDataset:
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
        dataset = MNIST_bool_label(torch.utils.data.Subset(self.global_dataset, indices), id)
        return dataset
