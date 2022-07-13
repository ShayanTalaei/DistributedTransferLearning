import torch
import torchvision.datasets as datasets
import torchvision

class MNISTDataset:
    def __init__(self, train) -> None:
        transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

        self.global_dataset=datasets.MNIST(root="./data" ,train=train,
        download=True, transform=transform)

    def get_dataset(self, id):
        if id==-1:
            return self.global_dataset
        all_labels=self.global_dataset.targets
        wanted_labels=[2*id, 2*id+1]

        dataset=torch.utils.data.Subset(self.global_dataset, indices)
        return dataset

a=MnistDataset(True)
a.get_dataset(0)





