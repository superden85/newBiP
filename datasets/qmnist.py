import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class QMNIST:
    """
    QMNIST dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std for QMNIST

        self.tr_train = [transforms.ToTensor()]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.QMNIST(
            root=os.path.join(self.args.data_dir, "QMNIST"),  # Change the dataset folder
            train=True,
            download=True,
            transform=self.tr_train,
        )

        valset = datasets.QMNIST(
            root=os.path.join(self.args.data_dir, "QMNIST"),  # Change the dataset folder
            train=True,
            download=True,
            transform=self.tr_train,
        )

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )

        val_loader = DataLoader(
            valset,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )

        testset = datasets.QMNIST(
            root=os.path.join(self.args.data_dir, "QMNIST"),  # Change the dataset folder
            train=False,
            download=True,
            transform=self.tr_test,
        )

        test_loader = DataLoader(
            testset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        print(
            f"Training loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader
