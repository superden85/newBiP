import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class Caltech101:
    """
    Caltech-101 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        # Normalize layer (you may need to adjust mean and std)
        self.norm_layer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.tr_train = [transforms.ToTensor()]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.Caltech101(
            root=os.path.join(self.args.data_dir, "Caltech101"),
            train=True,  # Use 'train' to specify the train split
            download=True,
            transform=self.tr_train,
        )

        valset = datasets.Caltech101(
            root=os.path.join(self.args.data_dir, "Caltech101"),
            train=False,  # Use 'train=False' to specify the validation/test split
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

        print(
            f"Training loader: {len(train_loader.dataset)} images, Test loader: {len(val_loader.dataset)} images"
        )
        return train_loader, val_loader
