import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class Caltech101:
    """
    Caltech-101 dataset.
    """

    def __init__(self, args, target_type='category', transform=None, target_transform=None, download=True, normalize=False):
        self.args = args

        # Normalize layer (you may need to adjust mean and std)
        self.norm_layer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.transform = transform
        self.target_transform = target_transform

        if normalize:
            self.transform = transforms.Compose([self.transform, self.norm_layer])

        self.dataset = datasets.Caltech101(
            root=os.path.join(self.args.data_dir, "Caltech101"),
            target_type=target_type,
            transform=self.transform,
            target_transform=self.target_transform,
            download=download,
        )

    def data_loaders(self, shuffle=True, **kwargs):
        # Split the dataset into train, validation, and test sets
        num_samples = len(self.dataset)
        num_train = int(0.7 * num_samples)
        num_val = int(0.15 * num_samples)
        num_test = num_samples - num_train - num_val

        train_set, val_set, test_set = torch.utils.data.random_split(
            self.dataset, [num_train, num_val, num_test]
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            **kwargs
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            **kwargs
        )

        test_loader = DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            **kwargs
        )

        print(
            f"Training loader: {len(train_loader.dataset)} images, Validation loader: {len(val_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )

        return train_loader, val_loader, test_loader
