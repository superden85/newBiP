import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class Flowers102:
    """
    Flowers-102 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.tr_train = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]

        self.tr_test = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.Flowers102(
            root=os.path.join(self.args.data_dir, "Flowers102"),
            split='train',
            transform=self.tr_train,
            download=True,
        )

        valset = datasets.Flowers102(
            root=os.path.join(self.args.data_dir, "Flowers102"),
            split='val',
            transform=self.tr_test,  # Use test transform for validation set
            download=True,
        )

        testset = datasets.Flowers102(
            root=os.path.join(self.args.data_dir, "Flowers102"),
            split='test',
            transform=self.tr_test,
            download=True,
        )

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs
        )

        val_loader = DataLoader(
            valset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            **kwargs
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
