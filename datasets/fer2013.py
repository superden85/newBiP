import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class FER2013:
    """
    FER2013 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.tr_train = [transforms.ToTensor()]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            mean = [0.5]
            std = [0.5]
            self.tr_train.append(transforms.Normalize(mean, std))
            self.tr_test.append(transforms.Normalize(mean, std))

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        help(datasets.FER2013)

    def data_loaders(self, **kwargs):
        trainset = datasets.FER2013(
            root=os.path.join(self.args.data_dir, "FER2013"),
            split='train',
            #download=True,
            transform=self.tr_train,
        )

        testset = datasets.FER2013(
            root=os.path.join(self.args.data_dir, "FER2013"),
            split='test',
            #download=True,
            transform=self.tr_test,
        )

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
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

        return train_loader, test_loader
