import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CustomDataset(Dataset):
    def __init__(self, data_point, label, transform=None):
        self.data_point = data_point
        self.label = label
        self.transform = transform

    def __len__(self):
        return 1  # Only one data point

    def __getitem__(self, idx):
        sample = [self.data_point, self.label]
        if self.transform:
            sample['data'] = self.transform(sample['data'])
        return sample

class MINI:

    def __init__(self, args, normalize=False):
        self.args = args

    def data_loaders(self, **kwargs):
        # Define your fixed vectors for train, val, and test
        train_data_point = torch.tensor([1.0, 2.0])  # Modify as needed
        val_data_point = torch.tensor([3.0, 4.0])    # Modify as needed
        test_data_point = torch.tensor([5.0, 6.0])   # Modify as needed

        # Define fixed labels for each data point
        train_label = torch.tensor([0.0])  # Modify as needed
        val_label = torch.tensor([1.0])    # Modify as needed
        test_label = torch.tensor([2.0])   # Modify as needed

        trainset = CustomDataset(train_data_point, train_label)
        valset = CustomDataset(val_data_point, val_label)
        testset = CustomDataset(test_data_point, test_label)

        train_loader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(valset, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs)

        print(f"Training loader: {len(train_loader.dataset)} training points, Test loader: {len(test_loader.dataset)} test points")
        return train_loader, val_loader, test_loader
