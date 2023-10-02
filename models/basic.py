import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GetSubnetUnstructured


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# lin_i: i layer linear feedforard network.
def lin_1(input_dim=3072, num_classes=10):
    model = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, num_classes))
    return model


def lin_2(input_dim=3072, hidden_dim=100, num_classes=10):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.Linear(hidden_dim, num_classes),
    )
    return model


def lin_3(input_dim=3072, hidden_dim=100, num_classes=10):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Linear(hidden_dim, num_classes),
    )
    return model


def lin_4(input_dim=3072, hidden_dim=100, num_classes=10):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Linear(hidden_dim, num_classes),
    )
    return model



class MnistModel(nn.Module):

    def __init__(self, conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
        super(MnistModel, self).__init__()
        self.conv1 = conv_layer(1, 16, 4, stride=2, padding=1)
        self.conv2 = conv_layer(16, 32, 4, stride=2, padding=1)
        self.fc1 = linear_layer(32 * 7 * 7, 100)
        self.fc2 = linear_layer(100, 10)
        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.unstructured_pruning = kwargs['unstructured'] if 'unstructured' in kwargs else False

    def _forward_impl(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel
        else:
            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            vec.w = attr
                            pointer += attr.numel()
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)

class MnistModelMini(nn.Module):

    def __init__(self, conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
        super(MnistModelMini, self).__init__()
        self.fc1 = linear_layer(28 * 28, 10)
        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.unstructured_pruning = kwargs['unstructured'] if 'unstructured' in kwargs else False

    def _forward_impl(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel
        else:
            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            vec.w = attr
                            pointer += attr.numel()
        
        x = Flatten()(x)
        x = self.fc1(x)
        
        return x
    
    def forward(self, x):
        return self._forward_impl(x)

class FashionMnistModel(nn.Module):

    def __init__(self, conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
        super(FashionMnistModel, self).__init__()
        self.conv1 = conv_layer(1, 48, 4, stride=2, padding=1)  # Increase output channels to 48
        self.conv2 = conv_layer(48, 96, 4, stride=2, padding=1)  # Increase output channels to 96
        self.fc1 = linear_layer(96 * 7 * 7, 300)  # Increase the number of neurons in fc1 to 300
        self.fc2 = linear_layer(300, 10)  # Keep fc2 as it is

        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.unstructured_pruning = kwargs['unstructured'] if 'unstructured' in kwargs else False

    def _forward_impl(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel
        else:
            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            vec.w = attr
                            pointer += attr.numel()
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)

class EmnistModel(nn.Module):

    def __init__(self, conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
        super(EmnistModel, self).__init__()
        self.conv1 = conv_layer(1, 48, 4, stride=2, padding=1)  # Increase output channels to 48
        self.conv2 = conv_layer(48, 96, 4, stride=2, padding=1)  # Increase output channels to 96
        self.fc1 = linear_layer(96 * 7 * 7, 300)  # Increase the number of neurons in fc1 to 300
        self.fc2 = linear_layer(300, 47)  # Keep fc2 as it is

        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.unstructured_pruning = kwargs['unstructured'] if 'unstructured' in kwargs else False

    def _forward_impl(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel
        else:
            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            vec.w = attr
                            pointer += attr.numel()
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)

class EuroSATModel(nn.Module):
    def __init__(self, conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
        super(EuroSATModel, self).__init__()
        self.conv1 = conv_layer(3, 64, 4, stride=2, padding=1)  # Input channels: 3 (RGB), Output channels: 64
        self.conv2 = conv_layer(64, 128, 4, stride=2, padding=1)  # Increase output channels to 128
        self.fc1 = linear_layer(128 * 16 * 16, 300)  # Increase the number of neurons in fc1 to 300
        self.fc2 = linear_layer(300, 10)  # Adjust the output dimension to match the number of classes in EuroSAT

        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.unstructured_pruning = kwargs['unstructured'] if 'unstructured' in kwargs else False

    def _forward_impl(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel
        else:
            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            vec.w = attr
                            pointer += attr.numel()
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)  

class FER2013Model(nn.Module):
    def __init__(self, conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
        super(FER2013Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 7)

        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 7
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.unstructured_pruning = kwargs['unstructured'] if 'unstructured' in kwargs else False
    
    def _forward_impl(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel
        else:
            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            vec.w = attr
                            pointer += attr.numel()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)

class FGVC_Aircraft_Model(nn.Module):
    def __init__(self, conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
        super(FGVC_Aircraft_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # Input channels: 3 (RGB), Output channels: 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # Output channels: 128
        self.fc1 = nn.Linear(128 * 14 * 14, 300)  # Increase the number of neurons in fc1 to 300
        self.fc2 = nn.Linear(300, 100)  # Adjust the output dimension to match the number of classes

        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 100
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.unstructured_pruning = kwargs['unstructured'] if 'unstructured' in kwargs else False
    
    def forward(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if hasattr(vec, "weight"):
                    attr = getattr(vec, "weight")
                    if attr is not None:
                        numel = attr.numel()
                        vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                        pointer += numel
        else:
            pointer = 0
            for (name, vec) in self.named_modules():
                if hasattr(vec, "weight"):
                    attr = getattr(vec, "weight")
                    if attr is not None:
                        vec.w = attr
                        pointer += attr.numel()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)

class Caltech101Model(nn.Module):

    def __init__(self, conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
        super(Caltech101Model, self).__init__()
        # Convolutional Layers
        self.conv1 = conv_layer(3, 64, 3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)  # Max-pooling layer
        self.conv2 = conv_layer(64, 128, 3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)  # Max-pooling layer
        self.conv3 = conv_layer(128, 256, 3, stride=1, padding=1)
        self.conv4 = conv_layer(256, 256, 3, stride=1, padding=1)
        self.conv5 = conv_layer(256, 256, 3, stride=1, padding=1)

        # Fully Connected Layers
        self.fc1 = linear_layer(256 * 7 * 7, 1024)  # Reduced FC layer
        self.fc2 = linear_layer(1024, 101)  # Output dimension for Caltech-101

        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 101
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.unstructured_pruning = kwargs['unstructured'] if 'unstructured' in kwargs else False


    def _forward_impl(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel
        else:
            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            vec.w = attr
                            pointer += attr.numel()
        
        # Forward pass through convolutional layers
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)

        # Flatten the output
        x = Flatten()(x)

        # Forward pass through fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)

def caltech101_model(conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = Caltech101Model(conv_layer, linear_layer, init_type, **kwargs)
    return model

def mnist_model(conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = MnistModel(conv_layer, linear_layer, init_type, **kwargs)
    return model

def mnist_model_mini(conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = MnistModelMini(conv_layer, linear_layer, init_type, **kwargs)
    return model

def fmnist_model(conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = FashionMnistModel(conv_layer, linear_layer, init_type, **kwargs)
    return model

def emnist_model(conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = EmnistModel(conv_layer, linear_layer, init_type, **kwargs)
    return model

def eurosat_model(conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = EuroSATModel(conv_layer, linear_layer, init_type, **kwargs)
    return model

def fer2013_model(conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = FER2013Model(conv_layer, linear_layer, init_type, **kwargs)
    return model

def fgvca_model(conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = FGVC_Aircraft_Model(conv_layer, linear_layer, init_type, **kwargs)
    return model
    
def mnist_model_large(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = nn.Sequential(
        conv_layer(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(64 * 7 * 7, 512),
        nn.ReLU(),
        linear_layer(512, 512),
        nn.ReLU(),
        linear_layer(512, 10),
    )
    return model


def cifar_model(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = nn.Sequential(
        conv_layer(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(32 * 8 * 8, 100),
        nn.ReLU(),
        linear_layer(100, 10),
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
    return model


def cifar_model_large(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = nn.Sequential(
        conv_layer(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(64 * 8 * 8, 512),
        nn.ReLU(),
        linear_layer(512, 512),
        nn.ReLU(),
        linear_layer(512, 10),
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
    return model


class DenseSequential(nn.Sequential):
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if "Dense" in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]


class Dense(nn.Module):
    def __init__(self, *Ws):
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], "out_features"):
            self.out_features = Ws[0].out_features

    def forward(self, *xs):
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x, W in zip(xs, self.Ws) if W is not None)
        return out


def cifar_model_resnet(conv_layer, linear_layer, init_type, N=5, factor=1, **kwargs):
    def block(in_filters, out_filters, k, downsample):
        if not downsample:
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else:
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(
                conv_layer(
                    in_filters, out_filters, k_first, stride=skip_stride, padding=1
                )
            ),
            nn.ReLU(),
            Dense(
                conv_layer(
                    in_filters, out_filters, k_skip, stride=skip_stride, padding=0
                ),
                None,
                conv_layer(out_filters, out_filters, k, stride=1, padding=1),
            ),
            nn.ReLU(),
        ]

    conv1 = [conv_layer(3, 16, 3, stride=1, padding=1), nn.ReLU()]
    conv2 = block(16, 16 * factor, 3, False)
    for _ in range(N):
        conv2.extend(block(16 * factor, 16 * factor, 3, False))
    conv3 = block(16 * factor, 32 * factor, 3, True)
    for _ in range(N - 1):
        conv3.extend(block(32 * factor, 32 * factor, 3, False))
    conv4 = block(32 * factor, 64 * factor, 3, True)
    for _ in range(N - 1):
        conv4.extend(block(64 * factor, 64 * factor, 3, False))
    layers = (
            conv1
            + conv2
            + conv3
            + conv4
            + [
                Flatten(),
                linear_layer(64 * factor * 8 * 8, 1000),
                nn.ReLU(),
                linear_layer(1000, 10),
            ]
    )
    model = DenseSequential(*layers)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def vgg4_without_maxpool(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = nn.Sequential(
        conv_layer(3, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(64, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(64, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        conv_layer(128, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        linear_layer(128 * 8 * 8, 256),
        nn.ReLU(),
        linear_layer(256, 256),
        nn.ReLU(),
        linear_layer(256, 10),
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
    return model


def cifar_model_resnet(N=5, factor=10):
    def block(in_filters, out_filters, k, downsample):
        if not downsample:
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else:
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(
                nn.Conv2d(
                    in_filters, out_filters, k_first, stride=skip_stride, padding=1
                )
            ),
            nn.ReLU(),
            Dense(
                nn.Conv2d(
                    in_filters, out_filters, k_skip, stride=skip_stride, padding=0
                ),
                None,
                nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1),
            ),
            nn.ReLU(),
        ]

    conv1 = [nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU()]
    conv2 = block(16, 16 * factor, 3, False)
    for _ in range(N):
        conv2.extend(block(16 * factor, 16 * factor, 3, False))
    conv3 = block(16 * factor, 32 * factor, 3, True)
    for _ in range(N - 1):
        conv3.extend(block(32 * factor, 32 * factor, 3, False))
    conv4 = block(32 * factor, 64 * factor, 3, True)
    for _ in range(N - 1):
        conv4.extend(block(64 * factor, 64 * factor, 3, False))
    layers = (
            conv1
            + conv2
            + conv3
            + conv4
            + [
                Flatten(),
                nn.Linear(64 * factor * 8 * 8, 1000),
                nn.ReLU(),
                nn.Linear(1000, 10),
            ]
    )
    model = DenseSequential(*layers)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
    return model
