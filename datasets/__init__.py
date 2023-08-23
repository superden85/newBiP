from datasets.cifar import CIFAR10, CIFAR100
from datasets.imagenet import imagenet_h5py as ImageNetH5
from datasets.imagenet import imagenet as ImageNetOrigin
from datasets.imagenet import imagenet_lmbd as ImageNet
from datasets.tiny_imagenet import TinyImageNet
from datasets.mnist import MNIST
from datasets.test import MINI

__all__ = ["CIFAR10", "CIFAR100", "ImageNet", "ImageNetOrigin", "TinyImageNet", "ImageNetH5"]