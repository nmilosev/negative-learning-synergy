import torch
from torchvision import datasets, transforms

kwargs = {'num_workers': 4, 'pin_memory': True} \
         if torch.cuda.is_available() else {}

f_v, f_h, f_d, f_t = torch.ones((28, 28)), torch.ones((28, 28)), \
                     torch.ones((28, 28)), torch.ones((28, 28))
f_v[:, :14] = 0  # vertical filter
f_h[:14, :] = 0  # horizontal filter
f_d[:14, 15:], f_d[15:, :14] = 0, 0  # diagonal filter
f_t[6:15, 6:15], f_t[18:27, 11:20], f_t[8:17, 17:26] = 0, 0, 0  # tcut


def mnist_loader(train=False):
    return torch.utils.data.DataLoader(
        datasets.MNIST('../data', download=True, train=train,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64 if train else 1, shuffle=train, **kwargs)


def emnist_loader(split='balanced'):
    def _loader(train=False):
        return torch.utils.data.DataLoader(
           datasets.EMNIST('../data', split=split, download=True, train=train,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])), batch_size=64 if train else 1,
           shuffle=train, **kwargs)
    return _loader

transform_cifar = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def cifar10_loader(train=False, batch_size=4):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', download=True, train=train,
                          transform=transform_cifar),
                          batch_size=batch_size,
                          shuffle=train, **kwargs)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def cifar10_loader_resnet(train=False, batch_size=1536):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', download=True, train=train,
                          transform=transform_train if train else transform_test),
                          batch_size=batch_size,
                          shuffle=train, **kwargs)

cifar10_classes = ['plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
