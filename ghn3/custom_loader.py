
"""
Prepares training and evaluation torch DataLoaders. Adds options to choose number of classes/train samples per class
Supports ImageNet and torchvision datasets, such as CIFAR-10 and CIFAR-100.

"""


import os
import torch
from torchvision.datasets import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ppuda.vision.transforms import transforms_cifar, transforms_imagenet
from ppuda.vision.imagenet import ImageNetDataset


def image_loader(dataset='imagenet', data_dir='./data/', test=True, im_size=32,
                 batch_size=64, test_batch_size=64, num_workers=0,
                 cutout=False, cutout_length=16, noise=False,
                 seed=1111, load_train_anyway=False, n_shots=None, ddp=False, transforms_train_val=None, verbose=True, n_classes=None):
    """
    Loads train/test DataLoaders with support for few-shot and class-limited scenarios.
    """
    train_data = None

    if dataset.lower() == 'imagenet':
        if transforms_train_val is None:
            transforms_train_val = transforms_imagenet(noise=noise, cifar_style=False, im_size=im_size)
        train_transform, valid_transform = transforms_train_val

        imagenet_dir = os.path.join(data_dir, 'imagenet')

        if not test or load_train_anyway:
            train_data = ImageNetDataset(imagenet_dir, 'train', transform=train_transform, has_validation=not test)

        valid_data = ImageNetDataset(imagenet_dir, 'val', transform=valid_transform, has_validation=not test)

        shuffle_val = True
        n_classes = len(valid_data.classes) if n_classes is None else n_classes
        generator = torch.Generator()
        generator.manual_seed(seed)

    else:
        dataset = dataset.upper()
        if transforms_train_val is None:
            transforms_train_val = transforms_cifar(cutout=cutout, cutout_length=cutout_length, noise=noise, sz=im_size)
        train_transform, valid_transform = transforms_train_val

        if test:
            valid_data = eval(f'{dataset}(data_dir, train=False, download=True, transform=valid_transform)')
            if load_train_anyway:
                train_data = eval(f'{dataset}(data_dir, train=True, download=True, transform=train_transform)')
                if n_shots is not None:
                    train_data = to_few_shot(train_data, n_shots=n_shots)
        else:
            if n_shots is not None and verbose:
                print('few shot regime is only supported for evaluation on the test data')
            train_data = eval(f'{dataset}(data_dir, train=True, download=True, transform=train_transform)')
            valid_data = eval(f'{dataset}(data_dir, train=True, download=True, transform=valid_transform)')

            n_all = len(train_data.targets)
            n_val = n_all // 10
            idx_train, idx_val = torch.split(torch.arange(n_all), [n_all - n_val, n_val])

            train_data.data = train_data.data[idx_train]
            train_data.targets = [train_data.targets[i] for i in idx_train]

            valid_data.data = valid_data.data[idx_val]
            valid_data.targets = [valid_data.targets[i] for i in idx_val]

            if n_shots is not None:
                train_data = to_few_shot(train_data, n_shots=n_shots)

        # === Apply n_classes filtering ===
        if n_classes is not None:
            def filter_classes(data):
                targets = data.targets if hasattr(data, 'targets') else data.labels
                targets = torch.tensor(targets)
                keep_mask = targets < n_classes
                if isinstance(data.data, list):
                    data.data = [d for i, d in enumerate(data.data) if keep_mask[i]]
                else:
                    data.data = data.data[keep_mask]
                filtered_targets = targets[keep_mask].tolist()
                if hasattr(data, 'targets'):
                    data.targets = filtered_targets
                else:
                    data.labels = filtered_targets
                return data

            if train_data is not None:
                train_data = filter_classes(train_data)
            valid_data = filter_classes(valid_data)

        if train_data is not None:
            train_data.checksum = train_data.data.mean()
            train_data.num_examples = len(train_data.targets)

        shuffle_val = False
        if n_classes is None:
            n_classes = len(torch.unique(torch.tensor(valid_data.targets)))
        generator = None

        valid_data.checksum = valid_data.data.mean()
        valid_data.num_examples = len(valid_data.targets)

    if verbose:
        print('loaded {}: {} classes, {} train samples (checksum={}), '
              '{} {} samples (checksum={:.3f})'.format(dataset,
                                                       n_classes,
                                                       train_data.num_examples if train_data else 'none',
                                                       ('%.3f' % train_data.checksum) if train_data else 'none',
                                                       valid_data.num_examples,
                                                       'test' if test else 'val',
                                                       valid_data.checksum))

    if train_data is None:
        train_loader = None
    else:
        sampler = DistributedSampler(train_data) if ddp else None
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=sampler is None,
                                  sampler=sampler,
                                  pin_memory=True,
                                  num_workers=num_workers)

    valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=shuffle_val,
                              pin_memory=True, num_workers=num_workers, generator=generator)

    return train_loader, valid_loader, n_classes

def to_few_shot(dataset, n_shots=100):
    """
    Transforms torchvision dataset to a few-shot dataset.
    :param dataset: torchvision dataset
    :param n_shots: number of samples per class
    :return: few-shot torchvision dataset
    """
    try:
        targets = dataset.targets  # targets or labels depending on the dataset
        is_targets = True
    except:
        targets = dataset.labels
        is_targets = False

    assert min(targets) == 0, 'labels should start from 0, not from {}'.format(min(targets))

    # Find n_shots samples for each class
    labels_dict = {}
    for i, lbl in enumerate(targets):
        lbl = lbl.item() if isinstance(lbl, torch.Tensor) else lbl
        if lbl not in labels_dict:
            labels_dict[lbl] = []
        if len(labels_dict[lbl]) < n_shots:
            labels_dict[lbl].append(i)

    idx = sorted(torch.cat([torch.tensor(v) for k, v in labels_dict.items()]))  # sort according to the original order in the full dataset

    dataset.data = [dataset.data[i] for i in idx] if isinstance(dataset.data, list) else dataset.data[idx]
    targets = [targets[i] for i in idx]
    if is_targets:
        dataset.targets = targets
    else:
        dataset.labels = targets

    return dataset
