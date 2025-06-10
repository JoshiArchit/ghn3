"""
Filename : custom_image_loader.py
Author : Archit
Date Created : 6/8/2025
Description : Custom image loader allowing data splits and more custom options for loading data.
Adapted from ppuda.vision.loader.image_loader.py.
Language : python3
"""

import os
import torch
from torchvision.datasets import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ppuda.vision.transforms import transforms_cifar, transforms_imagenet
from ppuda.vision.imagenet import ImageNetDataset


def load_images(dataset='imagenet', data_dir='./data/', test=True, im_size=32,
                batch_size=64, test_batch_size=64, num_workers=0,
                cutout=False, cutout_length=16, noise=False,
                seed=1111, load_train_anyway=False, n_shots=None, ddp=False, transforms_train_val=None,
                verbose=True, train_split=90, train_split_classes='all'):
    """
    Custom data loader with support for class-based and split-based filtering.
    """
    train_data = None
    train_classes_split = int(train_split_classes) if train_split_classes != 'all' else None
    train_ratio = float(train_split) / 100.0

    if dataset.lower() == 'imagenet':
        if transforms_train_val is None:
            transforms_train_val = transforms_imagenet(noise=noise, cifar_style=False, im_size=im_size)
        train_transform, valid_transform = transforms_train_val

        imagenet_dir = os.path.join(data_dir, 'imagenet')

        train_data = ImageNetDataset(imagenet_dir, 'train', transform=train_transform, has_validation=not test)
        valid_data = ImageNetDataset(imagenet_dir, 'val', transform=valid_transform, has_validation=not test)

        all_data = train_data.samples + valid_data.samples
        all_targets = train_data.targets + valid_data.targets

        # Class filtering
        if train_classes_split is not None:
            class_indices = [i for i, t in enumerate(all_targets) if t < train_classes_split]
            all_data = [all_data[i] for i in class_indices]
            all_targets = [all_targets[i] for i in class_indices]

        # Train/val split
        total = len(all_data)
        num_train = int(train_ratio * total)
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total, generator=g)

        idx_train = indices[:num_train]
        idx_val = indices[num_train:]

        train_data.samples = [all_data[i] for i in idx_train]
        train_data.targets = [all_targets[i] for i in idx_train]

        valid_data.samples = [all_data[i] for i in idx_val]
        valid_data.targets = [all_targets[i] for i in idx_val]

        n_classes = train_classes_split if train_classes_split is not None else len(set(all_targets))
        shuffle_val = True
        generator = torch.Generator().manual_seed(seed)

    else:
        dataset = dataset.upper()
        if transforms_train_val is None:
            transforms_train_val = transforms_cifar(cutout=cutout, cutout_length=cutout_length, noise=noise, sz=im_size)
        train_transform, valid_transform = transforms_train_val

        if test:
            valid_data = eval('{}(data_dir, train=False, download=True, transform=valid_transform)'.format(dataset))
            if load_train_anyway:
                train_data = eval('{}(data_dir, train=True, download=True, transform=train_transform)'.format(dataset))
                if n_shots is not None:
                    train_data = to_few_shot(train_data, n_shots=n_shots)
        else:
            if n_shots is not None and verbose:
                print('few shot regime is only supported for evaluation on the test data')

            train_data = eval(f'{dataset}(data_dir, train=True, download=True, transform=train_transform)')
            valid_data = eval(f'{dataset}(data_dir, train=True, download=True, transform=valid_transform)')

            targets = torch.tensor(train_data.targets)
            data = train_data.data

            # Class filtering
            if train_classes_split is not None:
                class_mask = torch.isin(targets, torch.arange(train_classes_split))
                indices = torch.nonzero(class_mask).squeeze()

                data = data[indices]
                targets = targets[indices]
                valid_data.data = valid_data.data[indices]
                valid_data.targets = [valid_data.targets[i] for i in indices]

            # Split
            total = len(targets)
            num_train = int(train_ratio * total)
            g = torch.Generator().manual_seed(seed)
            indices = torch.randperm(total, generator=g)

            idx_train = indices[:num_train]
            idx_val = indices[num_train:]

            train_data.data = data[idx_train]
            train_data.targets = [targets[i].item() for i in idx_train]

            valid_data.data = data[idx_val]
            valid_data.targets = [targets[i].item() for i in idx_val]

            if n_shots is not None:
                train_data = to_few_shot(train_data, n_shots=n_shots)

            train_data.checksum = train_data.data.mean()
            train_data.num_examples = len(train_data.targets)

            shuffle_val = False
            n_classes = train_classes_split if train_classes_split is not None else len(set(targets.tolist()))
            generator = None
            valid_data.checksum = valid_data.data.mean()
            valid_data.num_examples = len(valid_data.targets)

    if verbose:
        print('loaded {}: {} classes, {} train samples (checksum={}), '
              '{} {} samples (checksum={:.3f})'.format(dataset,
                                                       n_classes,
                                                       len(train_data.targets) if train_data else 'none',
                                                       ('%.3f' % train_data.data.mean()) if hasattr(train_data, 'data') else 'none',
                                                       len(valid_data.targets),
                                                       'test' if test else 'val',
                                                       valid_data.data.mean() if hasattr(valid_data, 'data') else 0.0))
        # Print classes chosen for training
        if train_classes_split is not None:
            print('classes chosen for training: {}'.format(
                ', '.join([str(i) for i in range(train_classes_split)])))

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
        targets = dataset.targets
        is_targets = True
    except:
        targets = dataset.labels
        is_targets = False

    assert min(targets) == 0, 'labels should start from 0, not from {}'.format(min(targets))

    labels_dict = {}
    for i, lbl in enumerate(targets):
        lbl = lbl.item() if isinstance(lbl, torch.Tensor) else lbl
        if lbl not in labels_dict:
            labels_dict[lbl] = []
        if len(labels_dict[lbl]) < n_shots:
            labels_dict[lbl].append(i)

    idx = sorted(torch.cat([torch.tensor(v) for v in labels_dict.values()]))

    dataset.data = [dataset.data[i] for i in idx] if isinstance(dataset.data, list) else dataset.data[idx]
    targets = [targets[i] for i in idx]
    if is_targets:
        dataset.targets = targets
    else:
        dataset.labels = targets

    return dataset
