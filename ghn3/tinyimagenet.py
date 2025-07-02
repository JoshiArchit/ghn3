import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class TinyImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        self.split = split
        self.data = []
        self.targets = []
        self.class_to_idx = {}
        self.idx_to_class = []

        if split == 'train':
            train_dir = os.path.join(root, 'train')
            self._load_train_data(train_dir)
        else:
            val_dir = os.path.join(root, 'val')
            self._load_val_data(val_dir)

    def _load_train_data(self, train_dir):
        classes = sorted(os.listdir(train_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = classes

        for cls in classes:
            img_dir = os.path.join(train_dir, cls, 'images')
            for img_file in os.listdir(img_dir):
                self.data.append(os.path.join(img_dir, img_file))
                self.targets.append(self.class_to_idx[cls])

    def _load_val_data(self, val_dir):
        val_img_dir = os.path.join(val_dir, 'images')
        val_ann_file = os.path.join(val_dir, 'val_annotations.txt')

        with open(val_ann_file, 'r') as f:
            lines = f.readlines()

        cls_names = sorted(list(set([line.strip().split('\t')[1] for line in lines])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(cls_names)}
        self.idx_to_class = cls_names

        for line in lines:
            img_file, cls = line.strip().split('\t')[:2]
            self.data.append(os.path.join(val_img_dir, img_file))
            self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]

    @property
    def num_examples(self):
        return len(self)

    @property
    def checksum(self):
        import hashlib
        m = hashlib.sha256()
        for path in self.data:
            m.update(path.encode())
        return int(m.hexdigest(), 16)