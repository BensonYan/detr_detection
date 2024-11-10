from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset.nt_object import NTDetectionDataset


class NTObjectDataModule(L.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, num_workers: int, image_size: int):
        super().__init__()
        root_dir = Path(root_dir)
        assert root_dir.exists(), f"provided path {root_dir} does not exist"

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.dataset_train, self.dataset_test = None, None

    def custom_collate_fn(self,batch):
        images = [item[0] for item in batch]
        bboxes = [item[1]['boxes'] for item in batch]
        labels = [item[1]['labels'] for item in batch]
        category = [item[1]['class'] for item in batch]
        id = [item[1]['image_id'] for item in batch]
        # 对 bbox 进行填充以匹配批次中最长的长度
        max_num_bboxes = max(len(bbox) for bbox in bboxes)
        padded_bboxes = [list(bbox) + [[0, 0, 0, 0]] * (max_num_bboxes - len(bbox)) for bbox in bboxes]
        converted_data = [[t if isinstance(t, list) else t.tolist() for t in sublist] for sublist in padded_bboxes
]

        return torch.stack(images), {
            # 'images': torch.stack(images),
            'idx': torch.tensor(id),
            'boxes': torch.tensor(converted_data),
            'labels': torch.stack(labels),
            "class": torch.tensor(category)
        }
    def setup(self, stage: str):
        # https://pytorch.org/vision/stable/transforms.html
        transform_train = v2.Compose([
            v2.ToImage(),
            # v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.RandomResizedCrop(size=(self.image_size, self.image_size), antialias=True),
            # v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        transform_test = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        if stage == "fit":
            self.dataset_train = NTDetectionDataset(str(self.root_dir), is_train=True, transforms=transform_train)
        self.dataset_test = NTDetectionDataset(str(self.root_dir), is_train=False, transforms=transform_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,collate_fn=self.custom_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,collate_fn=self.custom_collate_fn)
