from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes


class NTDetectionDataset(Dataset):
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    def __init__(self, root_dir, is_train, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        if is_train:
            self.dataset = load_dataset('imagefolder', data_dir=str(self.root_dir), split="train")
        else:
            self.dataset = load_dataset('imagefolder', data_dir=str(self.root_dir), split="test")
        self.dataset.set_format("torch")

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]

        height, width = image.shape[1:]
        boxes = BoundingBoxes(self.dataset[idx]["objects"]["bbox"], format="XYWH", canvas_size=(height, width))
        labels = self.dataset[idx]["objects"]["class"]
        category = self.dataset[idx]["objects"]["categories"]
        image_id = idx
        area = boxes[:, 2] * boxes[:, 3]  #(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros_like(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": is_crowd,
            "class": category
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.dataset)
