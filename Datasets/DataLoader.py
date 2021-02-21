from typing import Union, List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import torch


class DataLoader(LightningDataModule):

    def __init__(self, batch_size, dataset):
        super(DataLoader, self).__init__()
        self.batch_size = batch_size
        self.dataset = dataset

    def setup(self, stage: Optional[str] = None):
        split = 8 * len(self.dataset) // 10
        self.train, self.val = random_split(self.dataset, [split, len(self.dataset) - split],
                                            torch.random.manual_seed(10))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val, batch_size=self.batch_size)