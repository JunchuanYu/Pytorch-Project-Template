import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
from torch.utils.data.sampler import WeightedRandomSampler
import inspect
import importlib
import pickle as pkl

class DInterface(pl.LightningDataModule):
    
    def __init__(self,trainset, valset, testset=None,**kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.num_workers = 0
        self.batch_size = self.kwargs['batch_size']
        self.trainset=trainset
        self.valset=valset
        self.testset=testset

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
