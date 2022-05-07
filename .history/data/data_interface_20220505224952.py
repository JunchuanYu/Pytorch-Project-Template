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
        # self.dataset = self.kwargs['dataset']
        # self.load_data_module()
        # self.mode=self.kwargs['mode']

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    # def load_data_module(self):
    #     name=self.dataset
    #     # Please always name your model file name as `snake_case.py` and
    #     # class name corresponding `CamelCase`.
    #     camel_name = ''.join([i for i in name.split('_')])
    #     try:
    #         self.data_module = getattr(importlib.import_module(name, package=__package__), camel_name)
    #     except:
    #         raise ValueError(
    #             f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}'
    #             'Please always name your model file name as `snake_case.py` and class name corresponding `CamelCase`')

    # def instancialize(self, **other_args):
    #     """ Instancialize a model using the corresponding parameters
    #         from self.hparams dictionary. You can also input any args
    #         to overwrite the corresponding value in self.kwargs.
    #     """
    #     class_args = inspect.getargspec(self.data_module.__init__).args[1:]
    #     inkeys = self.kwargs.keys()
    #     args1 = {}
    #     for arg in class_args:
    #         if arg in inkeys:
    #             args1[arg] = self.kwargs[arg]
    #     args1.update(other_args)
    #     return self.data_module(**args1)