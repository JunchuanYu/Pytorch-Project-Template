# %% [markdown]
# # PyTorch lightning project template 

# %%
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from model import MInterface
from data import DInterface
from torch.utils.data import DataLoader,random_split
import importlib
from utils import load_model_path_by_args,load_callbacks

# %% [markdown]
# ## Hyperparameters Setting

# %%
parser = ArgumentParser()
# Basic Training Control
parser.add_argument('--train_datatype', default='hdf5', type=str)
parser.add_argument('--train_h5path', default='Y:\\Segmentation_SD\\codeV2\\train_5000.hdf5', type=str)
parser.add_argument('--test_h5path', default=None, type=str)
parser.add_argument('--image_dir', default='Y:\\Segmentation_SD\\data\\image\\', type=str)
parser.add_argument('--msk_dir', default='Y:\\Segmentation_SD\\data\\mask\\', type=str)
parser.add_argument('--test_img_dir', default=None, type=str)
parser.add_argument('--test_msk_dir', default=None, type=str)
parser.add_argument('--split_rate', default=0.2, type=int)
parser.add_argument('--aug', default='true', type=str)
parser.add_argument('--stage', default='train', type=str)


# Basic Training Control
parser.add_argument('--input_size', default=256, type=int)
parser.add_argument('--batch_size', default=60, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=1e-3, type=float)

# LR Scheduler
parser.add_argument('--lr_scheduler', choices='cosine', type=str)
parser.add_argument('--lr_decay_steps', default=20, type=int)
parser.add_argument('--lr_decay_rate', default=0.5, type=float)
parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

# Restart Control
parser.add_argument('--load_best', action='store_true')
parser.add_argument('--load_dir', default=None, type=str)
parser.add_argument('--load_ver', default=None, type=str)
parser.add_argument('--load_v_num', default=None, type=int)

# Training Info
parser.add_argument('--dataset', default='Segh5_data', type=str)
parser.add_argument('--data_dir', default='ref/data', type=str)
parser.add_argument('--loss', default='bce', type=str)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--no_augment', action='store_true')
parser.add_argument('--log_dir', default='lightning_logs', type=str)

# Model Hyperparameters
parser.add_argument('--model_name', default='unet_vgg', type=str)
parser.add_argument('--encoder_name', default='vgg16_bn', type=str)
parser.add_argument('--encoder_weights', default=None, type=str)
parser.add_argument('--activation', default=None, type=str)
parser.add_argument('--encoder_depth', default=5, type=int)
parser.add_argument('--class', default=2, type=int)

# Other
parser.add_argument('--aug_prob', default=0.5, type=float)

# Add pytorch lightning's args to parser as a group.
parser = Trainer.add_argparse_args(parser)

## Deprecated, old version
# parser = Trainer.add_argparse_args(
#     parser.add_argument_group(title="pl.Trainer args"))

# Reset Some Default Trainer Arguments' Default Values
parser.set_defaults(progress_bar_refresh_rate=5)
parser.set_defaults(max_epochs=10)

args = parser.parse_args([])

# %% [markdown]
# ## Training

# %%
def load_data_module(name,**kwargs):
    # Please always name your model file name as `snake_case.py` and
    # class name corresponding `CamelCase`.
    camel_name = ''.join([i for i in name.split('_')])
    try:
        data_module = getattr(importlib.import_module('data.'+name, package=__package__), camel_name)
    except:
        raise ValueError(
            f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}'
            'Please always name your model file name as `snake_case.py` and class name corresponding `CamelCase`')
    return data_module(**kwargs)
def data_loader(args):
    if  args.train_datatype == 'hdf5':
        alldata = load_data_module(args.dataset,hdf5_dir=args.train_h5path,model='train',aug=True,aug_prob=0.2,args=args)
    else: 
        alldata = load_data_module(args.dataset,image_dir=args.image_dir,msk_dir=args.msk_dir,model='train',aug=True,aug_prob=0.2,args=args)
    trainset,valset=random_split(alldata, [int(len(alldata)*args.split_rate), len(alldata)-int(len(alldata)*args.split_rate)])
    testset=None
    print(len(trainset),len(valset))
    if  args.train_datatype is 'hdf5' and args.test_h5path is not None:
        testset = load_data_module(args.dataset,hdf5_dir=args.test_h5path,model='test',aug=True,aug_prob=0.2,args=args)
        print(len(testset))
    elif args.train_datatype is not 'hdf5' and args.test_img_dir is not None:
        testset = load_data_module(args.dataset,image_dir=args.test_img_dir,msk_dir=args.test_msk_dir,model='test',aug=True,aug_prob=0.2,args=args)
        print(len(testset))
    data_module = DInterface(trainset,valset,testset,**vars(args))
    return trainset,valset,testset,data_module
def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    trainset,valset,testset,data_module = data_loader(args)
    if load_path is None:
        model_module = MInterface(**vars(args))
    else:
        model_module = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path

    logger = TensorBoardLogger(save_dir='logdir', name=args.log_dir)
    args.callbacks = load_callbacks(args)
    args.logger = logger
    trainer=Trainer(gpus=1, max_epochs=args.max_epochs,progress_bar_refresh_rate=1,logger=args.logger,callbacks=args.callbacks)
    trainer.fit(model_module,data_module)

    # trainer = Trainer.from_argparse_args(args)
       # from model.unetvggpl import unetvgg
    # model=unetvgg()
    # train_dataloader=DataLoader(trainset, batch_size=args.batch_size,  shuffle=True)
    # val_dataloader=DataLoader(valset, batch_size=args.batch_size,  shuffle=True)
    # trainer.fit(model,train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# %%
main(args)

# %%



