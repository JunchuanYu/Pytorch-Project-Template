import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import torchmetrics
import segmentation_models_pytorch as smp


class MInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.configure_loss()
        self.train_acc=torchmetrics.Accuracy()
        self.val_acc=torchmetrics.Accuracy()
        self.model=self.load_model()

        # self.model = smp.Unet(encoder_name,encoder_weights=None,classes=clsss, activation=activation,in_channels=in_channels)
        # self.model_name = model_name
        # self.loss = loss
        # self.lr = lr
        
    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        out = self(img)
        loss = self.loss_function(out, labels)
        self.log('train_loss', loss, prog_bar=True,logger=True)
        self.log('train_acc',self.train_acc(torch.sigmoid(out),labels.int()), prog_bar=True,logger=True)
        return {'loss':loss,'acc':self.train_acc(torch.sigmoid(out),labels.int())}
    
    # def training_epoch_end(self, outputs):
    #     self.log('train_acc_epoch',self.train_acc.compute())
        
    def validation_step(self, batch, batch_idx):
        img, labels = batch
        out = self(img)
        loss = self.loss_function(out, labels)
        self.log('val_loss', loss, prog_bar=True,logger=True)
        self.log('val_acc',self.val_acc(torch.sigmoid(out),labels.int()), prog_bar=True,logger=True)
        return {'val_loss':loss,'val_acc':self.val_acc(torch.sigmoid(out),labels.int())}
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        acc_val=torch.stack([x["val_acc"] for x in outputs]).mean()
        # log_dict = {"val_loss": loss_val}
        self.log('val_loss_epoch', loss_val, prog_bar=True,logger=True)
        self.log('val_acc_epoch',acc_val, prog_bar=True,logger=True)
        return {"val_loss": loss_val, "val_acc": acc_val}
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters())
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
        
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'cross entropy':
            self.loss_function = F.cross_entropy
        elif loss == 'dice':
            self.loss_function = DiceLoss(sigmoid=True)
        else:
            raise ValueError("Invalid Loss Type!")
        # self.loss_function = nn.BCEWithLogitsLoss()
        # self.loss_function = nn.CrossEntropyLoss()
    def load_model(self):
        name = self.hparams.model_name
        # print(self.hparams)
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module('.'+name, package=__package__), camel_name)
            # print(Model)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model_module = Model()
        return self.model_module
        # print(self.model)

