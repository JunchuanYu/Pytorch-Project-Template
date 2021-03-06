# Pytorch Project Template

## Goal

The goal of this repo is to common Pytorch template for real DL projects so that work can easily be extended and replicated.

## How to use
No installation is needed. Directly run `git clone https://github.com/JunchuanYu/Pytorch-Project-Template.git` to clone it to your local position. copy the template to your project directory.

## File Structure

```
root-
	|-data
		|-__init__.py
		|-data_interface.py
		|-xxx_data.py
		|-xxx_data.py
		|-...
	|-dataset
        |-image
        |-mask
        |-test
        |-xxx.hdf5
	|-model
		|-__init__.py
        |-common.py
		|-model_interface.py
		|-decoder_encoder.py
		|-decoder_encoder.py
		|-...
	|-main.py
	|-utils.py
```
***<font color=lightblue>One thing that you need to pay attention to is, in order to let the `MInterface` and `DInterface` be able to parse your newly added models and datasets automatically by simply specify the argument `--model_name` and `--dataset`, we use 'A_B' format dataset/model file, like `Seghd_data.py` or `unet_vgg.py`, and use the same content without underscore '_' for class name, like `Seghddata` and `unetvgg`.**</font>

 
## Description

- Thre are only `main.py` and `utils.py` in the root directory. The former is the entrance of the code, and the latter is a support file.

- There is a `__init__.py` file in both `data` and `modle` folder to make them into packages. In this way, the import becomes easier.

- Create a `class DInterface(pl.LightningDataModule):` in `data_interface ` to work as the interface of all different customeized Dataset files. When `trainset`,`valset` and `testset` are imported  `train_dataloader`, `val_dataloader`, `test_dataloader` functions are created by `data_interface`. In the template, we provide two dataset cases `Seghd_data` and `Segh5_data` for two different input image file formats, *.tif and *.hdf5.

- The original input data can be placed in the `datase` folder, and image format data such as *.tiff can be organized in the way of `image`, `mask`, and `test`.

- Similarly, class `class MInterface(pl.LightningModule):` are created in `model_interface` to work as the interface of all your model files. The only things you need to modify in the interface is the functions like `configure_optimizers`, `training_step`, `validation_step` which control your own training process. One interface for all models, and the difference are handled in args. 

- `main.py` is only responsible for the following tasks:

  - Define parser, add parse hyperparameters.  
  - Load the data, split the dataset for training evaluation and testing.
  - `DInterface` and `MInterface` class will automatically select and pass those arguments to the corresponding data/model class.
  - Define `Trainer` and `callback` functions, start training.

## Reference

Pytorch Lightning Explanation??? [article](https://zhuanlan.zhihu.com/p/353985363) Zhihu blog

Pytorch Lightning tutorial???[video](https://www.bilibili.com/video/BV1H64y1Q7KD?spm_id_from=333.999.0.0) bilibili

Segmentaion models library: [Repo](https://github.com/qubvel/segmentation_models.pytorch) Github