import torch
import numpy as np
import torch.utils.data as data
import glob,os,gdal
from gdalconst import *
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
    
class Seghddata(data.Dataset):
    def __init__(self, image_dir,msk_dir,mode='train',aug=True,aug_prob=0.2,**kwargs):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.mode=mode
        self.args=kwargs
        self.aug=aug
        self.aug_prob=aug_prob
        self.imgdirlist=glob.glob(os.path.join(image_dir,"*.tif"))
        self.mskdirlist=glob.glob(os.path.join(msk_dir,"*.tif"))
    
    
    def Load_image_by_Gdal(self,file_path):
        img_file = gdal.Open(file_path, GA_ReadOnly)
        img_bands = img_file.RasterCount#band num
        img_height = img_file.RasterYSize#height
        img_width = img_file.RasterXSize#width
        img_arr = img_file.ReadAsArray()#获取投影信息
    #     geomatrix = img_file.GetGeoTransform()#获取仿射矩阵信息
    #     projection = img_file.GetProjectionRef()
    #     return img_file, img_bands, img_height, img_width, img_arr, geomatrix, projection
        return img_bands, img_height, img_width, img_arr
    
    def read_tiff(self,file):
        img_bands, img_height, img_width, img_arr=self.Load_image_by_Gdal(file)
        # if img_bands >1 :
            # img_arr=img_arr.transpose(( 1, 2,0))
        return img_arr     
    
    def build_transform(self, mode,image,label):
        
        image=Image.fromarray(image)
        label=Image.fromarray(label)
        if mode == "train":
            if self.aug:
                data_transforms=transforms.Compose([
                    # transforms.RandomHorizontalFlip(self.aug_prob),
                    # transforms.RandomVerticalFlip(self.aug_prob),
                    # transforms.RandomRotation(10),
                    # transforms.RandomCrop(self.args.input_size),
                    transforms.ToTensor()])

                msk_transforms=transforms.Compose([
                    # transforms.RandomHorizontalFlip(self.aug_prob),
                    # transforms.RandomVerticalFlip(self.aug_prob),
                    # transforms.RandomRotation(10),
                    # transforms.RandomCrop(self.args.input_size),
                    transforms.ToTensor()])
                image=data_transforms(image)     
                label=msk_transforms(label)
                          
        else:
            if self.aug: 
                data_transforms=transforms.Compose([
                    transforms.ToTensor()])  
                msk_transforms=transforms.Compose([
                    transforms.ToTensor()])  
                image=data_transforms(image)     
                label=msk_transforms(label)
        #     transforms.CenterCrop(args.input_size)
        #     # transforms.Normalize(self.img_mean, self.img_std)
        return np.array(image,dtype=np.float32),np.array(label,dtype=np.float32)
    
    def __len__(self):
        return len(self.imgdirlist)

    def __getitem__(self, idx):
        imgpath = self.imgdirlist[idx]
        img=self.read_tiff(imgpath)
        labelpath=self.mskdirlist[idx]
        msk=self.read_tiff(labelpath)
        msk=np.expand_dims(msk,axis=0)
        # img,msk=self.build_transform(self.mode,img,msk)
        img=torch.from_numpy(np.array(img/1023,dtype=np.float32))
        msk=torch.from_numpy(np.array(msk/1023,dtype=np.float32))
        # img=img.type(torch.FloatTensor)
        # msk=msk.type(torch.FloatTensor)
        # return image/1.0, label/1.0
        return img,msk


