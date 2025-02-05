from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, data="train"):
        super(NPY_datasets, self).__init__()  # Thêm __init__() để gọi đúng
        self.convert_L_to_Class={
            0:0,                #background: 0->0
            147:1,              #R_Vofo: 147 ->1
            111:2,              #R_Arytenoid: 111 ->2
            230:3,              #Benign_Les: 230 ->3
            104:4,              #Malignant_les: 104 ->4
            124:5,              #L_Vofo: 124 ->5
            127:6               #L_Arytenoid: 127 ->6
        }
        if data=="train":
            image_folder="train/images/"
            mask_folder="train/masks/"
            #images_list = sorted(os.listdir(path_Data+'train/images/'))
            #masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            #self.data = []
            #for i in range(len(images_list)):
            #    img_path = path_Data+'train/images/' + images_list[i]
            #    if i < len(masks_list):  # Kiểm tra xem chỉ số i có hợp lệ không
            #        mask_path = path_Data+'train/masks/' + masks_list[i]
            #        self.data.append([img_path, mask_path])  # Chỉ thêm vào nếu mask_path hợp lệ
            #    else:
            #        print(f"Cảnh báo: Không tìm thấy mặt nạ cho hình ảnh {img_path}.")  # Thêm thông báo cảnh báo
            self.transformer = config.train_transformer
        elif data=="val":
            image_folder="val/images/"
            mask_folder="val/masks/"
            #images_list = sorted(os.listdir(path_Data+'val/images/'))
            #masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            #self.data = []
            #for i in range(len(images_list)):
            #    img_path = path_Data+'val/images/' + images_list[i]
            #    if i < len(masks_list):  # Kiểm tra xem chỉ số i có hợp lệ không
            #        mask_path = path_Data+'val/masks/' + masks_list[i]
            #        self.data.append([img_path, mask_path])  # Chỉ thêm vào nếu mask_path hợp lệ
            #    else:
            #        print(f"Cảnh báo: Không tìm thấy mặt nạ cho hình ảnh {img_path}.")  # Thêm thông báo cảnh báo
            self.transformer = config.test_transformer
        elif data=="test":
            image_folder="test/images/"
            mask_folder="test/masks/"
            #images_list = sorted(os.listdir(path_Data+'test/images/'))
            #masks_list = sorted(os.listdir(path_Data+'test/masks/'))
            #self.data = []
            #for i in range(len(images_list)):
            #    img_path = path_Data+'val/images/' + images_list[i]
            #    if i < len(masks_list):  # Kiểm tra xem chỉ số i có hợp lệ không
            #        mask_path = path_Data+'val/masks/' + masks_list[i]
            #        self.data.append([img_path, mask_path])  # Chỉ thêm vào nếu mask_path hợp lệ
            #    else:
            #        print(f"Cảnh báo: Không tìm thấy mặt nạ cho hình ảnh {img_path}.")  # Thêm thông báo cảnh báo
            self.transformer = config.test_transformer
        else:
            raise ValueError("Data argument is invalid, must on of these values:train,val, test!")

        images_list = sorted(os.listdir(path_Data+image_folder))
        masks_list = sorted(os.listdir(path_Data+mask_folder))
        self.data = []
        for i in range(len(images_list)):
            img_path = path_Data+ image_folder + images_list[i]
            if i < len(masks_list):  # Kiểm tra xem chỉ số i có hợp lệ không
                mask_path = path_Data+ mask_folder + masks_list[i]
                self.data.append([img_path, mask_path])  # Chỉ thêm vào nếu mask_path hợp lệ
            else:
                print(f"Warning: cannot find mask for image {img_path}.")  # Thêm thông báo cảnh báo

    # Data binary ISIC   
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) 
        mapping_array = np.vectorize(self.convert_L_to_Class.get)
        msk=mapping_array(msk)
        img, msk = self.transformer((img, msk))
        return img, msk, img_path

    # Data 6 classes VoFo 
    # def __getitem__(self, indx):
    #     img_path, msk_path = self.data[indx]
    #     img = np.array(Image.open(img_path).convert('RGB'))
    #     msk = np.array(Image.open(msk_path).convert('L'))  # Đọc mặt nạ như mảng 2D
    #     msk_one_hot = np.zeros((msk.shape[0], msk.shape[1], 6))  # Khởi tạo mảng one-hot với 6 lớp
    #     for i in range(6):
    #         msk_one_hot[:, :, i] = (msk == i).astype(int)  # Chuyển đổi mặt nạ thành one-hot encoding
    #     #print("b1",np.unique(msk_one_hot))
    #     img, msk_one_hot = self.transformer((img, msk_one_hot))  # Sử dụng msk_one_hot
    #     msk_one_hot = (msk_one_hot>=0.5).to(torch.float32)          #Note
    #     #print("b2",np.unique(msk_one_hot))
    #     return img, msk_one_hot


    def __len__(self):
        return len(self.data)
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
    