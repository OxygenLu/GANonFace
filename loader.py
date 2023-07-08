import os
import random
import shutil
import sys
import random
from PIL import Image
from torch.utils.data import Dataset

# # 根目录
# dir_path = "../GAN/Celebrity Faces Dataset"
# face = "/GAN/face"
# if not os.path.exists(face):
#     os.makedirs(face)

# 建立列表，用于保存图片信息
# file_list = []

# for file in os.listdir(dir_path):  # file为current_dir当前目录下图片名
#     filename = file
#     file_list.append(filename)  #
# print(len(file_list))

# iter = 0
# i = 0
# for iter in range(0, len(file_list)):
#     for item in os.listdir(os.path.join(dir_path, file_list[iter]) ):
#         image_name = file_list[iter]+'.'+item
#         shutil.copy2(os.path.join(dir_path, file_list[iter], item), os.path.join(face, image_name))
#         i = i+1


# 读取文件夹图片
class myDataset(Dataset):
    def __init__(self, data_dir, transform):

        self.data_dir = data_dir
        self.transform = transform
        self.img_names = [name for name in list(filter(lambda x: x.endswith(".jpg"), os.listdir(self.data_dir)))]

    def __getitem__(self, index):
        path_img = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        if len(self.img_names) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.img_names)
