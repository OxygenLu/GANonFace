from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from loader import myDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 3000
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

tb_write = SummaryWriter(log_dir="../GAN-ON-FACE/run/conv_exp1")
if os.path.exists("../GAN-ON-FACE/weights") is False:
    os.makedirs("../GAN-ON-FACE/weights")

# dataroot='face'
# dataset = datasets.ImageFolder(root=dataroot,
#                            transform=transforms.Compose([
#                                transforms.Resize(IMAGE_SIZE),
#                                transforms.CenterCrop(IMAGE_SIZE),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
# # Create the dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
#                                          shuffle=True, num_workers=2)
#--------------------datasets making ----------------------
dataset = '../GAN-ON-FACE/face'

trans = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

data = myDataset(data_dir=dataset,transform=trans)

dataloader = torch.utils.data.DataLoader(
    dataset=data,
    batch_size=BATCH_SIZE,
    shuffle = True,
    num_workers=2
)
# print(dataloader)
print("loader successd!")




class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            #注意不能用BN
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

# 权值初始化方案
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


#这里的critic就是discriminator
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

#测试用
fixed_noise = torch.randn(49, Z_DIM, 1, 1).to(device)

# 优化器初始化
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

#设置为训练状态
gen.train()
critic.train()


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    # 从均匀分布U(0,1)中采样
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # 插值
    interpolated_images = real * alpha + fake * (1 - alpha)

    # 计算混合后的判别器输出
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # 打平
    gradient = gradient.view(gradient.shape[0], -1)
    # 计算梯度向量的2范数
    gradient_norm = gradient.norm(2, dim=1)
    # 计算最终的gp
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


    img_list = []
print('training...')
if __name__ == '__main__':
    batches_done = 0
    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        for batch_idx, real in enumerate(dataloader):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # 训练判别器: max E[critic(real)] - E[critic(fake)]
            #等价于 min -（E[critic(real)] - E[critic(fake)]）

            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # 训练生成器: max E[critic(gen_fake)]
            # 等价于 min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            tb_write.add_scalar("generator loss",loss_gen, epoch)
            tb_write.add_scalar("discrminator loss",loss_critic, epoch)
            # 打印训练过程信息
            if batch_idx % 100 == 0 :

                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )
                #测试
                with torch.no_grad():
                    fake = gen(fixed_noise).detach().cpu()
                # 取49张图片
                # img_list.append(torchvision.utils.make_grid(fake[:25], normalize=True))
            if batches_done % 400 == 0:
                save_image(fake.data[:25], "wgp_image/%d.png" % batches_done, nrow=5, normalize=True)
                # 遍历文件夹中的所有图片
                # for filename in os.listdir("wgp_image/"):
                # # 仅处理JPEG文件
                #     # if filename.endswith(".png"):
                #     # 读取图像并将其转换为PyTorch张量
                #     img = Image.open(.path.join("wgp_image/", filename))
                #     img_tensor = F.toso_tensor(img)
                #     # 将张量转换为三维形状，并将其写入TensorBoard
                #     writer.add_image(filename, img_tensor, da
                # taformats="CHW")
                # while True:
    # 找到最新的JPEG文件
                    # newest_file = max(os.listdir("wgp_image/"), key=lambda x: os.path.getctime(os.path.join("wgp_image/", x)))
                    # if newest_file.endswith(".png"):
                    #     # 读取最新的图像并将其转换为PyTorch张量
                    #     img = Image.open(os.path.join("wgp_image/", newest_file)).convert('RGB')
                    #     img_tensor = F.to_tensor(img)

                    #     # 将张量转换为三维形状，并将其写入TensorBoard
                    #     writer.add_image("gen_face", img_tensor, dataformats="CHW")
                    # break
            batches_done += 5
