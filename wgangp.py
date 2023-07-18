import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from loader import myDataset
import argparse
import math
import matplotlib.image as im
from PIL import Image
from torchvision.utils import save_image
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

os.makedirs("images", exist_ok=True)


#----------------------parameters------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=6000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
lambda_gp = 10
print(opt,'\n')


img_shape = (opt.channels, opt.img_size, opt.img_size)
#
tb_write = SummaryWriter(log_dir="../GAN-ON-FACE/run/face_exp5")
if os.path.exists("../GAN-ON-FACE/weights") is False:
    os.makedirs("../GAN-ON-FACE/weights")


    
#--------------------datasets making ----------------------
dataset = '../GAN-ON-FACE/face'

trans = transforms.Compose([
        transforms.Resize([opt.img_size,opt.img_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

data = myDataset(data_dir=dataset,transform=trans)

dataloader = torch.utils.data.DataLoader(
    dataset=data,
    batch_size = opt.batch_size,
    shuffle = True,

)
# print(dataloader)
print("loader successd!")


#-------------------Network----------------
cuda = True if torch.cuda.is_available() else False
print('cuda is',cuda)


#Generator
class Generator(nn.Module):
    Embeddedsize = 128
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

#Generator
class Generator_cov(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Filters [1024, 512, 256,128,64]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.model = nn.Sequential(
            # Z latent vector 128
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=5, stride=1, padding=0, bias= False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias= False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            # output of main module --> Image (Cx32x32)
            nn.Tanh())

    # def forward(self, x):
    #      y = self.generator(x.view(x.shape[0], x.shape[1], 1, 1))
    #      return y
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # nn.Softmax()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class Discriminator_cov(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # output of main module --> State (1024x4x4)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=5, stride=1, padding=0, bias=False),
            # 扁平化
            nn.Flatten(),
            # 输出是否真实数据 (0 or 1)
            nn.Sigmoid()
            )
    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
    # def forward(self, x):
    #     y = self.model(x)
    #     return y




generator = Generator_cov()
discriminator = Discriminator_cov()


# -----------------Optimizers---------------------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr,betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if cuda:
    generator.cuda()
    discriminator.cuda()
    # adversarial_loss.cuda()


# gp function
def compute_gradient_penalty(D, real_sample, fake_sample):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_sample.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_sample + (1 - alpha) * fake_sample).requires_grad_(True)
    d_interpolates = D(interpolates)
    grad_tensor = Variable(Tensor(real_sample.size(0), 1).fill_(1.0), requires_grad = False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs = grad_tensor,
        create_graph = True, # 设置为True可以计算更高阶的导数 
        retain_graph = True, # 设置为True可以重复调用backward
        only_inputs = True, #默认为True，如果为True，则只会返回指定input的梯度值。 若为False，则会计算所有叶子节点的梯度，
                            #并且将计算得到的梯度累加到各自的.grad属性上去。
    )[0] # 因为返回的是一个只有一个tensor元素的list,索引0可以取出梯度张量
    gradients = gradients.view(real_sample.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim = 1) - 1)**2).mean()
    return gradient_penalty




# ----------
#  Training
# ----------
batches_done = 0
time_start = time.time()
for epoch in range(opt.n_epochs):
    # for i, (imgs, _) in enumerate(dataloader):
    for i, imgs in enumerate(dataloader):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_imgs = Variable(imgs.type(Tensor))

        # 噪音 z
        # print(f'img.shape = {imgs.shape}')
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim
        # , imgs.shape[2], imgs.shape[3]
        ))))
        fake_imgs = generator(z)
        gp = compute_gradient_penalty(discriminator, real_imgs, opt.latent_dim)
        # fake_imgs = generator(z).detach()
        d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))+ lambda_gp * gp
        d_loss.backward()
        optimizer_D.step()
    
        # Train the generator every n_critic iterations
        # 每五张图片更新一次生成器的参数
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        if i % opt.n_critic == 0:
            # Sample noise as generator input
            # Generate a batch of images
            fake_imgs = generator(z)

            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            # tb_write.add_scalar("fake loss", fake_loss, epoch)
            tb_write.add_scalar("generator loss", g_loss, epoch)
            # tb_write.add_scalar("real loss", real_loss, epoch)
            tb_write.add_scalar("discriminator loss", d_loss, epoch)
            # print(generator(z).size())
            
            # input_tensor = np.reshape(generator(z).cpu().detach().numpy(), (-1,3,128,128))
            # tb_write.add_image("wgp_image",input_tensor, epoch)

        if batches_done % opt.sample_interval == 0:
            save_image(fake_imgs.data[:25], "wgp_image/%d.png" % batches_done, nrow=5, normalize=True)
            # img_path = "./wgp_image/{}.png".format(batches_done)
            # img_PIL = Image.open(img_path)
            # img_array= np.array(img_PIL)   #将图片数据转换成numpy型才能被tensorboard读取
 
            # # writer.add_image("test", img_array, 1, dataformats='HWC')

            # tb_write.add_image("wgp_image",img_array, epoch, dataformats='HWC')
        batches_done += opt.n_critic
            
    # if (epoch+1) % 7 ==0:
    #     print('save..')
    #     torch.save(generator,'weights/g_w/g%d.pth' % epoch)
    #     torch.save(discriminator,'weights/d_w/d%d.pth' % epoch)
time_end = time.time()
time_sum = time_end - time_start
print("total time:", time_sum)

              