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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time


os.makedirs("images", exist_ok=True)


#----------------------parameters------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt,'\n')

img_shape = (opt.channels, opt.img_size, opt.img_size)

#
tb_write = SummaryWriter(log_dir="../GAN-ON-FACE/run/face_exp2")
if os.path.exists("../GAN-ON-FACE/weights") is False:
    os.makedirs("../GAN-ON-FACE/weights")

    
#--------------------datasets making ----------------------
dataset = '../GAN/face'

trans = transforms.Compose([
        transforms.Resize([opt.img_size,opt.img_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

data = myDataset(data_dir=dataset,transform=trans)
# print(data,type)
dataloader = torch.utils.data.DataLoader(
    dataset=data,
    batch_size = opt.batch_size,
    shuffle = True
)
# print(dataloader)
print("loader successd!")


#-------------------Network----------------
cuda = True if torch.cuda.is_available() else False
print('cuda is',cuda)


#Generator
class Generator(nn.Module):
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
        img = img.view(img.size(0), *img_shape)
        return img


# Discrtiminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity



# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
# init_img = torch.zeros((3,128,128))
# tb_write.add_graph(generator,init_img)
# tb_write.add_graph(Discriminator,init_img)

# -----------------Loss function------------------
adversarial_loss = torch.nn.BCELoss()


# -----------------Optimizers---------------------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# ----------
#  Training
# ----------

# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    return image


time_start = time.time()
for epoch in range(opt.n_epochs):
    # for i, (imgs, _) in enumerate(dataloader):
    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 3, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        aa = discriminator(gen_imgs)
        g_loss = adversarial_loss(aa, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        bb = discriminator(real_imgs)
        real_loss = adversarial_loss(bb, valid)

        # 此处需要注意，detach()是为了截断梯度流，不计算生成网络的损失，
        # 因为d_loss包含了fake_loss，回传的时候如果不做处理，默认会计算generator的梯度，
        # 而这里只需要计算判别网络的梯度，更新其权重值，生成网络保持不变即可。ss
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        tb_write.add_scalar("fake loss", fake_loss, epoch)
        tb_write.add_scalar("generator loss", g_loss, epoch)
        tb_write.add_scalar("real loss", real_loss, epoch)
        tb_write.add_scalar("discriminator loss", d_loss, epoch)

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:4], "images/%d.png" % batches_done, nrow=5, normalize=True)
            # gen_imgs_path = "D:\\GAN\\images"
            # path_name = r'E:\BreakHis_dataset\validation\12'
            # fig = Image.open(gen_imgs_path).data[-1]
            # for item in os.listdir(path=gen_imgs_path)[-1]:
                # fig = im.imread(os.path.join(gen_imgs_path,item))
            # fig = tensor_to_PIL(gen_imgs.data[-1])

            # def plot_(epoch,input):
            #     pre = np.squeeze(input.detach().cpu().numpy())

            # fig = plot_(epoch,gen_imgs)

            # tb_write.add_figure(
            #     "newest gen_img",
            #     figure = fig,
            #     global_step = epoch
            # )
    if (epoch+1) %2 ==0:
        print('save..')
        torch.save(generator,'weights/g_w/g%d.pth' % epoch)
        torch.save(discriminator,'weights/d_w/d%d.pth' % epoch)

time_end = time.time()
time_sum = time_end - time_start
print("total time:", time_sum)
              