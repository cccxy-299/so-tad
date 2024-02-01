import argparse
import logging
import os
import glob
import timeit
import copy
import numpy as np
from datetime import datetime
import socket

from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from res_vae import VAE
from vgg19 import VGG19
from gan import Generator,Discriminator
from torchvision import transforms
from dataset import GANData,VAEData
from torch.utils.data import Dataset, DataLoader
from evluations import eval_fun
import helper as hf
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import loss_F
import torch.distributed as dist

def reduce_tensor(tensor:torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt,op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt

logger = hf.set_logging(name='log', level=logging.INFO, verbose=True)

parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", dest="batch_size", type=int, required=True)
parser.add_argument("-num_works", dest="num_works", type=int, required=True)
parser.add_argument("-vae_ep", dest="vae_ep", type=int, required=True)
parser.add_argument("-gan_ep", dest="gan_ep", type=int, required=True)
parser.add_argument("-val_ep", dest="val_ep", type=int, required=True)
parser.add_argument("-lr_d", dest="lr_d", type=float, required=True)
parser.add_argument("-lr_g", dest="lr_g", type=float, required=True)
parser.add_argument("-path", dest="path", type=int, required=True)
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()
local_rank = args.local_rank

# DDP：DDP backend initialization
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl is the fastest and most recommended backend on GPU devices

# Log record preparation
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
resume_epoch = 0  # Default is 0, change if want to resume
runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir)

batch_size = args.batch_size
num_workers = args.num_works
vae_epoch = args.vae_ep
gan_epoch = args.gan_ep
val_epoch = args.val_ep
lr_d = args.lr_d
lr_g = args.lr_g
path_index = args.path

paths = [r'./dataset',r'./dataset/123',r'./dataset/456']
root = paths[path_index] # Data set root path needs to be configured

transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Prepare VAE training data set
vae_train_data = VAEData(root, mode="train",transform=transform)
vae_test_data = VAEData(root, mode="test",transform=transform)

train_vae_sampler = torch.utils.data.distributed.DistributedSampler(vae_train_data)
val_vae_sampler = torch.utils.data.distributed.DistributedSampler(vae_test_data,shuffle=False)
vae_train_loader = DataLoader(vae_train_data,sampler=train_vae_sampler,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          drop_last=True)
vae_test_loader = DataLoader(vae_test_data,sampler=val_vae_sampler,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          drop_last=True)

# As long as one process prints
can_log = args.local_rank == 0
if can_log:
    logger.info("batch_size :{} d_lr: {} g_lr: {}".format(args.batch_size,args.lr_d,args.lr_g))

# # Create AE network.
vae_net = VAE(channel_in=3,
              ch=64,
              latent_channels=256).to(local_rank)
vae_net = DistributedDataParallel (vae_net, device_ids=[local_rank], output_device=local_rank)
# vae_net.load_state_dict(torch.load('parms/vae.pt'))
# Setup optimizer

optimizer = optim.Adam(vae_net.parameters(), lr=1e-4)
# AMP Scaler
scaler = torch.cuda.amp.GradScaler()
feature_scale = 1
# Create the feature loss module if required
if feature_scale > 0:
    feature_extractor = VGG19().to(local_rank)
    if can_log:
        logger.info("-VGG19 Feature Loss ON")
        # print("-VGG19 Feature Loss ON")
else:
    feature_extractor = None
    if can_log:
        logger.info("-VGG19 Feature Loss OFF")
        # print("-VGG19 Feature Loss OFF")

# vae 训练
if can_log:
    logger.info("VAE model start training!")
    # print("VAE model start training!")
# print("Device being used:", device)
vae_iter = 0

for epoch in range(vae_epoch):
    # break
    start_time = timeit.default_timer()
    train_loss = 0.0
    vae_net.train()
    for i, data in enumerate(tqdm(vae_train_loader, leave=False)):
        images = data[0].to(local_rank)
        recon_img, mu, log_var = vae_net(images)
        kl_loss = hf.kl_loss(mu, log_var)
        mse_loss = F.mse_loss(recon_img, images)
        loss = 1 * kl_loss + mse_loss
        #  Perceptual Losses
        if feature_extractor is not None:
            feat_in = torch.cat((recon_img, images), 0)
            feature_loss = feature_extractor(feat_in)
            loss += feature_scale * feature_loss
        loss = loss.to(local_rank) # 11.19
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 40)
        scaler.step(optimizer)
        scaler.update()
        if can_log:
            writer.add_scalar('vae_data/train_loss_iters', loss.item(), vae_iter)
            vae_iter = vae_iter + 1
        train_loss += loss.item()
    train_loss = train_loss / (len(vae_train_data) // batch_size)
    stop_time = timeit.default_timer()
    if can_log:
        writer.add_scalar('vae_data/train_loss_epoch', train_loss, epoch)
        logger.info("")
        logger.info("[{}] Epoch: {}/{} Loss: {} ".format("VAE Train", epoch + 1, vae_epoch, train_loss))

        print("Execution time: " + str(stop_time - start_time) + "\n")
    # break
    if epoch % val_epoch == 0 and epoch != 0:
        if can_log:
            logger.info("VAE model start testing!")
            # print("VAE model start testing!")
        vae_net.eval()
        start_time = timeit.default_timer()
        test_loss = 0.0
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i, data in enumerate(tqdm(vae_test_loader, leave=False)):

                    images = data[0].to(local_rank)
                    recon_img, mu, log_var = vae_net(images)
                    kl_loss = hf.kl_loss(mu, log_var)
                    mse_loss = F.mse_loss(recon_img, images)
                    loss = 1 * kl_loss + mse_loss
                    #  Perceptual Losses
                    if feature_extractor is not None:
                        feat_in = torch.cat((recon_img, images), 0)
                        feature_loss = feature_extractor(feat_in)
                        loss += feature_scale * feature_loss
                    test_loss += loss.item()
                writer.add_images("imgs", recon_img, epoch)
        test_loss = test_loss / (len(vae_test_data) // batch_size)
        stop_time = timeit.default_timer()
        if can_log:
            writer.add_scalar('vae_data/test_loss_epoch', test_loss, epoch)
            logger.info("")
            logger.info("[{}] Epoch: {}/{} Loss: {} ".format("VAE Test", epoch + 1, vae_epoch, test_loss))
            logger.info("Execution time: " + str(stop_time - start_time) + "\n")

# Save VAE model
if can_log:
    logger.info("Vae parameters saveing!")
    torch.save(vae_net.state_dict(),'parms/vae.pt')
# This ensures that other processes can only read the model after the model is saved.
dist.barrier()

# Gan datasets preparing
if can_log:
    logger.info("Gan datasets preparing!")
    # print("Gan datasets preparing!")
train_dataset = GANData(root,mode="train",transform=transform)
test_dataset = GANData(root,mode="test",transform=transform)

train_gan_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
test_gan_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,shuffle=False)

train_loader = DataLoader(train_dataset,sampler=train_gan_sampler,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=True)

test_loader = DataLoader(test_dataset,sampler=test_gan_sampler,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=True)

# Instantiate the generator and initialize the weights
netG = Generator(16, 64, 16)
netG.apply(hf.weights_init)

netD = Discriminator(16, 16, 16)
netD.apply(hf.weights_init)

netG = DistributedDataParallel(netG.cuda(), device_ids=[local_rank], output_device=local_rank)
netD = DistributedDataParallel(netD.cuda(), device_ids=[local_rank], output_device=local_rank)

# loss function
criterion = nn.BCELoss()
# Real label and fake label
real_label = 1.
fake_label = 0.

# Create optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))

# training
img_list = []
G_losses = []
D_losses = []
iters = 0
# Initialize result container
accident_dic = hf.get_key_frames(root)
test_res_dic = hf.init_res_dic(root=root, mode="test")
test_res_dic1 = hf.init_res_dic(root=root, mode="test")
train_res_dic = hf.init_res_dic(root=root, mode="train")
vae_net.eval()

best_auc = 0
iter_gan = 0

for epoch in range(gan_epoch):
    start_time = timeit.default_timer()
    tmp_train_dic = copy.deepcopy(train_res_dic)
    G_losses.clear()
    D_losses.clear()
    for i, data in enumerate(tqdm(train_loader, leave=False)):
        '''
        (1) UpdateD: maximize log(D(x)) + log(1 - D(G(z)))
        '''
        # Batch training using real labels
        netD.zero_grad()
        real_cpu = data[0].to(local_rank)

        t0 = torch.squeeze(real_cpu[:, :, 0:1, :, :], 2)
        t1 = torch.squeeze(real_cpu[:, :, 1:2, :, :], 2)
        t2 = torch.squeeze(real_cpu[:, :, 2:3, :, :], 2)
        t3 = torch.squeeze(real_cpu[:, :, -1:, :, :], 2)
        with torch.no_grad():
            # vae Encoded features
            mu0 = vae_net(t0)[1].to(local_rank)
            mu1 = vae_net(t1)[1].to(local_rank)
            mu2 = vae_net(t2)[1].to(local_rank)
            mu3 = vae_net(t3)[1].to(local_rank)
            d01 = nn.functional.normalize(mu1 - mu0).to(local_rank)
            d12 = nn.functional.normalize(mu2 - mu1).to(local_rank)

        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float).to(local_rank)
        output = netD(mu0, mu1, mu2, mu3).view(-1)
        errD_real = criterion(output, label).to(local_rank)
        errD_real.backward()
        D_x = output.mean().item()

        # Batch training using false labels
        noise = torch.autograd.Variable(
            torch.Tensor(np.random.normal(0, 1, (b_size, mu0.size(1), mu0.size(2), mu0.size(3))))).to(local_rank)
        noise1 = torch.autograd.Variable(
            torch.Tensor(np.random.normal(0, 1, (b_size, mu0.size(1), mu0.size(2), mu0.size(3))))).to(local_rank)
        noise2 = torch.autograd.Variable(
            torch.Tensor(np.random.normal(0, 1, (b_size, mu0.size(1), mu0.size(2), mu0.size(3))))).to(local_rank)
        fake = netG(mu0 + noise1, mu1 + noise2, mu2 + noise, d01, d12).to(local_rank)

        label.fill_(fake_label)
        output = netD(mu0, mu1, mu2, fake.detach()).view(-1)
        errD_fake = criterion(output, label).to(local_rank)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizerD.step()


        ############################
        # (2) UpdateG: maximize log(D(G(z)))
        ###########################
        for j in range(1):
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(mu0, mu1, mu2, fake).view(-1)

            errG = criterion(output, label)
            errCOS = loss_F.cosine_similarity(fake, mu3)
            errGDL = loss_F.gdl_loss(fake, mu3,local_rank)
            errINT = loss_F.intensity_loss(fake, mu3)
            # # errG = -torch.mean(output)
            # errG = (errG + errCOS + errGDL + errINT).to(local_rank)
            errG = (errG + errCOS).to(local_rank)

            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
        # Output training status
        if i % 50 == 0:
            if can_log:
                logger.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, gan_epoch, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        # Save the loss of each round
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if can_log:
            writer.add_scalar('data/train_g_loss_iter', errG, iters)
            writer.add_scalar('data/train_d_loss_iter', errD, iters)
            iters += 1
        # break
    av_g_loss = sum(G_losses) / len(G_losses)
    av_d_loss = sum(D_losses) / len(D_losses)
    stop_time = timeit.default_timer()
    if can_log:
        writer.add_scalar('data/train_g_loss_epoch', av_g_loss, epoch)
        writer.add_scalar('data/train_d_loss_epoch', av_d_loss, epoch)
        logger.info("")
        logger.info("[{}] Epoch: {}/{} GLoss: {} DLoss: {}".format("GAN Train", epoch + 1, gan_epoch, av_g_loss, av_d_loss))
        logger.info("Execution time: " + str(stop_time - start_time) + "\n")

    # test
    if epoch % val_epoch == 0:
        if can_log:
            logger.info("GAN model start testing!")
            # print("GAN model start testing!")
        start_time = timeit.default_timer()
        tmp_test_dic_psnr = copy.deepcopy(test_res_dic1)

        for i, data in enumerate(tqdm(test_loader, leave=False)):
            # Batch training using real labels
            netD.zero_grad()
            netG.zero_grad()
            real_cpu = data[0].to(local_rank)

            t0 = torch.squeeze(real_cpu[:, :, 0:1, :, :], 2)
            t1 = torch.squeeze(real_cpu[:, :, 1:2, :, :], 2)
            t2 = torch.squeeze(real_cpu[:, :, 2:3, :, :], 2)
            t3 = torch.squeeze(real_cpu[:, :, -1:, :, :], 2)
            netG.eval()
            netD.eval()
            with torch.no_grad():
                # vae Encoded features
                mu0 = vae_net(t0)[1].to(local_rank)
                mu1 = vae_net(t1)[1].to(local_rank)
                mu2 = vae_net(t2)[1].to(local_rank)
                mu3 = vae_net(t3)[1].to(local_rank)
                d01 = nn.functional.normalize(mu1 - mu0).to(local_rank)
                d12 = nn.functional.normalize(mu2 - mu1).to(local_rank)
                fake = netG(mu0, mu1, mu2, d01, d12).detach()

                res_cos = []
                batch_size = fake.shape[0]
                p = fake.view(batch_size, -1)
                x = mu3.view(batch_size, -1)
                cos = 1 - F.cosine_similarity(p, x)
                for j in range(fake.shape[0]):
                    res_cos.append(cos[j].item())
                res_cos = torch.from_numpy(np.array(res_cos)).to(local_rank)


                gather_list = [torch.zeros(batch_size, dtype=torch.float64).to(local_rank) for _ in range(torch.distributed.get_world_size())]
                gather_list1 = [torch.zeros(batch_size, dtype=torch.int64).to(local_rank) for _ in range(torch.distributed.get_world_size())]
                gather_list2 = [torch.zeros(batch_size, dtype=torch.int64).to(local_rank) for _ in range(torch.distributed.get_world_size())]

                dist.all_gather(gather_list2, data[3].to(local_rank))
                dist.all_gather(gather_list1, data[1].to(local_rank))
                dist.all_gather(gather_list, res_cos)

                res1 = []
                res2 = []
                res3 = []
                for i in range(len(gather_list)):
                    for j in range(len(gather_list[i])):
                        res1.append(gather_list[i][j])

                for i in range(len(gather_list1)):
                    for j in range(len(gather_list1[i])):
                        res2.append(gather_list1[i][j])

                for i in range(len(gather_list2)):
                    for j in range(len(gather_list2[i])):
                        res3.append(gather_list2[i][j])
                hf.preds_to_dic(tmp_test_dic_psnr, res1, res2, res3)
        nrmse_10, F1_10, AUC_10 = eval_fun(tmp_test_dic_psnr, accident_dic, threshold=0.1)
        nrmse_20, F1_20, AUC_20 = eval_fun(tmp_test_dic_psnr, accident_dic, threshold=0.2)
        nrmse_30, F1_30, AUC_30 = eval_fun(tmp_test_dic_psnr, accident_dic, threshold=0.3)
        stop_time = timeit.default_timer()

        if can_log:
            logger.info("GAN parameters saveing!")
            torch.save(netD.state_dict(), 'parms/{}_netD.pt'.format(epoch))
            torch.save(netG.state_dict(), 'parms/{}_netG.pt'.format(epoch))
            writer.add_scalar('data/val_nrmse_10_epoch', nrmse_10, epoch)
            writer.add_scalar('data/val_F1_10_epoch', F1_10, epoch)
            writer.add_scalar('data/val_AUC_10_epoch', AUC_10, epoch)
            writer.add_scalar('data/val_nrmse_20_epoch', nrmse_20, epoch)
            writer.add_scalar('data/val_F1_20_epoch', F1_20, epoch)
            writer.add_scalar('data/val_AUC_20_epoch', AUC_20, epoch)
            writer.add_scalar('data/val_nrmse_30_epoch', nrmse_30, epoch)
            writer.add_scalar('data/val_F1_30_epoch', F1_30, epoch)
            writer.add_scalar('data/val_AUC_30_epoch', AUC_30, epoch)
            logger.info("")
            logger.info(
                "[{}] Epoch: {}/{} threshold: 0.1  nrmse: {} F1: {} AUC: {} ".format("GAN Val", epoch + 1, gan_epoch,
                                                                                    nrmse_10, F1_10, AUC_10))
            logger.info(
                "[{}] Epoch: {}/{} threshold:0.2 nrmse: {} F1: {} AUC: {} ".format("GAN Val", epoch + 1, gan_epoch, nrmse_20,
                                                                                   F1_20, AUC_20))
            logger.info(
                "[{}] Epoch: {}/{} threshold:0.3  nrmse: {} F1: {} AUC: {} ".format("GAN Val", epoch + 1, gan_epoch,
                                                                                    nrmse_30, F1_30, AUC_30))
            logger.info("Execution time: " + str(stop_time - start_time) + "\n")

        # This ensures that other processes can only read the model after the model is saved.
        dist.barrier()
        netG.train()
        netD.train()

