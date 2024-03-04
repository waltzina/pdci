# prerequisites
import argparse
import torch
import torch.nn as nn
import os
import torch.optim as optim
torch.manual_seed(0)
from torch.autograd import Variable
from torchvision.utils import save_image

import datasetGene as ds
import scipy.io as sio
import nets
import numpy as np



def tensor2np(x):
    return x.clone().detach().cpu().numpy()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def change_lr(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def gene_npsave(y,imsz):
    y_save = y.clone().detach().cpu().squeeze().numpy()
    return np.reshape(y_save, (y_save.shape[0], imsz[0], imsz[1]))

'''
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[1]
        w_x = x.size()[2]
        count_h = self._tensor_size(x[:,1:,:])
        count_w = self._tensor_size(x[:,:,1:])
        #print(count_h,count_w)
        h_tv = torch.pow((x[:,1:,:]-x[:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,1:]-x[:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]
'''

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # x is batchsize*h*w
        x = x.squeeze()
        [batch_size,h, w] = x.shape
        dx = x[...,1:, :-1] - x[...,:h - 1, :-1]
        dy = x[...,:-1, 1:] - x[...,:-1, :w - 1]
        return self.TVLoss_weight*torch.sqrt(dx ** 2 + dy ** 2 + 1e-7).mean()

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]

class L1Loss(nn.Module):
    def __init__(self,L1_weight=1):
        super(L1Loss,self).__init__()
        self.L1_weight = L1_weight

    def forward(self,x):
        batch_size = x.size()[0]
        return self.L1_weight*(torch.abs(x)).mean()

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]


def train_ae():
    #!clear
    GPU_number = args.gpu  # GPU number

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_number)

    if GPU_number not in [0, 1, 2, 3]:
        print("Invalid GPU_number")
        exit()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(201)
    batch_size = 256

    # dataset = ds.Dataset_duke_letter()
    fnames = ['train2speed_6fb.mat','train2speed_4fb.mat','train2speed_3fb.mat']
    for indf in range(len(fnames)):
        dataset = ds.Dataset_duke_dcs('../data/%s'%fnames[indf])
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        imsz = dataset.imsz
        z_dim = args.n_z
        x_dim = dataset.xdim
        y_dim = dataset.ydim

        n_hid = args.n_hid

        model = nets.SAE(x_dim=x_dim,y_dim=y_dim, z_dim=z_dim, n_hid=n_hid).to(device)

        k = 0
        for name, layer in model.named_modules():
            if k > 0:
                if not 'drop' in name:
                    torch.nn.init.xavier_normal_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
            k += 1

        criterion1 = nn.MSELoss(reduction='mean')
        criterion2 = TVLoss(TVLoss_weight=0.1)
        criterion3 = L1Loss(L1_weight=0.02)


        loss_best = 100.0

        # optimizer
        lr = 0.0008
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,nesterov=True)
        num_epochs = 2000
        loss_train_tot = []
        loss_test_tot = []
        for epoch in range(num_epochs):
            for iter, (x, y) in enumerate(train_loader):
                gnz = torch.randn(x.shape)
                x = x#+0.1*gnz*torch.abs(x).detach()
                x = x.to(device)

                y = y.to(device)
                y_hat,z_hat = model(x)
                #y_hat_reshaped = torch.reshape(y_hat,(y_hat.shape[0],imsz[0], imsz[1]))
                loss = criterion1(y_hat, y)+criterion2(y_hat)+criterion3(y_hat)#+5e-3*torch.pow(z_hat,2).mean()


                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % num_epochs== 0:
                x = dataset.xtrain.to(device)
                y = dataset.ytrain.to(device)

                y_hat,z_hat = model(x)
                loss_mse_train = criterion1(y_hat, y)
                loss_tv_train = criterion2(y_hat)
                loss_l1_train = criterion3(y_hat)

                y_train_save = gene_npsave(y,imsz)
                y_train_hat_save = gene_npsave(y_hat,imsz)

                ### test1
                x = dataset.xtest.to(device)
                y = dataset.ytest.to(device)

                y_hat,z_hat = model(x)
                loss_test1 = criterion1(y_hat, y)

                y_t1_save = y.clone().detach().cpu().squeeze().numpy()
                y_t1_hat_save = y_hat.clone().detach().cpu().squeeze().numpy()

                ### test2
                #x = dataset.xtest2.to(device)
                #y = dataset.ytest2.to(device)

                #y_hat,z_hat = model(x)
                #loss_test2 = criterion1(y_hat, y)

                #y_t2_save = gene_npsave(y,imsz)
                #y_t2_hat_save = gene_npsave(y_hat,imsz)

                print('Epoch [{}/{}], trainMSELoss: {:.8f}, trainTVLoss: {:.8f}, trainL1Loss: {:.8f}, testMSELoss: {:.8f}'
                      .format(epoch + 1, num_epochs, loss_mse_train.item(), loss_tv_train.item(), loss_l1_train.item(), loss_test1.item()))
                loss_train_tot.append(loss_mse_train.item())
                loss_test_tot.append(loss_test1.item())




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n_z', default=25, type=int)
    parser.add_argument('--n_hid', default=200, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    print(args)
    train_ae()
