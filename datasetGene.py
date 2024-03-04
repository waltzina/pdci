import torch
from torch.utils import data
#import scipy.io as sio
import h5py
import numpy as np
np.random.seed(101)
class Dataset_duke_dcs(data.Dataset):
    def __init__(self,filename1):



        'Initialization'
        #mat = sio.loadmat(filename)
        ### train
        #sel_tau = np.arange(30,260,1)
        f = h5py.File(filename1, 'r')
        x = np.array(f['Xtrain'])
        x = x[:,range(10,360,1),:]
        x = np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        y = np.array(f['ytrain'])
        self.imsz = [y.shape[1],y.shape[2]]
        #y = np.reshape(y,(y.shape[0],y.shape[1]*y.shape[2]))

        xtrain = torch.tensor(x, dtype=torch.float32, requires_grad=False).squeeze()
        ytrain = torch.tensor(y, dtype=torch.float32, requires_grad=False).squeeze()




        #### test
        #f = h5py.File(filename2, 'r')
        x = np.array(f['Xtest'])
        x = x[:, range(10,360,1), :]
        x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        y = np.array(f['ytest'])
        #y = np.reshape(y, (y.shape[0], y.shape[1] * y.shape[2]))

        xtest = torch.tensor(x, dtype=torch.float32, requires_grad=False).squeeze()
        ytest = torch.tensor(y, dtype=torch.float32, requires_grad=False).squeeze()

        ntrain = xtrain.shape[0]
        ntest = xtest.shape[0]



        x = torch.cat((xtrain,xtest),0)
        y = torch.cat((ytrain,ytest),0)

        x = x - x.min()
        x = x / x.max()
        x = x - x.mean()

        xtrain = x[0:ntrain]
        ytrain = y[0:ntrain]

        xtest = x[ntrain:(ntrain+ntest)]
        ytest = y[ntrain:(ntrain+ntest)]





        self.xtrain = xtrain.clone().detach()
        self.ytrain = ytrain.clone().detach().unsqueeze(1)

        self.xtest = xtest.clone().detach()
        self.ytest = ytest.clone().detach().unsqueeze(1)


        self.xdim = x.shape[1]
        self.ydim = y.shape[1]

        del f



    def __len__(self):
        'Denotes the total number of samples'
        return self.xtrain.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.xtrain[index]
        y = self.ytrain[index]

        return x, y


