import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, n_hid):
        super(SAE, self).__init__()

        self.z_dim = z_dim
        self.n_hid = n_hid

        self.enc1 = nn.Linear(x_dim, n_hid)
        self.edrop1 = nn.Dropout(p=0.05)
        self.enc2 = nn.Linear(n_hid, n_hid)
        self.edrop2 = nn.Dropout(p=0.05)
        self.enc3 = nn.Linear(n_hid, n_hid)
        self.edrop3 = nn.Dropout(p=0.05)
        self.enc4 = nn.Linear(n_hid, n_hid)
        self.edrop4 = nn.Dropout(p=0.05)
        self.enc5 = nn.Linear(n_hid, n_hid)
        self.edrop5 = nn.Dropout(p=0.05)
        self.enc6 = nn.Linear(n_hid, z_dim)

        '''

        self.dec2 = nn.Linear(n_hid, n_hid)
        self.ddrop2 = nn.Dropout(p=0.05)
        self.dec3 = nn.Linear(n_hid, n_hid)
        self.ddrop3 = nn.Dropout(p=0.05)
        self.dec4 = nn.Linear(n_hid, n_hid)
        self.ddrop4 = nn.Dropout(p=0.05)
        self.dec5 = nn.Linear(n_hid, n_hid)
        self.ddrop5 = nn.Dropout(p=0.05)
        self.dec6 = nn.Linear(n_hid, y_dim)
        '''
        self.dec1 = nn.Linear(z_dim, n_hid)
        self.dec2 = nn.Linear(n_hid, 16 * 4 * 3)
        self.d1 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)
        self.d4 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)
        self.d5 = nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # encoder
        enc_h1 = F.relu((self.enc1(x)))
        enc_h1 = self.edrop1(enc_h1)
        enc_h2 = F.relu((self.enc2(enc_h1)))
        enc_h2 = self.edrop1(enc_h2)
        enc_h3 = F.relu((self.enc3(enc_h2)))
        # enc_h4 = F.relu((self.enc4(enc_h3)))
        # enc_h5 = F.relu((self.enc5(enc_h4)))

        z = self.enc6(enc_h3)
        # z = F.tanh(z)

        # decoder
        dec_h1 = F.relu((self.dec1(z)))
        dec_h2 = F.relu(self.dec2(dec_h1))
        hd2 = F.relu(self.d1(dec_h2.view(dec_h2.size(0), 16, 4, 3)))
        hd3 = F.relu(self.d2(hd2))
        hd4 = F.relu(self.d3(hd3))
        y = F.relu(self.d4(hd4))
        #y = F.relu(self.d5(hd5))

        '''
        dec_h2 = F.relu((self.dec2(dec_h1)))
        dec_h2 = self.ddrop2(dec_h2)
        dec_h3 = F.relu((self.dec3(dec_h2)))
        dec_h3 = self.ddrop2(dec_h3)
        #dec_h4 = F.relu((self.dec4(dec_h3)))
        #dec_h5 = F.relu((self.dec5(dec_h4)))
        #y = torch.sigmoid(self.dec6(dec_h3))
        y = F.relu(self.dec6(dec_h3))

        '''

        return y, z






