import math
from operator import pos
import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn import Parameter

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class OCLayer(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCLayer, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()
        if labels == None:
            return -output_scores.squeeze(1)
        else: 
            scores[labels == 0] = self.r_real - scores[labels == 0]
            scores[labels == 1] = scores[labels == 1] - self.r_fake

        return self.alpha*scores

    
class OCSoftmaxWithLoss(nn.Module):
    """
    OCSoftmaxWithLoss()
    
    """
    def __init__(self):
        super(OCSoftmaxWithLoss, self).__init__()
        self.m_loss = nn.Softplus()

    def forward(self, inputs):

        return self.m_loss(inputs).mean()


class Model(torch.nn.Module):
    def __init__(self, loss_type):

        super().__init__()
        self.loss_type = loss_type
        self.fc = nn.Sequential(
            nn.Linear(160, 192),
            nn.LeakyReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )
        self.se = SEModule(channels=64,bottleneck=32)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc2 = nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256,64),
            nn.LeakyReLU(inplace=True),
        )
        self.fc_out = nn.Linear(64,2)
        self.oc_out = OCLayer(feat_dim=64, r_real=0.8, r_fake=0.2, alpha=10)

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm, labels=None):

        # asv_enr = torch.unsqueeze(embd_asv_enr, -1) # shape: (bs, 192)
        # asv_tst = torch.unsqueeze(embd_asv_tst, -1) # shape: (bs, 192)
        # cm_tst = torch.unsqueeze(embd_cm, -1) # shape: (bs, 160)
        cm_tst = self.fc(embd_cm)
        x = torch.stack((embd_asv_enr,embd_asv_tst,cm_tst),2)
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc2(x)

        if self.loss_type.lower() == "ocsoftmax":
            x = self.oc_out(x, labels)
            return x

        x = self.fc_out(x)  # (bs, 2)    
        return x