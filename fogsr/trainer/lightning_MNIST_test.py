import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch.nn.functional as F

# Test MNIST classification
class TestNet(pl.LightningModule):
    def __init__(self, input_size , hidden_size, lr,batch_size, num_classes):
        super(TestNet,self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
        self.lr = lr
        self.batch_size = batch_size
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

    def training_step(self, batch, batch_idx) :
        images, labels = batch
        images = images.reshape(-1,28*28)

        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)
        self.log('loss', loss,prog_bar=True)
        return {'loss': loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)
    
    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(root='~/datasets/MNIST',
                                                   train= True,
                                                   transform=transforms.ToTensor(),
                                                   download=True
                                                   )
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   num_workers=52,
                                                   shuffle= True,
                                                   )
        return train_loader
    
    def val_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(root='~/datasets/MNIST',
                                                 train=False,
                                                 transform=transforms.ToTensor(),
                                                 download=True
                                                 )
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=52,
                                                 shuffle= False
                                                 )
        return val_loader
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1,28*28)
        outputs = self(images)

        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)
        self.log('val_loss', loss,prog_bar=True)
        return {'val_loss': loss}
    
    # def on_validation_epoch_end(self,outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     return{'val_loss':avg_loss}
        