import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import pytorch_lightning as pl
import torch.nn.functional as F
import logging
from fogsr.trainer.loss import CharbonnierLoss, structural_similarity
from fogsr.utils import extract_cls_conf,restore_images
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio


class VRTNet(pl.LightningModule):
    def __init__(self,model:nn.Module,
                 train_loader:torch.utils.data.DataLoader,
                 val_loader:torch.utils.data.DataLoader,
                 optimizer: dict = None,
                 scheduler: dict = None,
                 batch_size=1
                 ):
        super(VRTNet,self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.model = model
        self.criterion = CharbonnierLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_init, self.optimizer_conf = extract_cls_conf(optimizer) \
            if optimizer is not None else (None, None)
        self.scheduler_init, self.scheduler_conf = extract_cls_conf(scheduler) \
            if scheduler is not None else (None, None)
        
    def forward(self, *args, **kwargs):
        y = self.model(*args, **kwargs)
        return y
    
    def training_step(self, batch,batch_idx) :
        inputs = batch['lq']
        targets = batch['gt']
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        self.lr_schedulers().step()
        self.log("Learning Rate", self.lr_schedulers().get_last_lr()[0],prog_bar=True)
        self.log('Train Loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch,batch_idx):
        inputs = batch['lq']
        targets = batch['gt']
        s = time.time()
        outputs = self.forward(inputs)
        e = time.time()
        loss = self.criterion(outputs, targets)
        self.log('Inference Time Cost (seconds)', e - s)
        output_images = restore_images(outputs)
        target_images = restore_images(targets)
        num_frames = outputs.shape[1]
        current_frame = 0
        for output, target in zip(output_images, target_images):
            psnr = peak_signal_noise_ratio(output, target)
            ssim = structural_similarity(output, target)
            if psnr > 100:
                logging.warning("psnr is %f? is impossible" % psnr)
            else:
                self.log('PSNR', psnr)
                self.log('SSIM', ssim)
                self.log('frame %d PSNR' % current_frame, psnr)
                self.log('frame %d SSIM' % current_frame, ssim)

            output_y = rgb2ycbcr(output)[..., 0]
            target_y = rgb2ycbcr(target)[..., 0]
            psnr_y = peak_signal_noise_ratio(output_y, target_y, data_range=255)
            ssim_y = structural_similarity(output_y, target_y)
            if psnr > 100:
                logging.warning("psnr is %f? is impossible" % psnr)
            else:
                self.log('Y channel PSNR', psnr_y)
                self.log('Y channel SSIM', ssim_y)
                self.log('frame %d Y channel PSNR' % current_frame, psnr_y)
                self.log('frame %d Y channel SSIM' % current_frame, ssim_y)
            current_frame = (current_frame + 1) % num_frames
            self.log('val_loss', loss,prog_bar=True)
            return {'val_loss': loss}
        
    def configure_optimizers(self):
        optim_params = [v for k, v in self.model.named_parameters() if v.requires_grad]
        optimizer = self.optimizer_init(optim_params, **self.optimizer_conf)
        lr_scheduler = self.scheduler_init(optimizer, **self.scheduler_conf)
        return [optimizer], [lr_scheduler]
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.val_dataloader()
    

    