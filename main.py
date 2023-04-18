from fogsr.trainer.lightning_MNIST_test import TestNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

input_size =  784   # 28x28
hidden_size = 500
num_classes = 10
epochs = 10
batch_size = 100
lr = 1e-4

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(save_top_k=-1)
    trainer = Trainer(max_epochs=epochs,fast_dev_run=False,log_every_n_steps=1,callbacks=[checkpoint_callback])
    model = TestNet(input_size,hidden_size,lr,batch_size,num_classes)
    trainer.fit(model)
