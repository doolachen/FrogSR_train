from fogsr.trainer.vrt_trainer import VRTNet
from pytorch_lightning import Trainer
from vrt_test import model_small
from fogsr.datasets.ugc.ugc_loader import ugc_loader

trainer_conf=dict(
        accelerator='gpu',
        devices=1,
        min_steps=800000,
        max_steps=800000,
        val_check_interval=1000,
        limit_val_batches=100,
        log_every_n_steps=1
)

optimizer_conf=dict(
        name='torch.optim.AdamW',
        lr=4e-4, betas=[0.9, 0.99],
)

scheduler_conf=dict(
        name='torch.optim.lr_scheduler.CosineAnnealingWarmRestarts',
        T_0=400000, eta_min=1.0e-7,
)

if __name__ == '__main__':
    trainer = Trainer(**trainer_conf)
    model,test_args = model_small()
    train_loader, val_loader = ugc_loader(test=False),ugc_loader(test=True)
    
    vrt_model = VRTNet(model=model,
                     train_loader=train_loader,
                     val_loader=val_loader,
                     optimizer=optimizer_conf,
                     scheduler=scheduler_conf,
                     )
    trainer.fit(vrt_model)
