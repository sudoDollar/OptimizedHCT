from data import LIU4K
from hct_base import HCTBase
from trainer import Trainer
from DistributedTrainer import DistributedTrainer
from inference import InferenceEngine
from params import Config
import torch
import time

config = Config().parse()
hct = HCTBase()

train_dataset = LIU4K("dataset/image_labels.csv", True)
val_dataset = LIU4K("dataset/image_labels_valid.csv", False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

#Training Analysis
if config.training_mode.lower() == "single":
    trainer = Trainer(hct, use_wandb=config.wandb)
    trainer.set_criterion("CE")
    trainer.set_data_loader(train_loader=train_loader, val_loader=val_loader)
    trainer.set_epochs(config.epochs)
    trainer.set_optimizer(config.optimizer, config.lr)
    trainer.set_scheduler("Cosine")
    trainer.set_device(config.device)
    trainer.wandbInit(config.project)
    trainer.train()
elif config.training_mode.lower() == "distributed":
    trainer = DistributedTrainer(hct, world_size=config.num_devices, use_wandb=config.wandb)
    trainer.set_data(train_dataset, val_dataset)
    trainer.set_batch_size(config.batch_size)
    trainer.set_criterion("CE")
    trainer.set_optimizer(config.optimizer, config.lr)
    trainer.set_epochs(config.epochs)
    trainer.set_num_workers(config.num_workers)
    trainer.set_log_path(config.log_path)
    trainer.wandbInit(config.project)
    if __name__ == '__main__':
        trainer.train()