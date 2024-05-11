from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import time
import wandb
import os

class DistributedTrainer:
    class Utils:
        SCHEDULER_START_POINT = 4
        def __init__(self):
            pass

        def get_data_loader(self, dataset, batch_size, num_workers, pin_memory, train:bool, rank, world_size):
            if train:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
                dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
            else:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
                dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
                # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            return dataloader
        
        def get_model(self):
            return self.model
        
        def accuracy(self, y_pred, y):
            top_pred = y_pred.argmax(1, keepdim =True)
            correct = top_pred.eq(y.view_as(top_pred)).sum()
            acc = correct.float()
            return acc, y.shape[0]
        
        def log(self, line:str, rank):
            file_path = os.path.join(self.log_path, "Log_{}.log".format(rank))
            with open(file_path, 'a') as file:
                file.write(line)

    def __init__(self, model, world_size, use_wandb, use_scheduler=True):
        self.utils = self.Utils()
        self.utils.model = deepcopy(model)
        self._model = model
        self.world_size = world_size
        self.utils.use_scheduler = use_scheduler
        if use_wandb:
            self.utils.use_wandb = "online"
        else:
            self.utils.use_wandb = "disabled"
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        

    def set_data(self, train_data, valid_data):
        self.utils.train_data = train_data
        self.utils.valid_data = valid_data

    def set_criterion(self, criterion = "CE"):
        if criterion == "CE":
            self.utils.criterion = nn.CrossEntropyLoss()

    def set_optimizer(self, optimizer:str, lr=2e-5):
        self.utils._optimizer = optimizer
        self.utils.lr = self.world_size * lr #Linear Scaling Rule

    def set_epochs(self, epochs:int):
        self.utils.epochs = epochs

    def set_batch_size(self, batch_size:int):
        self.utils.batch_size = batch_size

    def set_log_path(self, path:str):
        self.utils.log_path = path

    def set_num_workers(self, num_workers:int):
        self.utils.num_workers = num_workers
    
    @staticmethod
    def worker(rank, world_size, utils:Utils):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        train_loader = utils.get_data_loader(utils.train_data, batch_size=utils.batch_size, num_workers=utils.num_workers, pin_memory=False, train=True, rank=rank, world_size=world_size)
        val_loader = utils.get_data_loader(utils.valid_data, batch_size=utils.batch_size, num_workers=utils.num_workers, pin_memory=False, train=False, rank=rank, world_size=world_size)
        model = utils.get_model().to(rank)
        ddp_model = DDP(model, device_ids=[rank], output_device=rank)
        utils.optimizer = optim.Adam(ddp_model.parameters(), lr=utils.lr)
        utils.scheduler = lr_scheduler.CosineAnnealingLR(utils.optimizer, T_max=utils.epochs-5)
        
        if rank == 0:
            wandb.login()
        else:
            utils.use_wandb = "disabled"

        utils.log("Starting Training: \n", rank)
        utils.log("Optimizer: {}, num_workers: {}\n".format(utils._optimizer, utils.num_workers), rank)
        utils.log("Number of Devices: {}, Batch Size per GPU: {}\n".format(world_size, utils.batch_size), rank)
        utils.log("Number of Batches: {}\n\n".format(len(train_loader)), rank)
            
        with wandb.init(project=utils.project, config=utils.wConfig, mode=utils.use_wandb):
            with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/hct'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False
            ) as prof:
                for epoch in range(utils.epochs):
                    start_time_runnning = time.perf_counter()
                    train_loader.sampler.set_epoch(epoch)
                    val_loader.sampler.set_epoch(epoch)
                    max_val_acc = 0
                    
                    epoch_loss, epoch_acc, data_load_time, train_time = DistributedTrainer.train_one_epoch(ddp_model, train_loader, utils, rank, epoch, prof)
                    end_time_runnning = time.perf_counter()
                    
                    val_loss, val_acc, val_count = DistributedTrainer.evaluate_one_epoch(ddp_model, val_loader, utils, rank)

                    total = torch.tensor([val_acc, val_count], dtype=torch.float32).to(rank)
                    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
                    val_acc, val_count = total.tolist()
                    val_acc = val_acc / val_count

                    if rank == 0 and val_acc > max_val_acc:
                        max_val_acc = val_acc
                        torch.save(ddp_model.module.state_dict(), "saved_model/hct.pt")

                    if utils.use_scheduler and epoch >= utils.SCHEDULER_START_POINT:
                        utils.scheduler.step()

                    utils.log("Epoch: {}/{} Training loss: {}, Training acc: {}\n".format(epoch+1, utils.epochs, epoch_loss, epoch_acc), rank)
                    utils.log("Validation Loss: {}, Validation Accuracy: {}\n".format(val_loss, val_acc), rank)
                    utils.log("Data Loading Time: {} secs, Training time: {} secs, Running time: {} secs\n\n".format(data_load_time, train_time, end_time_runnning - start_time_runnning), rank)
                    wandb.log({"Train Loss":epoch_loss,
                                "Train Accuracy":epoch_acc,
                                "Epoch": epoch+1,
                                "Val Loss":val_loss,
                                "Val Accuracy":val_acc
                            })

        dist.destroy_process_group()

    def train(self):
        mp.spawn(DistributedTrainer.worker, args=(self.world_size, self.utils), nprocs=self.world_size)


    @staticmethod
    def train_one_epoch(model:DDP, train_loader:DataLoader, utils:Utils, rank, epoch:int, prof:torch.profiler.profile):
        train_time = []
        data_load_time = []
        epoch_loss = 0
        epoch_acc = 0
        epoch_count = 0
        model.train()
        start_time_data = time.perf_counter()
        for data in train_loader:
            end_time_data = time.perf_counter()
            data_load_time.append(end_time_data - start_time_data)
            if epoch == 4:
                prof.step()
            torch.cuda.synchronize()
            start_time_training = time.perf_counter()
            utils.optimizer.zero_grad()
            x = data[0].to(rank)
            y = data[1].to(rank)
            y_pred = model(x)
            loss = utils.criterion(y_pred, y)
            acc, n = utils.accuracy(y_pred, y)
            loss.backward()
            utils.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_count += n
            torch.cuda.synchronize()
            end_time_training = time.perf_counter()
            train_time.append(end_time_training - start_time_training)
            start_time_data = time.perf_counter()

        return epoch_loss/len(train_loader), epoch_acc/epoch_count, sum(data_load_time), sum(train_time)
        
    @staticmethod
    def evaluate_one_epoch(model:DDP, val_loader:DataLoader, utils:Utils, rank):
        epoch_loss = 0
        epoch_acc = 0
        epoch_count = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                x = data[0].to(rank)
                y = data[1].to(rank)
                y_pred = model(x)
                loss = utils.criterion(y_pred, y)
                acc, n = utils.accuracy(y_pred, y)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_count += n

        return epoch_loss/len(val_loader), epoch_acc, epoch_count
    
    def wandbInit(self, project:str):
        self.utils.wConfig = dict()
        self.utils.wConfig["learning_rate"] = self.utils.lr
        self.utils.wConfig["batch_size"] = self.utils.batch_size
        self.utils.wConfig["epochs"] = self.utils.epochs
        self.utils.wConfig["Optimizer"] = self.utils._optimizer
        self.utils.wConfig["num_workers"] = self.utils.num_workers
        self.utils.wConfig["num_devices"] = self.world_size
        self.utils.project = project

    def reset(self):
        self.model = deepcopy(self._model)
        self.utils = None
        self.world_size = None
