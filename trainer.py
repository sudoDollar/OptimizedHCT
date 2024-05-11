from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import wandb

class Trainer:

    SCHEDULER_START_POINT = 4

    def __init__(self, model, use_wandb):
        self.model = deepcopy(model)
        self._model = model
        self.device = torch.device('cuda')
        if use_wandb:
            wandb.login()
            self.use_wandb = "online"
        else:
            self.use_wandb = "disabled"
            
    def set_data_loader(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.num_batches = len(train_loader)
        self.val_loader = val_loader

    def set_criterion(self, criterion = "CE"):
        if criterion == "CE":
            self.criterion = nn.CrossEntropyLoss()

    def set_optimizer(self, optimizer:str, lr=2e-5):
        self._optimizer = optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def set_scheduler(self, scheduler:str):
        if self._optimizer == None:
            print("Set optimizer first")
            return
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs-5)

    def set_epochs(self, epochs:int):
        self.epochs = epochs

    def set_device(self, device:str):
        if "cuda" in device and not torch.cuda.is_available():
            device = 'cpu'
            print("GPU/CUDA not available. Set to CPU")
        self.device = torch.device(device)    # device = 'cuda', 'cpu'
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train_one_epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        train_time = []
        data_load_time = []

        self.model.train()
        start_time_runnning = time.perf_counter()
        start_time_data = time.perf_counter()

        for data in self.train_loader:
            end_time_data = time.perf_counter()
            data_load_time.append(end_time_data - start_time_data)
            if str(self.device) == "cuda":
                torch.cuda.synchronize()
            start_time_training = time.perf_counter()
            x = data[0].to(self.device)
            y = data[1].to(self.device)
            
            y_pred = self.model(x)
            self.optimizer.zero_grad()
            loss = self.criterion(y_pred, y)
            acc = self.accuracy(y_pred, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if str(self.device) == "cuda":
                torch.cuda.synchronize()
            end_time_training = time.perf_counter()
            train_time.append(end_time_training-start_time_training)
            start_time_data = time.perf_counter()
        
        end_time_runnning = time.perf_counter()

        return epoch_loss/len(self.train_loader), epoch_acc/len(self.train_loader), sum(data_load_time), sum(train_time), end_time_runnning - start_time_runnning

    def train(self):
        self.train_loss = []
        self.train_acc = []
        self.train_time = []
        self.run_time = []

        print("Starting Training: ")
        print("Optimizer: {}, num_workers: {}, Device: {}".format(self._optimizer, self.train_loader.num_workers, str(self.device)))
        print("Number of Devices: {}, Batch Size per GPU: {}".format(1, self.train_loader.batch_size))
        print("Number of Batches: {}\n".format(len(self.train_loader)))

        with wandb.init(project=self.project, config=self.wConfig, mode=self.use_wandb):
            for epoch in range(self.epochs):
                print("Learning Rate: {}\n".format(self.scheduler.get_last_lr()))
                train_loss, train_acc, data_load_time, train_time, run_time = self.train_one_epoch()
                val_loss, val_acc = self.evaluate()
                if epoch >= self.SCHEDULER_START_POINT:
                    self.scheduler.step()
                self.train_time.append(train_time)
                self.run_time.append(run_time)
                self.train_loss.append(train_loss)
                self.train_acc.append(train_acc)
                print("Epoch: {}/{} Training loss: {}, Training acc: {}".format(epoch+1, self.epochs, train_loss, train_acc))
                print("Validation Loss: {}, Validation Accuracy: {}".format(val_loss, val_acc))
                print("Data Loading Time: {} secs, Training time: {} secs, Running time: {} secs\n".format(data_load_time, train_time, run_time))
                wandb.log({"Train Loss":train_loss, "Train Accuracy":train_acc, "Epoch": epoch+1, "Val Loss":val_loss, "Val Accuracy":val_acc})

    def evaluate(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()

        with torch.no_grad():
            for data in self.val_loader:
                x = data[0].to(self.device)
                y = data[1].to(self.device)

                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                acc = self.accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(self.val_loader), epoch_acc / len(self.val_loader)


    def accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim =True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc
    
    def wandbInit(self, project:str):
        self.wConfig = dict()
        self.wConfig["learning_rate"] = self.optimizer.param_groups[0]['lr']
        self.wConfig["batch_size"] = self.train_loader.batch_size
        self.wConfig["epochs"] = self.epochs
        self.wConfig["Optimizer"] = self._optimizer

        self.project = project

    
    def reset(self):
        self.model = deepcopy(self._model)
        self.optimizer = None
        self.train_loss = None
        self.train_acc = None
        self.train_time = None
        self.run_time = None
        self.scheduler = None
        self._optimizer = None
        self.wConfig = None
        self.project = None
