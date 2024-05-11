import argparse

class Config:

    def __init__(self):
        parser = argparse.ArgumentParser(description='High Resolution Convolutional Transformer')
        parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                            help='input batch size for training (default: 16)')
        parser.add_argument('--num-workers', type=int, default=2,
                            help='Number of I/O processes (default: 2)')
        parser.add_argument('--epochs', type=int, default=80, metavar='N',
                            help='Number of epochs to train (default: 80)')
        parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                            help='learning rate (default: 2e-5)')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to be used for training(cpu/cuda)')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='Optimizer to be used (default: Adam)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='N',
                            help='Momentum Value (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='N',
                            help='Weight Decay value (default 5e-4)')
        parser.add_argument('--dataset-path', type=str, default='./dataset',
                            help='Path of LIU4K Dataset (default: ./dataset)')
        parser.add_argument('--wandb', type=int, default=1,
                            help='Use Weight & Biases to track training progress (0/1)')
        parser.add_argument('--project', type=str, default='HCT',
                            help='Project Name for W&B')
        parser.add_argument('--training-mode', type=str, default='Distributed',
                            help='Training Mode (Parallel/Distributed/Single)')
        parser.add_argument('--num-devices', type=int, default='4',
                            help='Number of Learners/GPUs')
        parser.add_argument('--log-path', type=str, default='./log',
                            help='Path where logs will be captured')
        
        
        self._parser = parser

    def parse(self):
        return self._parser.parse_args()
