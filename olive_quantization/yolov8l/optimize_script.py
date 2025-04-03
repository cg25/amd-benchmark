import torch

class DataLoader:
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __getitem__(self, idx):
        input_data = torch.rand((self.batchsize, 3, 640, 640), dtype=torch.float16)
        label = None
        return input_data, label


def create_dataloader(data_dir, batchsize, *args, **kwargs):
    return DataLoader(batchsize)
