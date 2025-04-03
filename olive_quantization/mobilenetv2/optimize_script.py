import torch

def load_pytorch_origin_model(torch_hub_model_path):
    return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

class DataLoader:
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __getitem__(self, idx):
        input_data = torch.rand((self.batchsize, 3, 400, 300), dtype=torch.float32)
        label = None
        return input_data, label


def create_dataloader(data_dir, batchsize, *args, **kwargs):
    return DataLoader(batchsize)
