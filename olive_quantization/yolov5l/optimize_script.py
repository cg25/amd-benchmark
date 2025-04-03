import torch


def load_pytorch_origin_model(torch_hub_model_path):
    return torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=True)


class DataLoader:
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __getitem__(self, idx):
        input_data = torch.rand((self.batchsize, 3, 640, 640), dtype=torch.float16)
        label = None
        return input_data, label


def create_dataloader(data_dir, batchsize, *args, **kwargs):
    return DataLoader(batchsize)