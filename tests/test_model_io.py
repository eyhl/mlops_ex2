import os
import torch
from torch.utils.data.dataloader import DataLoader
from src.models.model import cnnModel
from src.data.make_dataset import mnistDataset

def test_model_input_output_shapes(model=cnnModel(), data_path="data/processed", data_prefix="train", 
                                   batch_size=64):
    
    # find stored data and labels tensors
    data_file = data_prefix + "_images.pt"
    labels_file = data_prefix + "_labels.pt"
    data = torch.load(os.path.join(data_path, data_file))
    labels = train_labels = torch.load(os.path.join(data_path, labels_file))

    # load into dataloader
    data_set = mnistDataset(data, labels)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    # take first random iteration of data
    input, labels = next(iter(data_loader))
    output = model.forward(input)
    assert input.shape == torch.Size([batch_size, 1, 28, 28])
    assert output.shape == torch.Size([batch_size, 10])
    assert labels.shape == torch.Size([batch_size])