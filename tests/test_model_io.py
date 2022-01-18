import os
import torch
import pytest
from torch.utils.data.dataloader import DataLoader
from src.models.model import cnnModel
from src.data.make_dataset import mnistDataset

model = cnnModel()
@pytest.mark.parametrize(
    "test_input, expected", 
    [("model.forward(torch.rand((1, 1, 28, 28))).shape", torch.Size([1, 10])), 
    ("model.forward(torch.rand((1, 1, 28, 28))).isnan().all().item()", False)]
    )
def test_model_architecture(test_input, expected):
    '''
    test model io and if nans are produced
    '''
    msg = "Either model outputs wrong shape or nans"
    assert eval(test_input) == expected, msg


@pytest.mark.skipif(not os.path.exists('data/processed/train_images.pt'), reason="Data files not found")
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
    assert input.shape == torch.Size([batch_size, 1, 27, 28]), "Input shape is not [batch_size, 1, 28, 28]"
    assert output.shape == torch.Size([batch_size, 10]), "Output shape is not [batch_size, 10]"
    assert labels.shape == torch.Size([batch_size]), "Labels shape is not [batch_size]"
