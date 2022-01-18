import os
import torch
from src.data.make_dataset import mnistDataset
import pytest

@pytest.mark.skipif(not os.path.exists('data/processed/train_images.pt'), reason="Data files not found")
def test_processed_data():
    # relative path to processed data
    data_path = 'data/processed'

    train_images = torch.load(os.path.join(data_path, 'train_images.pt'))
    train_labels = torch.load(os.path.join(data_path, 'train_labels.pt'))
    test_images = torch.load(os.path.join(data_path, 'test_images.pt'))
    test_labels = torch.load(os.path.join(data_path, 'test_labels.pt'))

    train_data = mnistDataset(train_images, train_labels)
    test_data = mnistDataset(test_images, test_labels)

    # check number of data points
    assert len(train_data) == 40000 and len(test_data) == 5000, "Wrong number of data points: len(train data)=40000 and len(test data)=5000"

    # check that all images has shape [1, 28, 28]
    assert all([train_data[i][0].shape == torch.Size([1, 27, 28]) for i in range(len(train_data))]), "Some train images are not [1, 28, 28]"
    assert all([test_data[i][0].shape == torch.Size([1, 28, 28]) for i in range(len(test_data))]), "Some test images are not [1, 28, 28]"

    # check that all labels are present 0..9
    assert all(torch.unique(train_data[:][1]) == torch.arange(10)), "Train labels does not have 0..9 unique classes"
    assert all(torch.unique(test_data[:][1]) == torch.arange(10)),  "Test labels does not have 0..9 unique classes"
