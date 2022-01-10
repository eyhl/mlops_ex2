# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class mnistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]

        if self.transform:
            x = Image.fromarray(self.images[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)
        # else:
        #    x = torch.from_numpy(x)

        # if image is greyscale add unit color channel
        # torch expects (n_samples, channels, height, width)
        if len(x.shape) <= 3:
            x = x[np.newaxis, :, :]

        return torch.from_numpy(x).float(), y

    def __len__(self):
        return len(self.images)


def mnist_loader(path="../../corruptmnist", n_files=8, image_scale=255):
    """
    Loads .npz corruptedmnist, assumes loaded image values to be between 0 and 1
    """
    # load and stack the corrupted mnist dataset
    train_images = np.vstack(
        [np.load(path + "/train_{}.npz".format(str(i)))["images"] for i in range(n_files)]
    )
    train_labels = np.hstack(
        [np.load(path + "/train_{}.npz".format(str(i)))["labels"] for i in range(n_files)]
    )

    test_images = np.load(path + "/test.npz")["images"]
    test_labels = np.load(path + "/test.npz")["labels"]

    return train_images * image_scale, train_labels, test_images * image_scale, test_labels


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # loading data from .npz files
    train_images, train_labels, test_images, test_labels = mnist_loader(path=input_filepath)

    # creating train and test torch datasets
    # TODO: apply transforms
    train_dataset = mnistDataset(train_images, train_labels)
    test_dataset = mnistDataset(test_images, test_labels)

    # save datasets as tensors, contains both images and labels
    torch.save(train_dataset, os.path.join(output_filepath, "train_dataset_processed.pt"))
    torch.save(test_dataset, os.path.join(output_filepath, "test_dataset_processed.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
