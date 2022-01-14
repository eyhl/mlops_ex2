"""
@author: Eigil Lippert
@date: Jan 2022
@description: Run model on specified data in .npy or .npz format
@ NOTE: rename your data to test.npz, then it works
@usage:
$ python src/models/predict_model.py \
     models/my_trained_model.pt \  # file containing a pretrained model
     data/example_images.npy  # file containing just 10 images for prediction
"""
# TODO: add functionality to look for various file types and more flexible data loading

import argparse
import sys

import torch
from model import cnnModel, validation_loop
from torch.utils.data.dataloader import DataLoader

from src.data.make_dataset import mnist_loader, mnistDataset


class PredictPretrainedModel(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for running model on specified data.",
            usage="python predict_model.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def predict(self):
        print("Predicting...")
        parser = argparse.ArgumentParser(description="Pre-trained model arguments")
        parser.add_argument("pretrained_model", default="models/checkpoint.pth")
        # add any additional argument that you want
        parser.add_argument("raw_data_path", default="data/raw")
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here
        model = cnnModel()
        state_dict = torch.load(args.pretrained_model)
        model.load_state_dict(state_dict)

        # Model in inference mode, dropout is off
        model.eval()

        criterion = torch.nn.NLLLoss()
        _, _, test_images, test_labels = mnist_loader(args.raw_data_path)
        # TODO: conisder adding transform later
        test_set = mnistDataset(test_images, test_labels)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

        with torch.no_grad():
            test_loss, accuracy = validation_loop(model, test_loader, criterion)

        print(
            "Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
            "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)),
        )


if __name__ == "__main__":
    PredictPretrainedModel()
