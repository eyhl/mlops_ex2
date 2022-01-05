"""
@author: Eigil Lippert
@date: Jan 2022
@description: Wrapper class which enables running visualising model features
"""
import argparse
import os
import sys

os.chdir("/Users/eyhli/OneDriveDTU/PhD/learning/mlops/w2/mlops_ex2")

# from torch.fx.node import get_graph_node_names
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data.dataloader import DataLoader

from src.data.make_dataset import mnistDataset
from src.models.model import cnnModel


class VisualizeModel(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python train_model.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def visualize(self):
        print("Visualize... (takes around 20 sec)")
        parser = argparse.ArgumentParser(description="Evaluation arguments")
        parser.add_argument("load_model_from", default="models/checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # static relative data path
        processed_data_path = "data/processed"

        # TODO: Implement evaluation logic here
        model = cnnModel()
        state_dict = torch.load(args.load_model_from)
        model.load_state_dict(state_dict)

        # Model in inference mode, dropout is off
        model.eval()

        # Visualize feature maps
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        test_set = torch.load(os.path.join(processed_data_path, "test_dataset_processed.pt"))
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

        # extract first linear layer after activation for every sample
        model.fc1.register_forward_hook(get_activation("fc1"))
        features, label_list = [], []
        for images, labels in test_loader:
            output = model(images)
            act = activation["fc1"].squeeze()
            features.append(act)
            label_list.append(labels)

        # stack batches and convert to numpy
        features_ = torch.vstack(features).numpy()
        label_list_ = torch.hstack(label_list).numpy()

        # tsne with 2 dimensions
        tsne = TSNE(n_components=2, init="pca").fit_transform(features_)

        # plot one point at the time (slow but clear implementation)
        plt.figure()
        for idx, (x, y) in enumerate(tsne):
            col = plt.cm.Dark2(label_list_[idx])
            plt.scatter(x, y, color=col, alpha=0.425)
        plt.grid()
        plt.savefig(os.path.join("reports/figures", "tsne_features.png"), dpi=200)
        plt.show()


if __name__ == "__main__":
    VisualizeModel()
