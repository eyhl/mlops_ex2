"""
@author: Eigil Lippert
@date: Jan 2022
@description: Wrapper class which enables running training and evaluation 
of CNN networks on corrupted mnist data from terminal.
"""
import argparse
import os
import sys
sys.path.append("..")
print(sys.path)

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import cnnModel, train, validation
from torch.utils.data.dataloader import DataLoader

from src.data.make_dataset import mnistDataset


def train(self):
    # print("Training day and night")
    # parser = argparse.ArgumentParser(description="Training arguments")
    # parser.add_argument("save_model_to", default="models")
    # # add any additional argument that you want
    # parser.add_argument("save_plots_to", default="reports/figures")
    # args = parser.parse_args(sys.argv[2:])
    # print(args)
    args = {'save_model_to': "models", "save_plots_to": "reports/figures"}
    
    n_epochs = 10

    # static relative data path
    processed_data_path = "data/processed"
    # Load model and data
    model = cnnModel()
    # train_images, train_labels, _, _ = mnist_loader()

    train_images = torch.load(os.path.join(processed_data_path, 'train_images.pt'))
    train_labels = torch.load(os.path.join(processed_data_path, 'train_labels.pt'))

    train_set = mnistDataset(train_images, train_labels)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # TODO: Implement training loop here
    train_losses = train(
        model,
        train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=n_epochs,
    )

    plt.figure(figsize=(5, 5))
    plt.plot(range(n_epochs), train_losses, color="#4b0082", linewidth=2, alpha=0.8)
    plt.grid()
    plt.title("Training loss per epoch")
    plt.savefig(os.path.join(args.save_plots_to, "training-loss-per-epoch.png"), dpi=200)
    plt.show()

    # Save model
    torch.save(model.state_dict(), os.path.join(args.save_model_to, "checkpoint.pth"))

def evaluate(self):
    print("Evaluating until hitting the ceiling")
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

    criterion = torch.nn.NLLLoss()

    # TODO: conisder adding transform later
    test_images = torch.load(os.path.join(processed_data_path, 'test_images.pt'))
    test_labels = torch.load(os.path.join(processed_data_path, 'test_labels.pt'))
    test_set = mnistDataset(test_images, test_labels)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    with torch.no_grad():
        test_loss, accuracy = validation(model, test_loader, criterion)

    print(
        "Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
        "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)),
    )

    # Make sure dropout and grads are on for training
    model.train()


if __name__ == "__main__":
    train()
