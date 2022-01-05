import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class cnnModel(nn.Module):
    """
    A class to represent a CNN torch model.

    ...

    Attributes
    ----------
    conv1:
        2d convolutional layer
    pool:
        Max pool 2d layer
    conv2:
        2d convolutional layer
    fc1:
        Fully connected layer
    fc2:
        Fully connected layer
    fc3:
        Fully connected layer (output layer)
    dropout:
        Dropout layer

    Methods
    -------
    forward(x):
        Makes a forward pass through the network

    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the CNN.

        Parameters
        ----------
            conv1:
                2d convolutional layer
            pool:
                Max pool 2d layer
            conv2:
                2d convolutional layer
            fc1:
                Fully connected layer
            fc2:
                Fully connected layer
            fc3:
                Fully connected layer (output layer)
            dropout:
                Dropout layer with 0.3 probability
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Input shape: torch.Size([64, 1, 28, 28])
        x = self.pool(F.relu(self.conv1(x)))

        # shape: torch.Size([64, 6, 13, 13])
        x = self.pool(F.relu(self.conv2(x)))

        # shape: torch.Size([64, 16, 5, 5])
        x = torch.flatten(x, 1)  # flatten all dims except batch

        # shape: torch.Size([64, 400])
        x = self.dropout(F.relu(self.fc1(x)))

        # shape torch.Size([64, 120])
        x = self.dropout(F.relu(self.fc2(x)))

        # shape: torch.Size([64, 84])
        x = F.log_softmax(self.fc3(x), dim=1)

        # final shape: torch.Size([64, 10]), 10 classes
        return x


def validation(model, testloader, criterion):
    """
    Returns test loss and accuracy score of a given torch model.
    NOTE: divide outputs with the total number of epochs to get
    the final values.

            Parameters:
                    model (nn.Module): A torch model
                    testloader (DataLoader): A dataloader object
                    criterion (nn.Module): A torch loss function, i.e. NLLLoss

            Returns:
                    test_loss (float): the aggregated average epoch loss
                    accuracy: the aggregated average epoch loss
    """
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(model, trainloader, criterion, optimizer=None, epochs=5):
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=1e-2)

    steps = 0
    running_loss = 0
    train_losses = []
    for e in range(epochs):
        # activate dropout
        model.train()
        for images, labels in trainloader:
            steps += 1

            # reset gradients
            optimizer.zero_grad()

            # forward pass
            output = model.forward(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            epoch_loss = running_loss / len(trainloader)
            train_losses.append(epoch_loss)

            print(
                "Epoch: {}/{}.. ".format(e + 1, epochs),
                "Training Loss: {:.3f}.. ".format(epoch_loss),
            )
            running_loss = 0

    return train_losses
