import numpy as np
import torch
from torch.utils.data import *

"""
Training a linear regression model for the equation y = 2x + 1
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'     # allows training to be computed on gpu if available
np.random.seed(1)                                           # initializes a random seed for reproducible results

# ---DATA GENERATION AND PREPARATION---

# Data Generation
# Generate synthetic data (the model will try to match its parameters to this data)
true_c = 1
true_m = 2
N = 100                             # total data points (to be split between training and validation)

x = np.random.rand(N, 1)            # N x 1 numpy array of random numbers between 0-1
e = 0.1 * np.random.randn(N, 1)     # random noise epsilon
y = true_m * x + true_c + e         # y = mx + c + random noise     y is also an N x 1 numpy array

# Data Preparation
# convert the data from numpy arrays into torch.tensor data types (so Pytorch can use them)
x_tensor, y_tensor = torch.as_tensor(x), torch.as_tensor(y)

# create a class and object for a torch.utils.data.Dataset object (used for torch.utils.data.DataLoader function)


class CustomDataset(Dataset):
    """
    Creates a dataset that takes a tensor of features (x_tens) and a tensor of labels (y_tens).
    The arguments to __init__ builds a list of tuples. These y_tens labels are the true labels to x_tens feature input.
    """
    def __init__(self, x_tens, y_tens):
        self.x = x_tens
        self.y = y_tens

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


dataset = CustomDataset(x_tensor, y_tensor)

# splitting the data into training and validation sets
ratio = 0.8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train
train_data, val_data = random_split(dataset, [n_train, n_val])

# build a Dataloader for each dataset
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)  # powers of 2 for batch size is most common
val_loader = DataLoader(dataset=val_data, batch_size=16)


# ---MODEL CONFIGURATION---
class LinearRegressionModel(torch.nn.Module):
    """
    Creates model class, setting random values for parameters m and c, and defining the forward step function.
    """
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))
        self.c = torch.nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.m * x + self.c


# define the model, loss function, and optimizer that will be passed into the main class. Also define learning rate.
lr = 0.1                                                    # set the learning rate of the model
model = LinearRegressionModel().to(device)                  # linear regression model
loss_fn = torch.nn.MSELoss(reduction='mean')                # mean squared error loss function
optimizer = torch.optim.SGD(model.parameters(), lr=lr)      # defines SGD optimizer to update parameters


class MLModel:
    """
    The main class which will take a model, loss function, and optimizer. The class can train the model, store data,
    save the model, set train and validation loaders, and more.
    """
    def __init__(self, model, loss_fn, optimizer):
        # allows training to be computed on gpu if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # store model, loss function, and optimizer arguments as attributes
        self.model = model
        self.model.to(device)               # send model to the device (gpu if available)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # store data loaders as attributes - starts as None as user will define these with object.set_loaders()
        self.train_loader = None
        self.val_loader = None

        # attributes computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # creates train and val step functions that uses the model, loss_fn, and optimizer arguments
        # these functions take an x feature and y label as arguments and return a loss
        # the model performs a forward pass on the x feature, which is compared with the y label to compute a loss
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def set_loaders(self, train_loader, val_loader=None):
        """
        Takes as arguments data loaders from torch.utils.data.Dataloader, and sets these as attributes.
        Allows user to define their own data loaders.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step_fn(self):
        """
        Returns a function that takes a feature and label, and returns the loss of the model's prediction of
        the feature (x) compared with the label (y).
        But also computes gradients (loss.backward) and updates parameters (self.optimizer.step()). Model training.
        """
        def perform_train_step_fn(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        return perform_train_step_fn

    def _make_val_step_fn(self):
        """
        Returns a function that takes a feature and label, and returns the loss of the model's prediction of
        the feature (x) compared with the label (y).
        """
        def perform_val_step_fn(x, y):
            self.model.eval()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            return loss.item()
        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        """
        Sets data loader and step function as training or validation based on boolean argument.
        Computes and returns a loss from computing an average of mini-batch losses. This is all one epoch.
        """
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)

        return loss

    def train(self, n_epochs):
        """
        :param n_epochs: number of epochs to train.
        :return: does not return, but instead updates model parameters (training).
        """
        for epoch in range(n_epochs):
            # keeps track of number of epochs
            self.total_epochs += 1
            # performs training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # validation
            with torch.no_grad():                                   # no grad necessary so no gradients are computed
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

    def save_checkpoint(self, filename):
        """
        Saves the model as a Python dictionary in a .pth file.
        """
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses}
        # use torch.save() to save the model to a pth file (not designed to be opened, but for a model to load)
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """
        Load a previously saved model.
        """
        # loads dictionary
        checkpoint = torch.load(filename)
        # restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train()

    def predict(self, x):
        """
        Allows the model to predict y labels for given x feature input. Predictions should improve after training.
        """
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()
        # prediction must be detached from the computation graph and sent to the cpu before being converted to numpy
        return y_hat_tensor.detach().cpu().numpy()


# ---MODEL TRAINING---
LRModel = MLModel(model, loss_fn, optimizer)
LRModel.set_loaders(train_loader, val_loader)

print("\nModel parameters before training:")
print(LRModel.model.state_dict())                           # model parameters before training
LRModel.train(n_epochs=200)                                 # training the model
print("\nModel parameters after training:")
print(LRModel.model.state_dict())                           # model parameters after training
print(f"\nReal values: m = {true_m}, c = {true_c}")


# Saving the model (so it keeps its trained parameters etc.)
LRModel.save_checkpoint('LRModel.pth')
