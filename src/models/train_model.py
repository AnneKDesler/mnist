import argparse
import sys

import torch
import click

from model import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

import wandb
from omegaconf import OmegaConf

def train(cfg):
    print("Training day and night")

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_sets = []
    for i in range(5):
        train = np.load("data/processed/" + f"train_{i}.npz")
        train_sets.append(torch.utils.data.TensorDataset(torch.tensor(train["images"]).float(), torch.tensor(train["labels"])))
        
    
    test = np.load("data/processed/test.npz")
    
    hyps = cfg.hyperparameters
    lr = hyps.lr
    batch_size = hyps.batch_size
    epochs = hyps.epochs

    test_set = torch.utils.data.TensorDataset(torch.tensor(test["images"]).float(), torch.tensor(test["labels"]))
    train_set = torch.utils.data.ConcatDataset(train_sets)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    steps = 0
    running_loss = 0
    print_every = 60
    losses_train = []
    losses_test = []
    accuracys = []

    wandb.init(config=cfg)

    wandb.watch(model, log_freq=100)

    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        print("Epoch: {}/{}.. ".format(e+1, epochs))

        for images, labels in trainloader:
            steps += 1
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            losses_train.append(loss.item())
            wandb.log({"train loss": loss.item()})
        accuracy = 0
        loss = 0
        for images, labels in testloader:
            
            output = model(images)
            loss_i = criterion(output, labels).item()
            loss += loss_i

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy_i = equality.type_as(torch.FloatTensor()).mean()
            accuracy += accuracy_i
            wandb.log({"test loss": loss_i})
            wandb.log({"accuracy": accuracy_i})

        losses_test.append(loss/len(testloader))
        accuracys.append(accuracy/len(testloader))

    torch.save(model, 'models/checkpoints/checkpoint.pth')
    fig = plt.figure()
    plt.plot(losses_train)
    plt.ylabel('Train loss')
    plt.xlabel('Steps')
    plt.show()
    fig.savefig("reports/figures/train_loss.png")

    fig = plt.figure()
    plt.plot(losses_test)
    plt.ylabel('Test loss')
    plt.xlabel('Epochs')
    plt.show()
    fig.savefig("reports/figures/test_loss.png")

    fig = plt.figure()
    plt.plot(accuracys)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()
    fig.savefig("reports/figures/accuracy.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help = 'config file path')
    args =parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
