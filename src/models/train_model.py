import argparse
import sys

import torch
import click

from model import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_sets = []
    for i in range(5):
        train = np.load("data/processed/" + f"train_{i}.npz")
        train_sets.append(torch.utils.data.TensorDataset(torch.tensor(train["images"]).float(), torch.tensor(train["labels"])))
        
    
    test = np.load("data/processed/" + "test.npz")
    
    test_set = torch.utils.data.TensorDataset(torch.tensor(test["images"]).float(), torch.tensor(test["labels"]))
    train_set = torch.utils.data.ConcatDataset(train_sets)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=True)

    epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.NLLLoss()
    steps = 0
    running_loss = 0
    print_every = 60
    losses_train = []
    losses_test = []
    accuracys = []
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
        accuracy = 0
        loss = 0
        for images, labels in testloader:
            
            output = model(images)
            loss += criterion(output, labels).item()

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()
        losses_test.append(loss/len(testloader))
        accuracys.append(accuracy/len(testloader))

    torch.save(model, 'src/models/checkpoints/checkpoint.pth')
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



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    test_set = np.load("data/processed/test_data.npz")
    accuracy = 0
    for images, labels in test_set:
        
        output = model(images)

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    accuracy /= len(test_set)
    print(accuracy.item())


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
