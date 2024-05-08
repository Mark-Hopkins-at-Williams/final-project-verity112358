import numpy as np
import torch
from torch import tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets
from torch.nn.utils import prune
from copy import deepcopy


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_layer_size = 500
        self.theta1 = Parameter(torch.empty(hidden_layer_size, 28*28))
        self.theta2 = Parameter(torch.empty(hidden_layer_size, hidden_layer_size))
        self.theta3 = Parameter(torch.empty(hidden_layer_size, hidden_layer_size))
        self.theta4 = Parameter(torch.empty(hidden_layer_size, hidden_layer_size))
        self.theta_final = Parameter(torch.empty(10, hidden_layer_size))
        for param in self.parameters():
            torch.nn.init.kaiming_uniform_(param)

    def forward(self, x):
        result = torch.matmul(self.theta1, x.t())
        result = torch.relu(result)
        result = torch.matmul(self.theta2, result)
        result = torch.relu(result)
        result = torch.matmul(self.theta3, result)
        result = torch.relu(result)
        result = torch.matmul(self.theta4, result)
        result = torch.relu(result)
        result = torch.matmul(self.theta_final, result)
        result = torch.softmax(result, dim=0)
        return result
    

def minibatch_loss(net, X, y):
    predictions = net.forward(X)
    probs = torch.gather(predictions, dim=0, index=y.unsqueeze(0))
    probs = probs.clamp(min=0.00000001, max=0.99999999)
    losses = -torch.log(probs)
    loss = torch.mean(losses)
    return loss


def minibatch_gd(model, num_epochs, train_set, test_set, lr=0.01):

    """
    Performs minibatch gradient descent on a model. Also saves a dictionary that has
    k: epoch number, v: accuracy at epoch number k
    """

    accuracy_dict = dict()
    for i in range(num_epochs):    
        train_loader = DataLoader(train_set, batch_size=32)
        for X, y in tqdm(train_loader):
            loss = minibatch_loss(model, X, y)
            loss.backward()
            for param in model.parameters():
                with torch.no_grad():           
                    param -= lr*param.grad
                    param.grad = None
        test_loader = DataLoader(test_set, batch_size=128)
        accuracy = evaluate(model, test_loader)
        print(f"Accuracy: {accuracy}")
        accuracy_dict[i] = accuracy
    
    return accuracy_dict

def evaluate(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    for X, y in tqdm(test_loader):
        predictions = net.forward(X)
        preds = torch.max(predictions, 0)
        correct += torch.sum(preds.indices == y).item()
        total += torch.numel(y)
    net.train()    
    return correct/total


def load_mnist():
    def image_to_tensor(img):
        t = tensor(np.asarray(img)).flatten().float()
        return (t - 127) / 255
    mnist_train_raw = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_test_raw = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    mnist_train = [(image_to_tensor(t[0]), t[1]) for t in mnist_train_raw]
    mnist_test = [(image_to_tensor(t[0]), t[1]) for t in mnist_test_raw]
    return mnist_train, mnist_test

def winning_ticket(trained_network: NeuralNetwork, copy_network: NeuralNetwork, amount: float):

    '''
    Function that takes a trained network and a previously made copy of that network
    and prunes away the lowest valued parameter weights by l1 norm. The copy network is
    modified in place to be a "winning ticket," i.e. it retains the random initialization
    values that later became highest magnitude in the trained network.
    '''

    parameters = [p[0] for p in trained_network.named_parameters()]

    print("PRUNING TRAINED NETWORK")
    for p in parameters:
        prune.l1_unstructured(trained_network, p, amount)
    
    masks = [m for m in trained_network.named_buffers()]

    for m in masks:
        print("APPLYING MASK TO " + m[0][:-5])
        prune.custom_from_mask(copy_network, m[0][:-5], m[1])

    print("FINALIZING PRUNING")
    for m in masks:
        prune.remove(copy_network, m[0][:-5])
    
    print("NEW NETWORK")
    print(list(copy_network.named_parameters()))

def run_experiment(pruning_weights: list, epochs: int):

    training_network = NeuralNetwork()
    tickets = []

    for _ in pruning_weights:
        tickets.append(deepcopy(training_network))

    train_set, test_set = load_mnist()

    training_dict = minibatch_gd(training_network, epochs, train_set, test_set)

    ticket_dicts = []

    for weight, ticket in zip(pruning_weights, tickets):

        temp_trained_network = deepcopy(training_network)

        winning_ticket(temp_trained_network, ticket, weight)

        ticket_dicts.append(minibatch_gd(ticket, epochs, train_set, test_set))
    
    
    return training_dict, ticket_dicts





class MNistClassifier:
    def __init__(self, filename):
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
    
    def classify(self, img):
        predictions = self.model.forward(img.flatten())
        return predictions


if __name__ == "__main__":
    train_set, test_set = load_mnist()
    num_epochs = 10
    model = NeuralNetwork()
    minibatch_gd(model, num_epochs, train_set, test_set)
    torch.save(model.state_dict(), "mnistmodel.pt")
    