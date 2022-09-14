import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

#to make defining the input size easier
imageWidth = 28

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                               ])


#get the training data
trainset = datasets.MNIST('.\\training_set',
                          download=True, train=True, transform=transform)
valset = datasets.MNIST('.\\test_set',
                        download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()


#build the net
input_size = imageWidth ** 2
hidden_sizes = [128, 64]
output_size = 10

#Notes:
# ReLU is essentially a model of how diodes behave
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print()
print(model)


#define negative log-likelihood loss
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

#train the net (NN)
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))

print("\nTraining Time (in minutes) =", (time() - time0) / 60)


#test the NN on the first training image
images, labels = next(iter(valloader))

img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img)


# def view_classify(image, prediction):
#     #for some other time
#
# visualize the network's classification accuracy
# view_classify(img.view(1, 28, 28), ps)


#verify accuracy
correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))


#save the model
torch.save(model, './my_mnist_model.pt')

"""
************************************************************************************************************************
Resources used:
Command for installing PyTorch:                             https://pytorch.org/get-started/locally/
Tutorial:                                                   https://pytorch.org/get-started/locally/
Original creator's GitHub page for the tutorial:            https://github.com/amitrajitbose/handwritten-digit-recognition
MNIST Database:                                             http://yann.lecun.com/exdb/mnist/
Helpful for explaining Negative Log-Likelihood Loss:        https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/


"""