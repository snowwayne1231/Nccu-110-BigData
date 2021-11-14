import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

criterion = nn.CrossEntropyLoss()



class BasicNet(nn.Module):

    criterion = None
    optimizer = None

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    def setting(self):
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()


    def train(self, datasets, lables, epoch = 2):
        _optimizer = self.optimizer
        _criterion = self.criterion

        for _e in range(epoch):

            running_loss = 0.0
            for i, data in enumerate(datasets, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                _optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = _criterion(outputs, labels)
                loss.backward()
                _optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
