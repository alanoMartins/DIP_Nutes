import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)

activation =  F.relu

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
#trying to reproduce vanishing gradients
        self.fc1 = nn.Linear(10, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 1000)
        self.fc7 = nn.Linear(1000, 1000)
        self.fc8 = nn.Linear(1000, 1000)
        self.fc9 = nn.Linear(1000, 1000)
        self.fc10 = nn.Linear(1000, 1000)
        
        self.fc11 = nn.Linear(1000, 2)

    def forward(self, x):
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = activation(self.fc3(x))
        x = activation(self.fc4(x))
        x = activation(self.fc5(x))
        x = activation(self.fc6(x))
        x = activation(self.fc7(x))
        x = activation(self.fc8(x))
        x = activation(self.fc9(x))
        x = activation(self.fc10(x))
        
        x = self.fc11(x)

        return F.log_softmax(x)
        # pass
    
net = NeuralNetwork()

criterion = F.nll_loss

y = torch.tensor([1], dtype=torch.long)
x = torch.tensor([[10,4,63,1,85,0,7,36,46,11]], dtype=torch.float)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

for i in range(10):

    output = net(x) 

    optimizer.zero_grad()
    loss = criterion(output, y) 
    loss.backward()
    optimizer.step()

    print("epoch: {}, error: {}".format(i, loss))

    
