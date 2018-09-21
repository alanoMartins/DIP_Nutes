import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=5)
        self.conv2 = nn.Conv2d(50, 20, kernel_size=5)
        self.fc1 = nn.Linear(7200, 200)
        self.fc2 = nn.Linear(200, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print(x.shape)
        x = x.view(-1, 7200)


        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

def save_model(model, filename="mnist_weights.pth"):
    torch.save({'state_dict': model.state_dict()}, filename)

def load_model(model, filename="mnist_weights.pth"):
    state_dict = torch.load(filename)['state_dict']
    model.load_state_dict(state_dict)

#SGD
def train(model, image, label, optimizer):
    output = model(image)
    # negative log likelihood loss
    loss = F.nll_loss(output, label)

    optimizer.zero_grad()
    loss.backward()
        
    optimizer.step()

    return loss