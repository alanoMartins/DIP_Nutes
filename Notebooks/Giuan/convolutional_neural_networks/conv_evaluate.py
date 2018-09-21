import torch
import numpy as np

import ConvNet as cnn
import utils


model = cnn.ConvNet()
cnn.load_model(model, filename="mnist_weights.pth")

# for param in model.conv1.parameters():
#     print(param.shape)

img = utils.get_image("./data/training/4/342.png")

image = torch.tensor([img], dtype=torch.float).permute(0, 3, 1, 2)

output = model(image)

element, index = torch.max(output[0], 0)
print(index)