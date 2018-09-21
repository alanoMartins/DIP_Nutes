import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import ConvNet as cnn
import utils

                #  i  l 
# print(get_data()[0][0])

model = cnn.ConvNet()
lr = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# cnn.load_model(model, filename="mnist_weights.pth")

data = utils.get_data(directory='./data/testing/')#too many samples on training folder

for epoch in range(10):
    print("<-----------Epoch: {}".format(epoch))
    np.random.shuffle(data)
    for i, d in enumerate(data):
        image, label = d
        image = torch.tensor([image], dtype=torch.float).permute(0, 3, 1, 2)
        label = torch.tensor([label], dtype=torch.long)
        loss = cnn.train(model, image, label, optimizer)

        if(i%100 == 0):
            print("image {}/{}, loss: {}".format(i, len(data), loss))
            # cnn.save_model(model)

# print(output)