import glob
import os
import numpy as np
import cv2

def get_image(file):
    img = cv2.imread(file)
    # img_data = 255.0 - img_array.reshape(784)
    # img_data = (img_data/255.0*0.99 )+0.01
    return img

def get_data(directory='mnist/data/testing/'):
    numbers_dir = [x[0] for x in os.walk(directory)][1:]
    data = []
    for n in numbers_dir:
        files_dir = glob.glob(n+"/*.png")
        l = n.split('/')[-1]
        # print(files_dir)
        for file in files_dir:
            img_data = get_image(file)
            inputs2 = int(l)

            data.append([ img_data, inputs2 ])
    return np.array(data)
