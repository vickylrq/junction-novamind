import torch
import pandas as pd
import numpy as np
from model import SimpleCNN2D

input_data = pd.read_csv("label3.csv").values
for i in range(1800,1990):
    input_data0 = input_data[i:i+10,:]
    input_data_in = torch.from_numpy(input_data0[np.newaxis,:]).float()
    # print(input_data.shape)
    model = SimpleCNN2D(4)
    model.load_state_dict(torch.load('cnn_model_3.pth'))
    pred = torch.argmax(model(input_data_in)).numpy()
    print(pred)
