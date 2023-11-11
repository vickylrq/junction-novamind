import torch
import pandas as pd
import numpy as np
from model import SimpleCNN2D

input_data = pd.read_csv("data.csv").values
input_data = torch.from_numpy(input_data[np.newaxis,:]).float()
# print(input_data.shape)
model = SimpleCNN2D(4)
model.load_state_dict(torch.load('cnn_model_2.pth'))
pred = torch.argmax(model(input_data)).numpy()
print(pred)
