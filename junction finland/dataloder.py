import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
def dataset(path1,path2,path3,path4,path5,path6):
    data = pd.read_csv(path1).values
    data1 = pd.read_csv(path2).values
    data2 = pd.read_csv(path3).values
    data3 = pd.read_csv(path4).values
    data4 = pd.read_csv(path5).values
    data5 = pd.read_csv(path6).values

    train_len = 1280
    test_len = 128
    block_size = 10

    ix = torch.randint(1800, (train_len,))
    x  = torch.stack([torch.from_numpy((data[i:i+block_size,:]).astype(np.float32)) for i in ix])
    x1 = torch.stack([torch.from_numpy((data1[i:i+block_size,:]).astype(np.float32)) for i in ix])
    x2 = torch.stack([torch.from_numpy((data2[i:i+block_size,:]).astype(np.float32)) for i in ix])
    x3 = torch.stack([torch.from_numpy((data3[i:i+block_size,:]).astype(np.float32)) for i in ix])
    x4 = torch.stack([torch.from_numpy((data4[i:i+block_size,:]).astype(np.float32)) for i in ix])
    x5 = torch.stack([torch.from_numpy((data5[i:i+block_size,:]).astype(np.float32)) for i in ix])
    train_data = torch.cat((x,x1,x2,x3,x4,x5),dim=0)
    train_x_reshaped = train_data.unsqueeze(1) 
    x_label0 = torch.zeros(1280,dtype=torch.int64)
    x_label1 = torch.ones(1280,dtype=torch.int64)
    x_label2 = torch.tensor([2],dtype=torch.int64).repeat(1280)
    x_label3 = torch.tensor([3],dtype=torch.int64).repeat(1280)
    x_label4 = torch.tensor([4],dtype=torch.int64).repeat(1280)
    x_label5 = torch.tensor([5],dtype=torch.int64).repeat(1280)
    train_label = torch.cat((x_label0,x_label1,x_label2,x_label3,x_label4,x_label5))
    # print(train_label.shape)
    one_hot_labels = F.one_hot(train_label, num_classes=6)
    # iy = torch.randint(low=1800,high = 1988, size=(test_len,))
    # test  = torch.stack([torch.from_numpy((data[i:i+block_size,:]).astype(np.int64)) for i in iy])
    # test1 = torch.stack([torch.from_numpy((data1[i:i+block_size,:]).astype(np.int64)) for i in iy])
    # test2 = torch.stack([torch.from_numpy((data2[i:i+block_size,:]).astype(np.int64)) for i in iy])
    # test3 = torch.stack([torch.from_numpy((data3[i:i+block_size,:]).astype(np.int64)) for i in iy])
    # test_data = torch.cat((test,test1,test2,test3),dim=0)
    # test_label0 = torch.zeros(128,1)
    # test_label1 = torch.ones(128,1)
    # test_label2 = torch.tensor([[2]]).repeat(128,1)
    # test_label3 = torch.tensor([[3]]).repeat(128,1)
    # test_label = torch.cat((test_label0,test_label1,test_label2,test_label3))
    # print(train_x_reshaped.shape,one_hot_labels.shape)
    return train_x_reshaped,one_hot_labels


if __name__ == "__main__":
    dataset('label0.csv','label1.csv','label2.csv','label3.csv','label4.csv')
    