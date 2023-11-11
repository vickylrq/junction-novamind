import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


#数据预处理
def process_data(type):
    #读入数据
    if type=='train':
        data=pd.read_csv('./train.csv')
    elif type=='val':
        data=pd.read_csv('./val.csv')

    # 以闭市价作为股价构成标签
    price=data.loc[:,'close']
    #归一化处理，便于后期加速模型收敛
    price_norm=price/max(price)

    #定义etract_data函数，能够将数据转换为时间序列
    def etract_data(data, time_step):
        X,y = [],[]
        for i in range(len(data) - time_step):
            X.append([a for a in data[i:i + time_step]])
            y.append([data[i + time_step]])
        X = torch.tensor(X)
        X = X.reshape(-1, time_step, 1)
        y = torch.tensor(y)
        y = y.reshape(-1,1)
        return X, y

    #时间序列时间步长为8，通过前8个数据预测第9个
    time_step = 8
    batch_size = 1
    X,y=etract_data(price_norm,time_step)

    #构造迭代器，返回
    dataset=TensorDataset(X,y)
    #训练集数据随机打乱，验证集保持不变
    if type=='train':
        shuffle=True
    else:
        shuffle=False
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader
dataloader={'train':process_data('train'),'val':process_data('val')}


#设计网络(单隐藏层Rnn)
input_size,hidden_size,output_size=1,20,1
#Rnn初始隐藏单元hidden_prev初始化
hidden_prev=torch.zeros(1,1,hidden_size).cuda()
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.rnn=nn.RNN(
            input_size=input_size,    #输入特征维度,当前特征为股价，维度为1
            hidden_size=hidden_size,  #隐藏层神经元个数，或者也叫输出的维度
            num_layers=1,
            batch_first=True
        )
        self.linear=nn.Linear(hidden_size,output_size)

    def forward(self,X,hidden_prev):
        out,ht=self.rnn(X,hidden_prev)
        # out[batch_size=1,time_step=8,hidden_size=20]
        # ht[num_layer=1,batch_size=1,hidden_size=20]
        ht = ht.view(-1, hidden_size)   # ht[1,hidden_size=20]
        ht=self.linear(ht)              #ht[1,1]
        return out,ht


#设定超参数，训练模型
model=Net()
model=model.cuda()
criterion=nn.MSELoss()
learning_rate,epochs=0.01,500
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

best_loss=9999
for epoch in range(epochs):
    for phase in ['train','val']:
        if phase == 'train':
            model.train()
        elif phase == 'val':
            model.eval()

        losses = []
        for X,y in dataloader[phase]:
            X = X.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            _,yy=model(X,hidden_prev)
            yy=yy.cuda()

            loss = criterion(y, yy)
            model.zero_grad()

            if phase=='train':
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

        epoch_loss=sum(losses)/len(losses)

        if phase=='val' and epoch_loss<best_loss:
            best_loss=epoch_loss
            torch.save(model.state_dict(), "model.pth")

        if epoch%50==0:   #保留验证集损失最小的模型参数
            print("epoch:{},{} loss:{:.8f}".format(epoch+1,phase,epoch_loss))
            if phase=='val':
                print("the best loss of valuation is:{:.8f}".format(best_loss))
                print('*'*50)

#加载模型，绘图查看模型效果
model.load_state_dict(torch.load('model.pth'))
Val_y,Val_predict=[],[]
#将归一化后的数据还原
Val_max_price=max(pd.read_csv('./val.csv').loc[:,'close'])
for X,y in dataloader['val']:
    with torch.no_grad():
        X = X.cuda()
        _,predict=model(X,hidden_prev)
        y=y.cpu()
        predict=predict.cpu()
        Val_y.append(y[0][0]*Val_max_price)
        Val_predict.append(predict[0][0]*Val_max_price)

fig=plt.figure(figsize=(8,5),dpi=80)
# 红色表示真实值，绿色表示预测值
plt.plot(Val_y,linestyle='--',color='r')
plt.plot(Val_predict,color='g')
plt.title('stock price')
plt.xlabel('time')
plt.ylabel('price')
plt.show()
