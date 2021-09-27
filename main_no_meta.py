'''
输入:两个400维度的拼接，
NN的结构:后面的三层的维度分别是:2000 800 400
输出:就是关系
损失函数就是h + r - t得到的结果
'''

import torch.nn.functional as F
import torch.nn.init as init
import torch
from torch.autograd import Variable
import matplotlib.pyplot as  plt
import numpy as np
import math
import json
# %matplotlib inline
# #%matplotlib inline 可以在Ipython编译器里直接使用
# #功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。
#-----------------------parameters--------------------------
filepath = '../MetaDataPrepro/HW/test_tasks.json'
test_target_name = "DGIDB::ANTIBODY::Gene:Compound" #需要训练的目标，目标从test_tasks中获取
#-----------------------------------------------------------
with open(filepath, 'r', encoding='utf-8') as f:
    xy = json.load(f)

test_target = xy[test_target_name] #从一个对象中抽取出我们想要的训练对象的数据

#载入embedding数据
all_embedding = np.load('../MetaDataPrepro/HW/ent2vec.npy') #载入embedding
ent2id = np.load('../MetaDataPrepro/HW/ent2vec.npy') #载入结点对应的id，后面通过id找embedding

# xy=np.loadtxt('../MetaDataPrepro/HW/dev_tasks.json',delimiter=',',dtype=np.float32)
# x_data=torch.from_numpy(xy[:,0:-1])#取除了最后一列的数据
# y_data=torch.from_numpy(xy[:,[-1]])#取最后一列的数据，[-1]加中括号是为了keepdim


# print(x_data.size(),y_data.size())
# print(x_data.shape,y_data.shape)

#建立网络模型
class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model,self).__init__()
        self.l1=torch.nn.Linear(8,6)
        self.l2=torch.nn.Linear(6,4)
        self.l3=torch.nn.Linear(4,1)
        
    def forward(self,x):
        out1=F.relu(self.l1(x))
        out2=F.dropout(out1,p=0.5)
        out3=F.relu(self.l2(out2))
        out4=F.dropout(out3,p=0.5)
        y_pred=F.sigmoid(self.l3(out3))
        return y_pred
    
def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Linear')!=-1:
        m.weight.data=torch.randn(m.weight.data.size()[0],m.weight.data.size()[1])
        m.bias.data=torch.randn(m.bias.data.size()[0])



#我自己的模型
model=Model()
model.apply(weights_init)
criterion=torch.nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

#training loop
Loss=[]
for epoch in range(2000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    if epoch%100 == 0:
        print("epoch = ",epoch," loss = ",loss.data)
        Loss.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
hour_var = Variable(torch.randn(1,8))
print("predict",model(hour_var).data[0]>0.5)
plt.plot(Loss)
