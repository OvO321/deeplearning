import numpy as np
class MyModel(object):
    #两层的权重和偏置
    def __init__(self,sizes):
        #网络层数（+输入层）
        self.num_layers=len(sizes)
        #每层神经元数量
        self.sizes=sizes
        print(sizes)
        #每层的偏置
        self.b1=np.zeros(sizes[1])
        self.b2=np.zeros(sizes[2])
        #每层的权重
        self.w1=np.random.rand(sizes[0],sizes[1])
        self.w2=np.random.rand(sizes[1],sizes[2])

    #前向传播
    def forward(self,x):
        self.z1=np.dot(x,self.w1)+self.b1
        self.a1=self.ReLu(self.z1)
        self.z2=np.dot(self.a1,self.w2)+self.b2
        return self.z2
    
   
    def SGD(self,trained_x,trained_y,epochs,eta,delay,test_data=None):
        #测试数据总量
        #每一轮一次处理一批数据
        for j in range(epochs):
            #print(trained_x)
            predict_y=self.forward(trained_x)
            loss=self.cost(trained_y,predict_y)
            #反向传播更新
            self.backprop(trained_x,trained_y,eta)
            if j%2000==0:
                print(f"Epoch{j} learning-rate:{eta} loss:{loss} complete")
            #学习率衰减
            eta*= (1.0/(1.0+ delay * j))

    def backprop(self,x,y,eta):
        #数据个数
        num=x.shape[0]
        #反向传播开始
        #输出层梯度
        dz2=self.cost_derivative(self.z2,y)
        #计算w2梯度
        dw2=self.a1.T.dot(dz2)/num
        #计算b2梯度
        db2=np.sum(dz2,axis=0)/num
        #反向传播
        dz1 = dz2.dot(self.w2.T) * self.D_Relu(self.z1)  
        #计算w1梯度
        dw1 = x.T.dot(dz1) / num 
        #计算b1梯度
        db1 = np.sum(dz1, axis=0) / num  
        #更新
        self.w2-=eta*dw2
        self.b2-=eta*db2
        self.w1-=eta*dw1
        self.b1-=eta*db1
        

    #损失对激活值的导数，这里使用的是平方差做损失函数
    def cost_derivative(self,x,y):
        return x-y
        pass
    #平均损失值
    def cost(self,x,y):
        return np.mean((x-y)**2)

    #ReLu函数
    def ReLu(self,z):
        return np.maximum(0, z)
    #ReLu函数的导数
    def D_Relu(self,z):
        return z>0
    def ChangeW(self):
        pass
    def ChangeB(self):
        pass
    