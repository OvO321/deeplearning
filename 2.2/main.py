import model
import numpy as np
import matplotlib.pyplot as plt

#定义了被测试的函数
def tested_fun(x):
    """
    输入参数为np向量
    """
    #print("x=",x)
    return np.exp(x)+np.cos(x)

#初始化模型，分别为各层的神经元数量
model=model.MyModel([1,50,1])

#产生训练集数据
#训练集不要过多，不然训练时间太长
trained_x=np.linspace(-2*np.pi,2*np.pi,1000).reshape(-1,1)
trained_y=tested_fun(trained_x)


#训练模型
model.SGD(trained_x=trained_x,trained_y=trained_y,epochs=20000,eta=0.0015,delay=1e-12)

#print(trained_data)

#产生测试集数据
test_x=np.linspace(-2*np.pi,2*np.pi,500).reshape(-1,1)
test_y=tested_fun(test_x)


# 使用模型进行预测
predict_y=model.forward(test_x)


#结果显示
plt.figure(figsize=(8, 5))
plt.plot(test_x,test_y, label='raw function')  
plt.plot(test_x,predict_y , label='predict function', linestyle='--')  
plt.legend()
plt.title("Compare")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
