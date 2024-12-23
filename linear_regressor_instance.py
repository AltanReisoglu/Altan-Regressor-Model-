import torch 
import numpy as np
import torch.nn as nn
import sklearn.datasets as datasets
import matplotlib.pyplot as plt

x_train,y_train=datasets.make_regression(n_samples= 100, n_features=1,noise=20)
x=torch.from_numpy(x_train.astype("float32"))
y=torch.from_numpy(y_train.astype("float32"))
y=y.view(y.shape[0],1)
n_samples,n_features=x.shape
x_test=torch.tensor(([7]),dtype=torch.float32)

class AltanRegressor():
    def __init__(self,x,y):
        self.model=nn.Linear(n_features,n_features)

        self.lr=0.001
        self.epoch=200
        self.loss=nn.MSELoss()
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=self.lr)
    def predict(self,deger):
        return self.model(deger)
    
    def draw(self,deger):
        y_pred=self.model(deger).detach().numpy()
        plt.plot(x_train,y_pred)
        plt.scatter(x_train,y_train)
        plt.show()
    
    def __call__(self, *args, **kwds):
        for i in range(200):
            y_pred=self.model(x)
            loss=self.loss(y,y_pred)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if i % 1 == 0:
                [self.weight,self.b]=self.model.parameters()
                print(f"Epoch {i+1}: Weight = {self.weight[0][0].item()}, Loss = {loss.item()}")

rgr=AltanRegressor(x,y)
rgr()
rgr.draw(x)

