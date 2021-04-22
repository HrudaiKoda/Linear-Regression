import csv
import numpy as np
import matplotlib.pyplot as plt
def load_dataset():
    dataset = list()
    with open('data.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    dataset = dataset.astype("float")
    x_row = dataset[:,0]
    y_row = dataset[:,1]
    print(dataset.shape)
    return x_row,y_row
def loss_function(x,y,m,c):
    pred = m*x + c
    loss = y - pred
    final_loss = sum(loss**2)/len(y)
    return final_loss
def descent(x,y,m,c):
    pred = m*x + c
    m_des = sum((y-pred)*(-2*x))/len(x)
    c_des = sum(y-pred)*(-2/len(y))
    return m_des,c_des
def train(x,y):
    m=0
    c=0
    lr = 0.0001
    epochs = 1000
    #Used 0.01 for data_1.csv and 1500 epochs 
    loss = list()
    for i in range(epochs):
        loss.append(loss_function(x,y,m,c))
        m_new , c_new = descent(x,y,m,c)
        m = m - (lr*m_new)
        c = c - (lr*c_new)
    loss = np.array(loss)
    #print(loss)
    #x = np.arange(0,len(loss))
    Y_pred = x*m + c
    plt.scatter(x,y,marker = "x")
    plt.plot([min(x), max(x)], [min(Y_pred), max(Y_pred)], color='red')
    #plt.plot(x,loss)
    plt.savefig("Plot.png")
    plt.show()
x,y = load_dataset()
train(x,y)
